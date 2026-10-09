#!/usr/bin/env python3
"""Plan by default; reviewed guard+contract authorization precede native/OV runs."""
from __future__ import annotations
import argparse, hashlib, importlib.util, json, math, os, queue, struct, subprocess, sys, threading, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))
CONTRACT = Path(__file__).with_name("execution-contract.json")
CAP = 16 * 1024**2


def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""): digest.update(block)
    return digest.hexdigest()


def document(path):
    with path.open("rb") as stream: raw = stream.read((2 << 20) + 1)
    if len(raw) > 2 << 20: raise ValueError("metadata document exceeds bound")
    return json.loads(raw.decode("utf-8-sig"))


def inspect(c):
    if c.get("schema_version") != 1 or c["source_revision"] != "3cf03257f219afbe7334045ff7c6a06ac68c627d":
        raise ValueError("contract/source identity differs")
    if len(c["fixtures"]) != 4 or len(c["selectors"]) != 11 or c["protocol"]["formal_repeats"] != 3:
        raise ValueError("fixture/selector/repeat contract differs")
    if c["protocol"]["per_image_tap_byte_cap"] != CAP or c["resource_gate"] != {
            "physical_bytes": 12 * 2**30, "commit_bytes": 4 * 2**30,
            "interval_seconds": 1, "known_telemetry_required": True}:
        raise ValueError("fixed capture/resource gates differ")
    if "--gpu" in c["helper_argv"] or c["helper_argv"][-1] != "--enable-stage-taps":
        raise ValueError("only opt-in CPU native stage argv is permitted")
    if c["helper_argv"][c["helper_argv"].index("--flash-attn") + 1] != "off":
        raise ValueError("frozen flash-attn-off reference is required")
    for name, digest in c["code_and_build_sha256"].items():
        if sha(ROOT / name) != digest: raise ValueError("bound code/build receipt changed: " + name)
    for fixture in c["fixtures"]:
        if (fixture["height"], fixture["width"]) not in ((96, 96), (96, 192)):
            raise ValueError("unselected static fixture shape")
        if set(fixture["frozen_ov_decoded_sha256"]) != {"graph_f32", "cpu_vec_dot_rounding"}:
            raise ValueError("both frozen policy fingerprints are required")
    return {"schema_version": 1, "status": "plan_only", "execution_authorized": c["execution_authorized"],
        "native_commands": 20, "baseline_native_checks": 4, "warmups": 4, "formal_tap_captures": 12,
        "selectors": list(c["selectors"]), "max_tap_bytes_per_image": CAP,
        "helper_started": False, "gguf_read": False, "core_created": False,
        "guard_binding": c["owned_process_guard"], "ov_fingerprint_changes_are_explicit": True}


def require_frozen_sve(actual, expected):
    if actual["sha256"] != expected["sha256"]: raise ValueError("native final SVE differs from frozen run23 full-byte SHA")
    for key in ("n_tokens", "nx", "ny", "n_embd", "size_bytes"):
        if actual[key] != expected[key]: raise ValueError("native SVE layout differs: " + key)


def validate_manifest(prefix, fixture, c, ordinal, previous_bytes, expected_sve=None):
    prefix = prefix.resolve(strict=True)
    manifest = document(prefix / "manifest.json")
    n = fixture["height"] // 16 * (fixture["width"] // 16)
    if (manifest.get("schema_version"), manifest.get("status"), manifest.get("source_revision")) != (
            1, "stages_captured", c["source_revision"]): raise ValueError("native stage manifest identity/status differs")
    if Path(manifest["image"]).resolve() != Path(fixture["path"]).resolve() or manifest["capture_number"] != ordinal:
        raise ValueError("manifest image or capture ordinal differs")
    if expected_sve is not None and Path(manifest["sve_output"]).resolve() != expected_sve.resolve():
        raise ValueError("manifest names a different final SVE output")
    stages = manifest["stages"]
    if len(stages) != 11 or {s["stage"] for s in stages} != set(c["selectors"]):
        raise ValueError("duplicate/missing stage coverage")
    total = 0
    for stage in stages:
        name = stage["stage"]
        features, rows = (3456, n) if name == "qkv.0" else ((4608, n//4) if name == "merged" else (1152, n))
        ne = [features, rows, 1, 1]; nb = [4, features*4, features*rows*4, features*rows*4]
        if (stage["selector"], stage["actual_dtype"], stage["type_id"], stage["ne"], stage["nb"], stage["bytes"]) != (
                c["selectors"][name], "F32", 0, ne, nb, features*rows*4):
            raise ValueError("stage dtype/ne/nb/source selector differs: " + name)
        selected_ne, selected_nb = ([72,16,n,1],[4,288,13824,13824*n]) if name == "qkv.0" else (ne,nb)
        if stage["selector_ne"] != selected_ne or stage["selector_nb"] != selected_nb:
            raise ValueError("selected native view axes/strides differ")
        declared = Path(stage["path"])
        path = declared.resolve(strict=True)
        if path != prefix / (name + ".f32") or declared.is_symlink() or path.stat().st_size != stage["bytes"]:
            raise ValueError("stage file outside owned prefix or wrong size")
        if total + stage["bytes"] > CAP or previous_bytes + total + stage["bytes"] > CAP:
            raise ValueError("stage payload aggregate exceeds unchanged 16 MiB cap")
        if sha(path) != stage["raw_sha256"]: raise ValueError("native raw stage SHA differs")
        with path.open("rb") as stream: raw = stream.read(stage["bytes"] + 1)
        if len(raw) != stage["bytes"]: raise ValueError("stage bytes changed during read")
        if any(not math.isfinite(v[0]) for v in struct.iter_unpack("<f", raw)):
            raise ValueError("native stage contains NaN or infinity")
        total += stage["bytes"]
    if (total, previous_bytes+total, CAP) != (manifest["capture_bytes"], manifest["image_total_bytes"], manifest["image_byte_cap"]):
        raise ValueError("manifest aggregate byte ledger differs")
    return manifest, total


def compare_ov_fingerprints(receipt, c):
    changes = []
    expected = {f["name"]: f["frozen_ov_decoded_sha256"] for f in c["fixtures"]}
    for arm in receipt["arms"]:
        for image in arm["images"]:
            for formal in image["formal"]:
                old = expected[image["fixture"]][arm["policy"]]
                now = formal["output"]["decoded_sha256"]
                changes.append({"policy": arm["policy"], "fixture": image["fixture"], "iteration": formal["iteration"],
                    "frozen_run31_sha256": old, "diagnostic_sha256": now, "changed": old != now})
    if len(changes) != 24: raise ValueError("OV diagnostic formal coverage incomplete")
    return {"status": "diagnostic_only", "projection_changed_from_frozen_run31": any(r["changed"] for r in changes),
            "rows": changes, "native_reference_replaced": False, "numerical_gate_relaxed": False}


def module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    loaded = importlib.util.module_from_spec(spec); sys.modules[name] = loaded; spec.loader.exec_module(loaded)
    return loaded


def guard_api(c):
    binding = c["owned_process_guard"].get("reviewed_binding")
    if not isinstance(binding, dict) or binding.get("root_reviewed") is not True:
        raise ValueError("execution withheld: final shared guard module/SHA has not been root-bound")
    path = ROOT / binding["module_file"]
    if sha(path) != binding["sha256"]: raise ValueError("reviewed shared guard SHA changed")
    api = module(path, "vision_reviewed_guard")
    for required in ("OwnedServiceTree", "ResourceMonitor", "BenchError", "utc", "_process_record"):
        if not hasattr(api, required): raise ValueError("reviewed guard lacks required API")
    return api


def verify_native_root(tree, c):
    if tree.root_record["command_line"] != c["helper_argv"]:
        raise ValueError("native direct root full argv differs from bound helper_argv")
    return {"full_argv_verified": True, "identity": tree.root_record}


class DirectOwner:
    def __init__(self, proc): self.proc, self.pid = proc, proc.pid
    @property
    def returncode(self): return self.proc.poll()
    def close(self):
        if self.proc.poll() is None:
            self.proc.terminate()
            try: self.proc.wait(timeout=3)
            except subprocess.TimeoutExpired: self.proc.kill(); self.proc.wait(timeout=3)


def native_run(c):
    if c["execution_authorized"] is not True: raise ValueError("contract execution is not authorized")
    shared = guard_api(c)  # Resolve approved process safety before model reads or launches.
    import psutil
    from tools.hetero_verify_existing import fd_matches_path, file_stat, stat_unchanged
    oracle = module(ROOT / "bench/hetero/20261009-22-vision-oracle-cpu/oracle_client.py", "vision_frozen_protocol")
    out = Path(c["output_root"]); out.mkdir(parents=True, exist_ok=False)
    journal = (out / "progress.jsonl").open("x", encoding="utf-8")
    record = {"schema_version":1,"status":"running","results":[],"helper_started":False,"core_created":False}
    owner = tree = monitor = proc = stderr = None
    def event(name, **fields):
        journal.write(json.dumps({"utc":shared.utc(),"event":name,**fields})+"\n");journal.flush();os.fsync(journal.fileno())
    try:
        record["controller_identity"]=shared._process_record(psutil.Process(os.getpid()))
        monitor = shared.ResourceMonitor(out/"resource.jsonl", psutil, minimum_physical=12*2**30, minimum_commit=4*2**30)
        monitor.start()
        deadline = time.monotonic()+5
        while True:
            try: monitor.require(); break
            except shared.BenchError:
                if monitor.failed is not None or time.monotonic()>=deadline: raise
                time.sleep(.05)
        for item in (c["helper"], c["mmproj"], c["text_model"]):
            path = Path(item["path"]); before = file_stat(path)
            monitor.require()
            expected_bytes=item.get("size_bytes",item.get("bytes"))
            if before["size_bytes"] != expected_bytes: raise ValueError("bound helper/model byte count changed")
            if item is c["text_model"] and (before["mtime_ns"],before["ctime_ns"],before["file_id"]["volume"],before["file_id"]["index"]) != (
                    item["mtime_ns"],item["ctime_ns"],item["volume"],item["file_index"]):
                raise ValueError("frozen text GGUF1 stat identity changed")
            digest=hashlib.sha256()
            with path.open("rb",buffering=0) as stream:
                if not fd_matches_path(os.fstat(stream.fileno()),before): raise ValueError("opened input differs from checked path")
                for block in iter(lambda:stream.read(8<<20),b""): digest.update(block)
                if not fd_matches_path(os.fstat(stream.fileno()),before): raise ValueError("opened input changed during hash")
            if digest.hexdigest() != item["sha256"] or not stat_unchanged(before,file_stat(path)):
                raise ValueError("stable helper/model identity mismatch")
            record.setdefault("input_identities",[]).append({**item,"current_stat":before})
        for fixture in c["fixtures"]:
            if sha(Path(fixture["path"])) != fixture["sha256"]: raise ValueError("fixture bytes changed")
        monitor.require(); stderr = (out/"helper.stderr.log").open("xb")
        proc = subprocess.Popen(c["helper_argv"],stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=stderr,
            shell=False,creationflags=getattr(subprocess,"CREATE_NO_WINDOW",0)|getattr(subprocess,"CREATE_NEW_PROCESS_GROUP",0),bufsize=0)
        owner = DirectOwner(proc); oracle.idle(proc)
        tree = shared.OwnedServiceTree(owner,psutil,Path(c["helper"]["path"]),Path(c["helper"]["path"]),
                                      Path(c["helper"]["path"]),c["helper_argv"][1:])
        record["native_direct_root_binding"] = verify_native_root(tree,c)
        monitor.attach(owner,tree)  # Native root is the actual runtime; never await a Cpython child here.
        record.update(helper_started=True,pid=proc.pid,created_utc=oracle.process_created_utc(proc),argv=c["helper_argv"],tree=tree.snapshot())
        event("owned_native_started",identity=record["tree"])
        lines = queue.Queue(maxsize=8); threading.Thread(target=oracle.reader,args=(proc.stdout,lines),daemon=True).start()
        if oracle.read_line(lines,180) != c["protocol"]["ready"]: raise ValueError("native READY differs")
        def encode(fixture, label, taps):
            monitor.require(); sve=out/(fixture["name"]+"-"+label+".sve")
            if sve.exists(): raise FileExistsError(sve)
            prefix=out/(fixture["name"]+"-"+label+"-taps")
            command=f"ENC_TAPS {fixture['path']} {sve} {prefix}" if taps else f"ENC {fixture['path']} {sve}"
            event("command",command=command); replies=queue.Queue(maxsize=1)
            def exchange():
                try: proc.stdin.write((command+"\n").encode());proc.stdin.flush();replies.put(oracle.read_line(lines,120))
                except BaseException as exc: replies.put(exc)
            threading.Thread(target=exchange,daemon=True).start(); response=replies.get(timeout=120)
            if isinstance(response,BaseException): raise response
            event("response",response=response);monitor.require()
            actual=oracle.validate_sve(sve,response);require_frozen_sve(actual,fixture["frozen_native_sve"])
            row={"fixture":fixture["name"],"iteration":label,"sve_path":str(sve),"native_frozen_sha_pass":True,**actual}
            record["results"].append(row);return row,prefix
        for fixture in c["fixtures"]: encode(fixture,"baseline-normal",False)
        for fixture in c["fixtures"]:
            encode(fixture,"warmup",False);previous=0
            for ordinal in range(1,4):
                row,prefix=encode(fixture,f"formal-{ordinal}",True)
                manifest,size=validate_manifest(prefix,fixture,c,ordinal,previous,Path(row["sve_path"]));previous+=size;row["manifest"]=manifest
                event("stage_capture_verified",fixture=fixture["name"],iteration=ordinal,bytes=size)
        proc.stdin.write(b"QUIT\n");proc.stdin.flush();code=proc.wait(timeout=10)
        if code: raise ValueError("native helper QUIT exit differs")
        record.update(status="native_stage_fingerprint_pass",exit_code=code,final_tree=tree.snapshot())
    except BaseException as exc:
        record.update(status="failed_preserved",error=f"{type(exc).__name__}: {exc}")
    finally:
        for label,operation in (("descendant_cleanup",tree.cleanup_descendants if tree else None),
                                ("root_cleanup",owner.close if owner else None),
                                ("monitor_cleanup",monitor.close if monitor else None)):
            if operation is not None:
                try: record[label]=operation()
                except Exception as exc: record.update(status="failed_preserved",**{label+"_error":f"{type(exc).__name__}: {exc}"})
        if owner is not None: record["exit_code"]=owner.returncode
        if stderr is not None: stderr.close()
        with (out/"native-stage-receipt.json").open("x",encoding="utf-8") as stream:
            json.dump(record,stream,indent=2);stream.flush();os.fsync(stream.fileno())
        journal.close()
    return record


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract",type=Path,default=CONTRACT)
    parser.add_argument("--run-native",action="store_true")
    parser.add_argument("--inspect-ov-receipt",type=Path)
    args=parser.parse_args(argv)
    try:
        c=document(args.contract);result=inspect(c)
        if args.run_native: result=native_run(c)
        if args.inspect_ov_receipt: result=compare_ov_fingerprints(document(args.inspect_ov_receipt),c)
        print(json.dumps(result,indent=2));return 1 if result.get("status")=="failed_preserved" else 0
    except Exception as exc: print(json.dumps({"status":"rejected","error":f"{type(exc).__name__}: {exc}"}));return 2


if __name__=="__main__": raise SystemExit(main())
