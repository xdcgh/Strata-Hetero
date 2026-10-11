"""Synthetic private-stdio protocol and adapter-flow fixtures; no engine/API/device access."""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import queue
import tempfile
import threading
from types import SimpleNamespace
import unittest
from unittest import mock

from tools import hetero_kv_parking as kv
from tools import hetero_kv_stdio as stdio


class _NoopMonitor:
    def require(self): pass


def _source_fingerprint():
    return {"L": "10", "gdn": "0000000000000001", "ple": "0000000000000002",
        "tail": "0000000000000003", "pooled": "0000000000000004", "kv": "0000000000000005",
        "mtp": "0000000000000006", "stale": "0000000000000007", "dead": "0000000000000008",
        "pooled_full": "0000000000000009", "ple_prev": "1,2"}


class StdioProtocolTests(unittest.TestCase):
    def engine_shell(self, lines):
        engine = object.__new__(stdio._Engine)
        engine.lines = queue.Queue()
        for line in lines: engine.lines.put(line.encode("utf-8") + b"\n")
        engine.read_error = None
        engine.proc = SimpleNamespace(poll=lambda: None)
        engine.monitor = _NoopMonitor()
        engine.stderr_lines = []
        engine.states = []
        engine.lock = threading.RLock()
        return engine

    def test_ready_info_and_stop_cap_are_parsed(self):
        engine = self.engine_shell(["INFO context=32768 kv=int8 pool_workers=10 conversation_cache_mib=0 engine=0.1.41",
                                    "READY 32768 stop"])
        result = engine.ready(1e20, 32768)
        self.assertEqual(result["context"], 32768)
        self.assertEqual(result["info"]["kv"], "int8")
        self.assertEqual(result["info"]["pool_workers"], 10)

    def test_gen_t_done_uses_emitted_stop_id_and_post_gen_fnv(self):
        engine = self.engine_shell(["T 42", "T 248046",
            "DONE 2 2 1.0 2.0 stop 0 0 2 0 0 0 0 0.0 2 0"])
        engine.tokenizer = SimpleNamespace(decode=lambda ids, errors="strict": '{"ok":true}')
        engine.states = []
        engine.send = lambda command: engine.states.append(_source_fingerprint())
        req = {"request_id":"A-formal1","prompt_sha256":"a"*64,"prompt_token_ids":[1,2],
            "max_tokens":4,"sampling":{"temperature":0,"top_p":1,"seed":42},"prompt_cache":3}
        result = engine.generate(req, [248046,248044], 1e20)
        self.assertEqual(result["emitted_token_ids"], [42,248046])
        self.assertEqual(result["raw_output_utf8"], '{"ok":true}')
        self.assertEqual(result["state_fingerprint"]["algorithm"], "fnv1a64")
        self.assertFalse(result["state_fingerprint"]["cryptographic"])
        self.assertEqual(set(result["state_fingerprint"]["domains"]), set(stdio.STATE_HASH_DOMAINS))
        self.assertNotIn("L_tokens", result["state_fingerprint"]["domains"])
        self.assertEqual(result["state_fingerprint"]["metadata"]["L_tokens"], 10)
        self.assertEqual(result["state_fingerprint"]["metadata"]["ple_prev"], [1,2])

    def test_save_restore_session_lines_are_typed_and_bounded(self):
        engine = self.engine_shell(["SESSION 0 100", "SWAIT transfer 2", "SESSION 100 100", "SAVED 30 100 1.0"])
        commands = []; engine.send = commands.append
        lines, result = engine.protocol_session("save", Path("E:/unused/A-seed.bin"), 1e20, 10)
        self.assertEqual(commands, [f"SAVE {str(Path('E:/unused/A-seed.bin'))}"])
        self.assertEqual(result["action"], "save")
        self.assertEqual(result["tokens"], 30)
        self.assertEqual(lines[1], "SWAIT transfer 2")
        engine = self.engine_shell(["SESSION 0 100", "SESSION 100 100", "RESTORED 30 100 1.5"])
        engine.send = commands.append
        _, restored = engine.protocol_session("restore", Path("E:/unused/A-seed.bin"), 1e20, 10)
        self.assertEqual(restored["action"], "restore")
        self.assertEqual(restored["bytes"], 100)

    def test_factory_import_is_private_and_default_off(self):
        adapter = stdio.create_adapter()
        self.assertEqual(adapter.api_version, 1)
        self.assertEqual(adapter.owned_process_trees(), [])

    def test_monitor_offset_timestamp_normalizes_to_utc_z_and_preserves_source(self):
        with tempfile.TemporaryDirectory(prefix="kv-monitor-time-") as temp:
            path = Path(temp) / "monitor.jsonl"
            source_time = "2026-10-11T09:00:00.125+08:00"
            path.write_text(json.dumps({"record_type":"resource_sample","observed_at_utc":source_time,
                "gate_ok":True,"errors":[],"physical_available_bytes":20*1024**3,
                "commit_available_bytes":20*1024**3,"memory_source":"GlobalMemoryStatusEx physical + PSAPI GetPerformanceInfo",
                "owned_service_tree":{"owner_tree":[{"role":"engine","pid":55}]}})+"\n",encoding="utf-8")
            adapter = stdio.create_adapter(); adapter._collect_monitor(path)
            row = adapter._continuous_checkpoints[0]
            self.assertEqual(row["observed_at_utc"], "2026-10-11T01:00:00.125000Z")
            self.assertEqual(row["monitor_observed_at_utc"], source_time)
            self.assertEqual(kv.validate_checkpoint_history([row])["sample_count"], 1)

    def test_monitor_naive_timestamp_is_rejected(self):
        with tempfile.TemporaryDirectory(prefix="kv-monitor-naive-") as temp:
            path = Path(temp) / "monitor.jsonl"
            path.write_text(json.dumps({"record_type":"resource_sample","observed_at_utc":"2026-10-11T09:00:00",
                "gate_ok":True,"errors":[],"physical_available_bytes":20*1024**3,
                "commit_available_bytes":20*1024**3,"memory_source":"PSAPI GetPerformanceInfo",
                "owned_service_tree":{"owner_tree":[{"role":"engine","pid":55}]}})+"\n",encoding="utf-8")
            with self.assertRaisesRegex(stdio.StdioAdapterError, "include a timezone"):
                stdio.create_adapter()._collect_monitor(path)


class ExecutePlanMockTests(unittest.TestCase):
    def test_execute_plan_runs_full_sequence_with_fake_engine_and_temp_seed_files(self):
        base, _, _ = kv.load_contract(kv.V2_CONTRACT)
        with tempfile.TemporaryDirectory(prefix="kv-stdio-fixture-") as temp:
            root = Path(temp); contract = copy.deepcopy(base)
            contract_sha = "c" * 64
            contract["runtime_output_directory"] = str(root / "runtime")
            contract["snapshot_paths"]["directory"] = str(root / "snapshots")
            fake_exe = root / "fake-strata.exe"
            tokenizer_dir = root / "tokenizer"
            protocol = contract["stdio_protocol_binding"]
            protocol["tokenizer_directory"] = str(tokenizer_dir)
            configs = {}
            for index, (name, spec) in enumerate(contract["configs"].items(), 1):
                path = root / f"{name}.json"
                doc = {"exe":str(fake_exe),"tokenizer":str(tokenizer_dir),
                    "log":str(root / "logs" / name / "engine.log"),"args":[],"env":{},
                    "hetero_transport":"private stdio StrataEngine; no HTTP listener"}
                raw=(json.dumps(doc,sort_keys=True)+"\n").encode(); path.write_bytes(raw)
                configs[name]={"path":str(path),"sha256":hashlib.sha256(raw).hexdigest(),
                    "name":name,"cache_mib":4096 if name=="ram-pair" else 0,
                    "prompt_cache":3,"max_context":32768,"transport":doc["hetero_transport"],
                    "sampling":{"temperature":0,"top_p":1,"seed":42},"config_sha256_checked":True}

            manifest={"schema_version":2,"contract_sha256":contract_sha,
                "stdio_protocol_binding":copy.deepcopy(protocol),"conversations":{}}
            for conv in ("A","B"):
                turns=[]
                for turn in kv.TURN_IDS:
                    text=f"{conv}-{turn} fixture"; ids=list(range(30720)) if turn=="warmup" else [1,2]
                    turns.append({"turn_id":turn,"prompt_utf8":text,"prompt_sha256":hashlib.sha256(text.encode()).hexdigest(),
                        "prompt_token_ids":ids,"max_tokens":128,"sampling":{"temperature":0,"top_p":1,"seed":42}})
                manifest["conversations"][conv]={"prompt_cache":3,"turns":turns}
            counts={"made":0}; saved={}

            class Monitor:
                def __init__(self,path,*args,**kwargs): self.path=path
                def start(self): self.path.parent.mkdir(parents=True,exist_ok=True); self.path.write_text("",encoding="utf-8")
                def close(self): pass

            class Engine:
                def __init__(self,name,config,exe,env,run42,psutil,tokenizer,monitor,session_wait):
                    counts["made"]+=1; self.name=name; self.config=config; self.monitor=monitor
                    self.proc=SimpleNamespace(pid=9000+counts["made"],returncode=None,poll=lambda:self.proc.returncode)
                    owner={"role":"engine","pid":self.proc.pid,"create_time_utc":"2026-10-11T00:00:00Z",
                        "executable":str(fake_exe),"parent_pid":1,"argv":[str(fake_exe),"--serve"]}
                    self.tree=SimpleNamespace(owner_rows=lambda: [owner] if self.proc.returncode is None else [],cleanup_descendants=lambda:None)
                    self.tokenizer=tokenizer; self.last_conv=None; self.seen=set()
                def ready(self,deadline,ctx):
                    cache=4096 if self.name=="ram-pair" else 0
                    return {"line":"READY 32768 stop","context":32768,"info":{"context":32768,"kv":"int8",
                        "pool_workers":10,"conversation_cache_mib":cache,"engine":"0.1.41"}}
                def generate(self,req,stop_ids,deadline):
                    conv=req["conversation"]; turn=req["turn_id"]
                    text=json.dumps(contract["canonical_outputs"][conv],sort_keys=True,separators=(",",":"))
                    tokens=[100+ord(conv),kv.TURN_IDS.index(turn)+1,248046]
                    logs=[]
                    if self.name=="ram-pair":
                        if self.last_conv:
                            logs.append(f"strata serve: conversation cache: parked 30720 tokens in 1.0 ms; parked=1 bytes=1024 evictions=0 snapshot_bytes=512 reused_kv_bytes=256")
                        if conv in self.seen:
                            logs.append("strata serve: conversation cache: restored 30720 tokens (checkpoint) in 1.0 ms; parked=1 bytes=1024")
                        self.last_conv=conv; self.seen.add(conv)
                    return {"request_id":req["request_id"],"prompt_sha256":req["prompt_sha256"],
                        "prompt_token_ids":req["prompt_token_ids"],"max_tokens":req["max_tokens"],"sampling":req["sampling"],
                        "prompt_cache_entries":3,"raw_output_utf8":text,"visible_json":json.loads(text),
                        "emitted_token_ids":tokens,"finish_reason":"stop","stop_token_ids":stop_ids,
                        "state_fingerprint":None,"prompt_ms":1.0,"decode_ms":1.0,"prompt_tokens_reused":30720,
                        "done_line":"DONE 3 2 1.0 1.0 stop 0 0 2 0 0 0 0 0.0 2 0","stderr_protocol_lines":logs}
                def protocol_session(self,action,path,deadline,wait_cap):
                    conv=path.stem[0]
                    if action=="save":
                        req_token_count=30720+3; path.write_bytes(b"x"*100); saved[conv]=req_token_count
                        word="SAVED"; n=req_token_count
                    else: word="RESTORED"; n=saved[conv]
                    line=f"{word} {n} 100 1.0"
                    return ["SESSION 0 100","SESSION 100 100","SWAIT transfer 1",line],kv.parse_session_line(line,wait_cap)
                def close(self): self.proc.returncode=0
                def finish_logs(self): pass

            checkpoint_index=[0]
            def checkpoint(label,owners,additional_bytes=0,additional_disk_bytes=0):
                checkpoint_index[0]+=1
                seconds=checkpoint_index[0]
                stamp=f"2026-10-11T00:{seconds//60:02d}:{seconds%60:02d}.000Z"
                return {"label":"prelaunch" if not owners else label,
                    "resource":{"status":"pass","physical_available_bytes":20*1024**3,
                        "commit_available_bytes":20*1024**3,"source":"PSAPI GetPerformanceInfo","errors":[]},
                    "disk":{"status":"pass"} if "disk" in label or "save" in label or "restore" in label else None,
                    "owner_tree":owners,"observed_at_utc":stamp}

            plan={"execution_contract_version":2,"contract_sha256":contract_sha,"run_id":contract["run_id"],
                "contract_path":str(root/"contract.json"),"config_bindings":configs,
                "request_manifest_sha256":hashlib.sha256(kv.canonical_json_bytes(manifest)).hexdigest()}
            with mock.patch.object(kv,"load_contract",return_value=(contract,contract_sha,configs)), \
                 mock.patch.object(stdio,"_tokenizer",return_value=SimpleNamespace(encode=lambda text,parse_special=True:[248046] if text=="<|im_end|>" else [248044])), \
                 mock.patch.object(stdio,"_helpers",return_value=SimpleNamespace(ResourceMonitor=Monitor)), \
                 mock.patch.object(stdio,"_binary",return_value=(fake_exe,stdio.BIN_SHA,stdio.SOURCE_SHA)), \
                 mock.patch.object(stdio,"_Engine",Engine):
                bundle=stdio.create_adapter().execute_plan(plan,manifest,checkpoint)

            self.assertEqual(counts["made"],4)
            self.assertEqual(kv.validate_arm_completions(bundle["ram_pair"]["completions"],
                kv.validate_request_manifest(manifest,contract,contract_sha),
                kv.validate_reference_manifest(bundle["references"],kv.validate_request_manifest(manifest,contract,contract_sha),
                    contract,contract_sha,plan["request_manifest_sha256"]),contract)["status"],"pass")
            self.assertEqual(kv.validate_disk_evidence(bundle["disk_pair"]["evidence"],contract,
                bundle["disk_pair"]["expected_seed_tokens"])["status"],"pass")
            self.assertTrue((root/"snapshots"/"A-seed.bin").is_file())


if __name__ == "__main__":
    unittest.main()
