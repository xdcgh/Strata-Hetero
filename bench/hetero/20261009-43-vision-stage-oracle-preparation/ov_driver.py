#!/usr/bin/env python3
"""Plan-only default for the existing CPU comparer with diagnostic output aliases."""
from __future__ import annotations
import argparse, importlib.util, json, os, subprocess, sys, time
from pathlib import Path

HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location("vision_stage_client",HERE/"client.py")
client=importlib.util.module_from_spec(spec);sys.modules[spec.name]=client;spec.loader.exec_module(client)
ROOT=client.ROOT


def argv(c):
    args=["E:/Strata-Hetero-data/venv-intel/Scripts/python.exe","-u","-B",
          str(ROOT/"tools/hetero_vision_cpu_compare.py"),"--run","--export-receipt",
          "E:/Strata-Hetero-data/vision-payloads/20261009-30-native-vision-export/export-receipt.json",
          "--output-dir",c["ov_diagnostic_root"]]
    for tap in c["ov_output_taps"]: args.extend(("--tap",tap))
    return args


def plan(c):
    client.inspect(c)
    return {"status":"plan_only","execution_authorized":c["execution_authorized"],"argv":argv(c),
            "reference":"frozen run23 native SVE","frozen_ov_projection":"run31 per-policy decoded hashes",
            "fingerprint_changes_explicit":True,"core_created":False,"child_started":False}


def execute(c):
    if c["execution_authorized"] is not True: raise ValueError("contract execution is not authorized")
    shared=client.guard_api(c)
    native=client.document(Path(c["output_root"])/"native-stage-receipt.json")
    if native["status"]!="native_stage_fingerprint_pass" or len(native["results"])!=20:
        raise ValueError("native baseline and all stage final-SVE fingerprint checks must pass first")
    import psutil
    output=Path(c["ov_diagnostic_root"])
    if output.exists(): raise FileExistsError(output)
    controller=output.with_name(output.name+"-controller");controller.mkdir(parents=True,exist_ok=False)
    record={"schema_version":1,"status":"running","argv":argv(c),"mode":"CPU diagnostic aliases only"}
    stdout=(controller/"stdout.json").open("xb");stderr=(controller/"stderr.log").open("xb")
    monitor=tree=owner=None
    try:
        record["controller_identity"]=shared._process_record(psutil.Process(os.getpid()))
        monitor=shared.ResourceMonitor(controller/"resource.jsonl",psutil,minimum_physical=12*2**30,minimum_commit=4*2**30)
        monitor.start();deadline=time.monotonic()+5
        while True:
            try: monitor.require();break
            except shared.BenchError:
                if monitor.failed is not None or time.monotonic()>=deadline:raise
                time.sleep(.05)
        proc=subprocess.Popen(record["argv"],stdout=stdout,stderr=stderr,shell=False,
            cwd=ROOT,creationflags=getattr(subprocess,"CREATE_NO_WINDOW",0)|getattr(subprocess,"CREATE_NEW_PROCESS_GROUP",0))
        owner=client.DirectOwner(proc)
        tree=shared.OwnedServiceTree(owner,psutil,Path(record["argv"][0]),Path("C:/Python314/python.exe"),
                                    ROOT/"tools/hetero_vision_cpu_compare.py",record["argv"][1:])
        monitor.attach(owner,tree)
        record["actual_worker_tree"]=tree.await_actual_child()  # Mandatory for the E shim -> C Python worker.
        while owner.returncode is None:monitor.require();time.sleep(.2)
        record["worker_exit_code"]=owner.returncode
        if owner.returncode not in (0,1):raise ValueError("CPU diagnostic worker failed")
        result=client.document(output/"comparison-receipt.json")
        if result["status"]!="compared":raise ValueError("CPU diagnostic arm coverage incomplete")
        record["projection_fingerprints"]=client.compare_ov_fingerprints(result,c)
        record.update(status="diagnostics_completed",native_reference_replaced=False,
                      policy_quality_pass=result["policy_quality_pass"],numerical_gate_relaxed=False)
    except BaseException as exc:record.update(status="failed_preserved",error=f"{type(exc).__name__}: {exc}")
    finally:
        for label,fn in (("descendant_cleanup",tree.cleanup_descendants if tree else None),
                         ("root_cleanup",owner.close if owner else None),("monitor_cleanup",monitor.close if monitor else None)):
            if fn:
                try:record[label]=fn()
                except Exception as exc:record.update(status="failed_preserved",**{label+"_error":str(exc)})
        stdout.close();stderr.close()
        with (controller/"controller-receipt.json").open("x",encoding="utf-8") as stream:
            json.dump(record,stream,indent=2);stream.flush();os.fsync(stream.fileno())
    return record


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("--contract",type=Path,default=client.CONTRACT)
    parser.add_argument("--run",action="store_true");args=parser.parse_args()
    try:
        c=client.document(args.contract);result=execute(c) if args.run else plan(c)
        print(json.dumps(result,indent=2));return 1 if result["status"]=="failed_preserved" else 0
    except Exception as exc:print(json.dumps({"status":"rejected","error":f"{type(exc).__name__}: {exc}"}));return 2


if __name__=="__main__":raise SystemExit(main())
