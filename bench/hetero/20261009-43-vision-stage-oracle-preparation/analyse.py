#!/usr/bin/env python3
"""Explicit post-run native/OV stage inspection; retains every negative comparison."""
from __future__ import annotations
import argparse, importlib.util, json, sys
from pathlib import Path

HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location("vision_stage_analysis_client",HERE/"client.py")
client=importlib.util.module_from_spec(spec);sys.modules[spec.name]=client;spec.loader.exec_module(client)
ORDER=("patch_merge","positioned","ln1.0","qkv.0","block.0","ln1.1","block.1","ln1.26","block.26","post_norm","merged")


def compare_stages(native,ov,c):
    if native["status"]!="native_stage_fingerprint_pass":raise ValueError("native frozen final fingerprints did not pass")
    import numpy as np
    from tools.hetero_vision_cpu_compare import load_array
    from tools.hetero_xpu_worker import DEFAULT_TOLERANCES, quality_report
    captured={(r["fixture"],int(r["iteration"].split("-")[1])):r for r in native["results"] if r["iteration"].startswith("formal-")}
    if len(captured)!=12:raise ValueError("native formal stage coverage incomplete")
    comparisons=[];first=[]
    fixtures={f["name"]:f for f in c["fixtures"]}
    for row in captured.values():
        expected=fixtures[row["fixture"]]["frozen_native_sve"]
        client.require_frozen_sve(row,expected)
        if client.sha(Path(row["sve_path"]))!=expected["sha256"]:raise ValueError("native final SVE changed after acceptance")
    for arm in ov["arms"]:
        if arm["execution_devices"]!=["CPU"] or arm["reported_inference_precision"]!="<Type: 'float32'>" or arm["reported_execution_mode"]!="ExecutionMode.ACCURACY":
            raise ValueError("diagnostic execution properties differ")
        for image in arm["images"]:
            for formal in image["formal"]:
                load_array(formal["output"])  # Reverify the diagnostic projection file before its fingerprint report.
                row=captured[(image["fixture"],formal["iteration"])];manifest=row["manifest"]
                stages={s["stage"]:s for s in manifest["stages"]}
                if set(stages)!=set(ORDER) or set(formal["taps"])!=set(ORDER):raise ValueError("exact stage coverage required")
                earliest=None
                for stage in ORDER:
                    item=stages[stage];path=Path(item["path"])
                    if path.stat().st_size!=item["bytes"] or client.sha(path)!=item["raw_sha256"]:raise ValueError("native stage changed after acceptance")
                    with path.open("rb") as stream:raw=stream.read(item["bytes"]+1)
                    if len(raw)!=item["bytes"]:raise ValueError("native stage bounded read differs")
                    reference=np.frombuffer(raw,dtype="<f4").reshape(item["ne"][1],item["ne"][0])
                    actual=load_array(formal["taps"][stage])
                    report=quality_report(actual,reference,DEFAULT_TOLERANCES)
                    comparisons.append({"policy":arm["policy"],"fixture":image["fixture"],"iteration":formal["iteration"],
                        "stage":stage,"quality":report,"native_raw_sha256":item["raw_sha256"],"ov_decoded_sha256":formal["taps"][stage]["decoded_sha256"]})
                    if earliest is None and not report["pass"]:earliest=stage
                first.append({"policy":arm["policy"],"fixture":image["fixture"],"iteration":formal["iteration"],"first_failing_stage":earliest})
    if len(comparisons)!=264:raise ValueError("both policies x4fixtures x3formal x11stages required")
    return {"schema_version":1,"status":"stage_diagnostics","stage_order":ORDER,"comparisons":comparisons,
        "first_divergence":first,"projection_fingerprints":client.compare_ov_fingerprints(ov,c),
        "tolerances":dict(DEFAULT_TOLERANCES),"native_reference_replaced":False,"gate_relaxed":False,"npu_accepted":False}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract",type=Path,default=client.CONTRACT)
    parser.add_argument("--native-receipt",type=Path,required=True)
    parser.add_argument("--ov-receipt",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():raise FileExistsError(args.output)
    c=client.document(args.contract);client.inspect(c)
    result=compare_stages(client.document(args.native_receipt),client.document(args.ov_receipt),c)
    with args.output.open("x",encoding="utf-8") as stream:json.dump(result,stream,indent=2)
    print(json.dumps({"status":result["status"],"comparison_count":len(result["comparisons"]),"npu_accepted":False}))


if __name__=="__main__":main()
