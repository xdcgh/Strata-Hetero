#!/usr/bin/env python3
"""Small saved-array norm replay only; no GGUF, encoder, OpenVINO or device."""
from __future__ import annotations
import argparse, datetime, hashlib, io, json, os, sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT))
SOURCE=Path("E:/Strata-Hetero-data/source/llama-3cf0325")
REV="3cf03257f219afbe7334045ff7c6a06ac68c627d"
EXPORT=Path("E:/Strata-Hetero-data/vision-payloads/20261009-30-native-vision-export/export-receipt.json")
NATIVE=Path("E:/Strata-Hetero-data/vision-stage-oracle/20261009-43-native/native-stage-receipt.json")
OV=Path("E:/Strata-Hetero-data/vision-stage-oracle/20261009-43-openvino/comparison-receipt.json")
INVENTORY=ROOT/"bench/hetero/20261009-16-vision-encoder-headers/vision-header-inventory.json"
OUTPUT=Path("E:/Strata-Hetero-data/vision-replays/20261009-44-ln1-source")
FROZEN={"tools/hetero_vision_qwen3vl.py":"7888d8d1c7c1be3fa919aaed5691038467d4b1d53d884747da279c6fffec482e",
 "bench/hetero/20261009-43-vision-stage-oracle-preparation/client.py":"ea0dbe8745be86496b9bcf3c05631835882a89e4b2f3a3a0f7134d6b451633fe",
 "bench/hetero/20261009-43-vision-stage-oracle-preparation/ov_driver.py":"8da72cbc7c2128ff272821a67796b07ea972e6a61ede8480d2f79e0b03b33fc7",
 "bench/hetero/20261009-43-vision-stage-oracle-preparation/analyse.py":"a161378939fc7c00d12dcb452dda0e5e1405c4a0fecacfdde5245fd5d319e336"}
SOURCES=("ggml/src/ggml-cpu/ops.cpp","ggml/src/ggml-cpu/vec.h","ggml/src/ggml-cpu/vec.cpp",
 "ggml/src/ggml-cpu/ggml-cpu.c","ggml/src/ggml-cpu/ggml-cpu-impl.h","ggml/src/ggml-cpu/simd-mappings.h",
 "ggml/src/ggml-cpu/llamafile/sgemm.cpp","ggml/src/ggml-impl.h","ggml/src/ggml-cpu/CMakeLists.txt",
 "tools/mtmd/models/qwen3vl.cpp","tools/mtmd/clip.cpp")


def digest(raw):return hashlib.sha256(raw).hexdigest()
def document(path):return json.loads(path.read_text(encoding="utf-8-sig"))
def sha(path):return digest(path.read_bytes())


def checked_raw(path, size, expected):
    if not 0<size<=1<<20:raise ValueError("small saved-array byte bound exceeded")
    before=path.stat()
    with path.open("rb") as stream:raw=stream.read(size+1)
    after=path.stat()
    identity=lambda s:(s.st_size,s.st_mtime_ns,s.st_ctime_ns,s.st_ino,s.st_dev)
    if identity(before)!=identity(after) or len(raw)!=size or digest(raw)!=expected:
        raise ValueError("saved-array identity changed: "+str(path))
    return raw


def load_npy(item,np):
    raw=checked_raw(Path(item["path"]),item["file_bytes"],item["file_sha256"])
    array=np.load(io.BytesIO(raw),allow_pickle=False)
    if array.dtype!=np.float32 or list(array.shape)!=item["shape"] or not array.flags.c_contiguous or not np.isfinite(array).all():
        raise ValueError("saved NPY dtype/layout/finite mismatch")
    if digest(array.tobytes())!=item["decoded_sha256"]:raise ValueError("decoded saved-array SHA differs")
    return array


def native_array(item,np):
    if item["actual_dtype"]!="F32" or item["ne"][0]!=1152 or item["ne"][1] not in (36,72) or item["ne"][2:]!=[1,1]:
        raise ValueError("native norm-replay stage layout differs")
    raw=checked_raw(Path(item["path"]),item["bytes"],item["raw_sha256"])
    array=np.frombuffer(raw,dtype="<f4").reshape(item["ne"][1],1152).copy()
    if not np.isfinite(array).all():raise ValueError("native stage nonfinite")
    return array


def norm_source_avx2(x,w,b,eps,np):
    # vec.h:1495-1502: sequential binary64 sum -> binary32 sum -> binary32 mean.
    total=np.zeros((x.shape[0],1),dtype=np.float64)
    for col in range(x.shape[1]):total=np.add(total,x[:,col:col+1].astype(np.float64))
    mean=np.divide(total.astype(np.float32),np.float32(x.shape[1]),dtype=np.float32)
    centered=np.subtract(x,mean,dtype=np.float32)
    square=np.multiply(centered,centered,dtype=np.float32)
    # vec.cpp:467-478: high/low 128-bit halves, then lanes 0+2 and 1+3, then 0+1.
    total=np.zeros((x.shape[0],1),dtype=np.float64)
    for col in range(0,x.shape[1],8):
        lanes=square[:,col:col+8]
        four=np.add(lanes[:,4:],lanes[:,:4],dtype=np.float32)
        even=np.add(four[:,0:1],four[:,2:3],dtype=np.float32)
        odd=np.add(four[:,1:2],four[:,3:4],dtype=np.float32)
        total=np.add(total,np.add(even,odd,dtype=np.float32).astype(np.float64))
    variance=(total/np.float64(x.shape[1])).astype(np.float32)
    scale=np.divide(np.float32(1),np.sqrt(np.add(variance,eps,dtype=np.float32)),dtype=np.float32)
    normalized=np.multiply(centered,scale,dtype=np.float32)
    result=np.add(np.multiply(normalized,w,dtype=np.float32),b,dtype=np.float32)
    return result,np.concatenate((mean,variance,scale),axis=1)


def norm_port_numpy(x,w,b,eps,np):
    # Algebra of the current port; NumPy reduction order does not emulate a compiled OV kernel.
    mean=np.mean(x,axis=-1,keepdims=True,dtype=np.float32)
    centered=np.subtract(x,mean,dtype=np.float32)
    variance=np.mean(np.multiply(centered,centered,dtype=np.float32),axis=-1,keepdims=True,dtype=np.float32)
    normalized=np.divide(centered,np.sqrt(np.add(variance,eps,dtype=np.float32)),dtype=np.float32)
    return np.add(np.multiply(normalized,w,dtype=np.float32),b,dtype=np.float32)


def plan():
    for name,expected in FROZEN.items():
        if sha(ROOT/name)!=expected:raise ValueError("frozen executed code changed: "+name)
    return {"status":"saved_array_replay_plan","source_revision":REV,"frozen_code_sha256":FROZEN,
      "source_sha256":{name:sha(SOURCE/name) for name in SOURCES},"core_created":False,"gguf_read":False}


def run(output):
    import numpy as np
    from tools.hetero_xpu_worker import DEFAULT_TOLERANCES,quality_report
    from tools.hetero_resources import memory_snapshot
    metadata=plan();export=document(EXPORT);native=document(NATIVE);ov=document(OV);inventory=document(INVENTORY)
    if export["asset_sha256"]!="b1a82259702816a5330d7bd7607cd9676b11780e79ff7348c21103ff3ce49bd0" or export["source_revision"]!=REV:
        raise ValueError("pinned export identity differs")
    if sha(EXPORT)!="9293d413d80a89bea202d9c79c480fb3e8c07ca79e8fae740ca52cfca0ff4e01":raise ValueError("export receipt changed")
    if native["status"]!="native_stage_fingerprint_pass" or ov["status"]!="compared":raise ValueError("saved stage receipts incomplete")
    eps=np.float32(inventory["metadata_keys_and_values"]["clip.vision.attention.layer_norm_epsilon"]["value"])
    witem=export["tensors"]["v.blk.1.ln1.weight"];bitem=export["tensors"]["v.blk.1.ln1.bias"]
    w=load_npy(witem,np);b=load_npy(bitem,np)
    if w.shape!=(1152,) or b.shape!=(1152,):raise ValueError("ln1.1 affine width differs")
    mem,errors=memory_snapshot()
    if errors or mem["physical_available_bytes"]<12*2**30 or mem["commit_available_bytes"]<4*2**30:raise ValueError("host memory gate failed")
    output.mkdir(parents=True,exist_ok=False)
    receipt={**metadata,"status":"running","numpy_version":np.__version__,"python":sys.version,"pid":os.getpid(),
      "started_utc":datetime.datetime.now(datetime.timezone.utc).isoformat(),"core_created":False,"encoder_run":False,"gguf_read":False,
      "receipt_sha256":{"export":sha(EXPORT),"native":sha(NATIVE),"ov":sha(OV),"inventory":sha(INVENTORY)},
      "weights":{"weight":witem,"bias":bitem},"epsilon":float(eps),"epsilon_f32_bits":hex(int(eps.view(np.uint32))),
      "tolerances":dict(DEFAULT_TOLERANCES),"preflight_memory":mem,"rows":[]}
    def save(name,array):
        path=output/(name+".npy")
        with path.open("xb") as stream:np.save(stream,array,allow_pickle=False)
        return {"path":str(path),"file_bytes":path.stat().st_size,"file_sha256":sha(path),"shape":list(array.shape),"dtype":str(array.dtype),"decoded_sha256":digest(array.tobytes())}
    def compare(actual,reference):
        return {**quality_report(actual,reference,DEFAULT_TOLERANCES),"bit_identical":actual.tobytes()==reference.tobytes(),
          "different_f32_elements":int(np.count_nonzero(actual.view(np.uint32)!=reference.view(np.uint32)))}
    try:
        groups={i["fixture"]:i for arm in ov["arms"] if arm["policy"]=="cpu_vec_dot_rounding" for i in arm["images"]}
        for row in native["results"]:
            if row["iteration"]!="formal-1":continue
            fixture=row["fixture"];stages={s["stage"]:s for s in row["manifest"]["stages"]}
            exact_input=native_array(stages["block.0"],np);target=native_array(stages["ln1.1"],np)
            formal=groups[fixture]["formal"][0];ov_input=load_npy(formal["taps"]["block.0"],np);ov_target=load_npy(formal["taps"]["ln1.1"],np)
            source_native,stats=norm_source_avx2(exact_input,w,b,eps,np)
            port_native=norm_port_numpy(exact_input,w,b,eps,np)
            source_ov,_=norm_source_avx2(ov_input,w,b,eps,np);port_ov=norm_port_numpy(ov_input,w,b,eps,np)
            receipt["rows"].append({"fixture":fixture,"native_inputs":{"block.0":stages["block.0"],"ln1.1":stages["ln1.1"]},
              "ov_inputs":{"block.0":formal["taps"]["block.0"],"ln1.1":formal["taps"]["ln1.1"]},
              "comparisons":{"source_native_vs_native":compare(source_native,target),"port_native_vs_native":compare(port_native,target),
                "source_ov_input_vs_native":compare(source_ov,target),"port_ov_input_vs_native":compare(port_ov,target),
                "port_ov_input_vs_saved_ov":compare(port_ov,ov_target),"source_ov_input_vs_saved_ov":compare(source_ov,ov_target),
                "saved_ov_ln1_vs_native":compare(ov_target,target),"saved_ov_block0_vs_native":compare(ov_input,exact_input)},
              "outputs":{name:save(fixture+"-"+name,array) for name,array in (("source-native-input",source_native),("port-native-input",port_native),
                ("source-ov-input",source_ov),("port-ov-input",port_ov),("calculated-source-native-mean-variance-scale",stats))}})
        if len(receipt["rows"])!=4:raise ValueError("four saved fixture groups required")
        receipt.update(status="saved_array_replay_completed",native_mean_variance_intermediates_captured=False,
          source_norm_replay_scope="Source AVX2 F32/double order with NumPy sqrt/div/mul/add; no native norm invoked.",
          port_replay_scope="Current graph algebra with NumPy F32 reductions; not an exact compiled OpenVINO reduction/fusion emulator.",
          original_reference_replaced=False,gate_relaxed=False,npu_accepted=False)
    except BaseException as exc:receipt.update(status="failed_preserved",error=f"{type(exc).__name__}: {exc}")
    finally:
        receipt["finished_utc"]=datetime.datetime.now(datetime.timezone.utc).isoformat()
        receipt["replay_script_sha256"]=sha(Path(__file__))
        with (output/"replay-receipt.json").open("x",encoding="utf-8") as stream:json.dump(receipt,stream,indent=2)
    return receipt


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("--run",action="store_true");parser.add_argument("--output",type=Path,default=OUTPUT)
    args=parser.parse_args();result=run(args.output) if args.run else plan()
    print(json.dumps(result if not args.run else {"status":result["status"],"fixture_groups":len(result["rows"]),"core_created":False},indent=2))
    return 1 if result["status"]=="failed_preserved" else 0


if __name__=="__main__":raise SystemExit(main())
