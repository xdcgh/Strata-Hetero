"""Root-reviewed, no-inference admission and authorization for joint-only run61."""
from pathlib import Path
import datetime as dt
import hashlib
import json
import shutil
import subprocess

import psutil
from tools import hetero_native_ipc_compare as compare
from tools.hetero_resources import memory_snapshot

BASE = Path(__file__).resolve().parent
CLIENT_SHA = "bbd8a426c012d5f768175cd898e9dffe127a51ee8b4419eb003017332adbc45b"
CONTRACT_SHA = "4af7eeb868202cee93d28ad534ee1b3add0642541c7dc9c934e6b72f7a2c1e36"
FREEZE_SHA = "9f5758af2359426ec2e9710cd4f6edcc08f1c54ad8855b1061d2830a33b1bc90"

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def exclusive_json(path, value):
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")

contract_path = BASE / "compare-contract.json"
freeze_path = BASE / "source-freeze-final.json"
client = Path(compare.__file__)
assert sha(client) == CLIENT_SHA
assert sha(contract_path) == CONTRACT_SHA
assert sha(freeze_path) == FREEZE_SHA
contract = compare.load_contract(contract_path)
freeze = json.loads(freeze_path.read_text(encoding="utf-8-sig"))
assert len(freeze["files"]) == 33
for row in freeze["files"]:
    path = Path(row["path"])
    assert path.is_file() and path.stat().st_size == row["bytes"], row
    assert sha(path) == row["sha256"], row
assert not Path(contract["output"]["private_output_root"]).exists()
assert not Path(contract["output"]["receipt_dir"]).exists()
pins = compare._check_source_pins(contract)
binaries = {b["id"]: compare._verify_cpu_binary(contract, b) for b in contract["backends"]}
baseline = compare._verify_paired_baseline(contract)
assert baseline["run_id"] == "native-cpu-arc-ipc-compare-20261011-60"
assert shutil.disk_usage(Path(contract["output"]["private_output_root"]).parent).free >= 12 * 2**30

wsl = subprocess.run(["wsl.exe", "--list", "--running", "--quiet"], capture_output=True, check=True)
wsl_text = wsl.stdout.decode("utf-16-le", errors="replace").replace("\x00", "").strip()
wsl_idle_proof = None
if wsl_text:
    # The desktop app restarts its /bin/sleep keepalive. Require actual quiescence;
    # a dormant VM is not a competing model workload for this CPU/Arc microcase.
    assert wsl_text == "Ubuntu-24.04", wsl_text
    def wsl_check(argv):
        return subprocess.run(["wsl.exe", "-d", "Ubuntu-24.04", "-u", "root", "--", *argv], capture_output=True, text=True)
    active = wsl_check(["systemctl", "is-active", "comfyui.service"])
    enabled = wsl_check(["systemctl", "is-enabled", "comfyui.service"])
    linux_ps = wsl_check(["ps", "-eo", "pid,ppid,comm,args"])
    dxg = wsl_check(["fuser", "/dev/dxg"])
    assert active.stdout.strip() == "inactive" and enabled.stdout.strip() == "disabled", (active.stdout, enabled.stdout)
    assert linux_ps.returncode == 0, linux_ps.stderr
    assert dxg.returncode in (0, 1) and not dxg.stdout.strip() and not dxg.stderr.strip(), (dxg.returncode, dxg.stdout, dxg.stderr)
    forbidden = ("python", "comfy", "strata", "vllm", "sglang", "ollama", "llama", "nvcc", "cmake", "ninja", "torchrun")
    assert not [line for line in linux_ps.stdout.splitlines()[1:] if any(x in line.lower() for x in forbidden)], linux_ps.stdout
    wsl_idle_proof = {"process_snapshot": linux_ps.stdout, "comfyui_active": active.stdout.strip(),
        "comfyui_enabled": enabled.stdout.strip(), "dxg_holders": [], "dxg_fuser_exit": dxg.returncode,
        "scope": "idle app keepalive; no Linux model, compiler, Python worker, or GPU device holder"}
competing = []
for p in psutil.process_iter(["pid", "name", "exe", "cmdline", "create_time"]):
    info = p.info
    name = (info["name"] or "").lower()
    argv = " ".join(info["cmdline"] or []).lower()
    if name.startswith("strata") or name in {"nvcc.exe", "cl.exe", "cmake.exe", "ninja.exe", "msbuild.exe", "hetero_native_cpu_service.exe"}:
        competing.append(info)
    if name.startswith("python") and any(x in argv for x in ("hetero_xpu_service", "serve/server.py", "serve\\server.py", "hetero_pool_sweep", "comfyui")):
        competing.append(info)
assert not competing, competing
listeners = [c._asdict() for c in psutil.net_connections("tcp") if c.status == "LISTEN" and c.laddr.port in {8080, 8081, 8188}]
assert not listeners, listeners
gpu = subprocess.run(["nvidia-smi", "--query-gpu=memory.free,memory.used,utilization.gpu,driver_version", "--format=csv,noheader,nounits"], capture_output=True, text=True, check=True)
fields = [s.strip() for s in gpu.stdout.strip().split(",")]
assert len(fields) == 4 and int(fields[0]) >= 46000 and int(fields[2]) == 0, fields
compute = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,process_name,used_gpu_memory", "--format=csv,noheader,nounits"], capture_output=True, text=True, check=True)
assert not compute.stdout.strip(), compute.stdout
memory, errors = memory_snapshot()
assert not errors and memory["physical_available_bytes"] >= 12 * 2**30 and memory["commit_available_bytes"] >= 4 * 2**30, (memory, errors)
admission = {"schema": "run61-root-admission-v1", "utc": dt.datetime.now(dt.timezone.utc).isoformat(),
    "wsl_running": wsl_text.splitlines(), "wsl_idle_proof": wsl_idle_proof,
    "competing_processes": competing, "owned_ports_listening": listeners,
    "gpu_csv": gpu.stdout.strip(), "compute_processes_csv": compute.stdout.strip(),
    "memory": memory, "gate_pass": True, "source_freeze_files_checked": 33,
    "source_pins_checked": len(pins), "cpu_binary_bindings": binaries,
    "root_baseline_validator_pass": True, "baseline_raw_responses_checked": 162,
    "pure_fixture_validation": str(BASE / "receipts-validation-04/validation-receipt.json")}
exclusive_json(BASE / "root-admission-01.json", admission)
with (BASE / "executed-client.py").open("xb") as stream:
    stream.write(client.read_bytes())
assert sha(BASE / "executed-client.py") == CLIENT_SHA
auth = {"schema": "strata-native-ipc-compare-root-authorization-v1", "run_id": contract["run_id"],
    "execution_authorized": True, "model_matrix_stopped": True, "root_review_complete": True,
    "client_sha256": CLIENT_SHA, "contract_sha256": CONTRACT_SHA, "source_freeze_sha256": FREEZE_SHA,
    "cpu16_none_authorized": True, "execution_mode": "paired_only",
    "utc": admission["utc"], "admission_sha256": sha(BASE / "root-admission-01.json"),
    "scope": "Only the two previously unrun synchronized CPU+Arc joint arms; no solo repeats or model route; stop first quality/resource/error gate",
    "comparison_scope": "cross-run solo baseline; not interleaved", "pure_fixture_count": 16}
for key in ("baseline_run_id", "baseline_contract_sha256", "baseline_receipt_sha256", "baseline_client_sha256", "root_solo_review_sha256", "root_arc_numerical_review_sha256"):
    auth[key] = contract["paired_only"][key]
exclusive_json(BASE / "root-authorization-01.json", auth)
print(json.dumps({"status": "root_review_admission_authorized", "run_id": contract["run_id"],
    "source_files_checked": 33, "physical_available_gib": memory["physical_available_bytes"] / 2**30,
    "commit_available_gib": memory["commit_available_bytes"] / 2**30, "gpu_free_mib": int(fields[0])}))
