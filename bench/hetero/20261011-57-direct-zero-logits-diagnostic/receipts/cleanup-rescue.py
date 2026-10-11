from __future__ import annotations
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO))
from tools import hetero_pool_sweep as sweep

RUN = REPO / "bench/hetero/20261011-57-direct-zero-logits-diagnostic"
MATRIX = RUN / "matrix.json"
AUTH = RUN / "root-authorization-01.json"
PHASE = RUN / "arms/w10-all/quality-on"
CONTROLLER = PHASE / "controller"
OUT = RUN / "receipts/cleanup-rescue.json"

def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()

def read(name: str):
    return json.loads((PHASE / name).read_text(encoding="utf-8"))

if OUT.exists():
    raise SystemExit("cleanup rescue receipt already exists; refusing another cleanup attempt")
if not (PHASE / "cleanup").is_dir() or any((PHASE / "cleanup").iterdir()):
    raise SystemExit("precreated cleanup leaf is missing or nonempty")
failure = json.loads((CONTROLLER / "coordinator-failure.json").read_text(encoding="utf-8"))
quality_progress = json.loads((PHASE / "quality/quality-progress.jsonl").read_text(encoding="utf-8").splitlines()[-1])
if failure.get("execution_state", {}).get("live_owner_state") != "owned_service_ready_task_pending":
    raise SystemExit("failure receipt does not prove expected live owned-service state")
if quality_progress.get("event", {}).get("partial_prompts") != 0:
    raise SystemExit("quality controller did not stop before any prompt")
if read("process.json").get("engine_sha256") != "6bbc426bc6299cd8030b50634845c3f07d68ab2fe8168128ce66645fa5715fbb":
    raise SystemExit("recorded owner does not bind the authorized engine binary")
if read("sampler-process.json").get("run_id") != "20261011-57-w10-all-quality-on":
    raise SystemExit("sampler owner record does not bind the authorized phase")
watch = json.loads((PHASE / "resource/startup-watch-result.json").read_text(encoding="utf-8"))
if watch.get("status") != "startup_ready":
    raise SystemExit("watch receipt does not prove this run reached owned ready state")

matrix_sha, auth_sha, adapter_sha = sha(MATRIX), sha(AUTH), sha(Path(sweep.__file__).resolve())
authorization = sweep.validate_root_authorization(MATRIX, AUTH)
if authorization.get("max_phases") != 1 or matrix_sha != authorization.get("matrix_sha256"):
    raise SystemExit("authorization does not bind the exact one-phase run57 matrix")
plan = sweep.build_plan(MATRIX)
step = plan["steps"][0]
if step["run_id"] != "20261011-57-w10-all-quality-on":
    raise SystemExit("cleanup target is not the exact authorized first phase")

adapter = sweep.Run41LifecycleAdapter(MATRIX, authorization)
phase_result_path = CONTROLLER / "phase-result.json"
synthetic_failed_phase = {
    "step": step,
    "path": str(phase_result_path),
    "status": "failed_before_first_quality_prompt",
    "launcher_pid": read("process.json")["launcher_pid"],
    "sampler_pid": read("sampler-process.json")["sampler_launcher_pid"],
    "run_dir": str(PHASE),
}
# The normal coordinator cannot expose this failed-but-still-live phase to
# cleanup_previous. Seed only its in-memory target pointer; cleanup itself still
# uses the frozen template's full PID/exe/parent/CreateTime/argv and listener checks.
adapter.live_result = synthetic_failed_phase
adapter.execution_state.update({
    "execution_attempted": True,
    "lifecycle_controller_started": True,
    "model_launch_reported": True,
    "stage": "exact_owned_cleanup_after_quality_preflight_failure",
    "run_id": step["run_id"],
    "run_dir": str(PHASE),
    "controller_dir": str(CONTROLLER),
    "launcher_pid": synthetic_failed_phase["launcher_pid"],
    "sampler_pid": synthetic_failed_phase["sampler_pid"],
    "live_owner_state": "owned_service_ready_task_failed_cleanup_required",
})
started = dt.datetime.now(dt.timezone.utc).isoformat().replace("+00:00", "Z")
result = adapter.cleanup_previous(synthetic_failed_phase)
if result.get("status") != "terminal_owned_cleanup_pass":
    raise SystemExit("frozen adapter exact-owned cleanup did not reach terminal pass")
raw = result["raw_receipt"]
if raw.get("status") != "terminal_owned_server_and_sampler_released":
    raise SystemExit("cleanup template did not prove terminal owner release")
receipt = {
    "schema_version": 1,
    "run_id": step["run_id"],
    "status": "terminal_owned_cleanup_pass_after_quality_controller_preflight_failure",
    "cleanup_started_utc": started,
    "cleanup_finished_utc": dt.datetime.now(dt.timezone.utc).isoformat().replace("+00:00", "Z"),
    "matrix_sha256": matrix_sha,
    "authorization_sha256": auth_sha,
    "adapter_sha256": adapter_sha,
    "quality_failure_path": str(CONTROLLER / "coordinator-failure.json"),
    "quality_failure_sha256": sha(CONTROLLER / "coordinator-failure.json"),
    "quality_controller_requests_sent": 0,
    "quality_formal_prompts_completed": 0,
    "cleanup_result": result,
    "live_model_owner_ids": {
        "launcher_pid": synthetic_failed_phase["launcher_pid"],
        "server_pid": read("process.json")["actual_server_child_pid"],
        "engine_pid": watch["engine_pid"],
        "sampler_launcher_pid": synthetic_failed_phase["sampler_pid"],
        "sampler_child_pid": read("sampler-process.json")["actual_child_pid"],
    },
    "cleanup_executed_by_frozen_run41_adapter": True,
    "no_model_request_or_retry_during_cleanup": True,
}
OUT.write_text(json.dumps(receipt, indent=2, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n")
print(json.dumps({
    "status": receipt["status"],
    "cleanup_receipt": str(OUT),
    "cleanup_sha256": sha(OUT),
    "adapter_result": result["status"],
    "raw_cleanup_status": raw["status"],
    "live_owner_ids": receipt["live_model_owner_ids"],
}, indent=2))
