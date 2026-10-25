"""Rehash retained host results and check outcomes without spawning helpers."""
import hashlib
import json
from pathlib import Path

BASE = Path(__file__).resolve().parent


def document(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def main():
    build = document(BASE / "build-receipt.json")
    lifecycle = document(BASE / "lifecycle-01/receipt.json")
    disabled = document(BASE / "disabled-receipt.json")
    terminal = document(BASE / "terminal-process-check.json")
    assert build["status"] == "host_fixture_compiled" and build["exit_code"] == 0 and build["source_unchanged"]
    assert disabled["exit_code"] == 0 and terminal["owners_absent"] is True and terminal["matching_processes"] == []
    assert lifecycle["all_passed"] is True and lifecycle["timeout_scenario_run"] is True
    assert [x["scenario"] for x in lifecycle["records"]] == ["success", "reject", "partial", "short-output", "nan", "timeout"]
    verified = set()
    def verify(path, record):
        raw = path.read_bytes()
        assert len(raw) == record["bytes"] and hashlib.sha256(raw).hexdigest() == record["sha256"], path
        verified.add(str(path))
    for name, value in lifecycle["files"].items():
        verify(Path(name), value)
    for artifact in build["artifacts"]:
        verify(Path(artifact["path"]), artifact)
        verify(BASE / Path(artifact["path"]).name, artifact)
    for name, digest in build["source_sha256"].items():
        assert hashlib.sha256((Path(build["source"]) / name).read_bytes()).hexdigest() == digest
    owners = 0
    for case in lifecycle["records"]:
        scenario = case["scenario"]
        assert case["exit_code"] == 0 and case["runner_timed_out"] is False and case["passed"] is True
        assert len(case["process_cleanup"]) == 2
        for identity in case["process_cleanup"]:
            owners += 1
            assert identity["owned_process_gone"] is True and identity["observations"]
            assert identity["original"]["pid"] > 0 and identity["original"]["creation_filetime"] > 0
            observed = identity["observations"][-1]
            assert observed["state"] in ("absent", "exited") or observed["creation_filetime"] != identity["original"]["creation_filetime"]
        stdout = (BASE / "lifecycle-01" / (scenario + ".stdout.txt")).read_text()
        success = int(scenario == "success")
        assert f"eligible_fixture success_expected={success} completed={success} outputs_verified=1" in stdout
        if not success:
            stderr = (BASE / "lifecycle-01" / (scenario + ".stderr.txt")).read_text()
            assert "completed=0 fallback=1" in stderr and "bridge_disabled_after_failure" in stderr
        if scenario == "timeout":
            assert 120 <= case["wall_seconds"] < 145
            assert "reason=bridge_timeout" in stderr
    result = {"status": "host_lifecycle_independently_verified", "scenarios": 6, "owned_mock_and_descendant_identities_gone": owners,
              "hashed_files": len(verified), "source_commit": build["source_commit"], "executable": build["executable"],
              "claim_limit": "Real C++ pipe with synthetic stdlib children only. No Arc, NPU, expert math, model output or end-to-end performance acceptance.",
              "artifacts": {str(p.relative_to(BASE)): {"sha256": hashlib.sha256(p.read_bytes()).hexdigest(), "bytes": p.stat().st_size}
                            for p in sorted(BASE.rglob("*")) if p.is_file() and p.name != "root-review.json"}}
    with (BASE / "root-review.json").open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2)
    print(json.dumps({k: v for k, v in result.items() if k != "artifacts"}, indent=2))


if __name__ == "__main__":
    main()
