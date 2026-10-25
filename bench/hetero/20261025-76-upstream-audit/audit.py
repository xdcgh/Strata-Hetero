"""Retain a source-only upstream checkpoint; no checkout/build/model mutation."""
import datetime
import hashlib
import json
from pathlib import Path
import subprocess

BASE = "61b3fb5dd3f1e8ec09cf7e4e05208bc6d3c46406"
CURRENT = "7dd5cf18e68c639a8773e096c5c79f971ea2a9f5"
GUARD = "f332a85570282466a37293eb6be4eeada9566f29"
RETRY = "894f133b7d492382513cbe9673012d6c8e937bb8"
OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[2]


def git(*args):
    result = subprocess.run(["git", *args], cwd=ROOT, capture_output=True, check=True)
    return result.stdout


def main():
    assert git("rev-parse", "upstream/main").decode().strip() == CURRENT
    records = []
    commands = {
        "guard-commit.txt": ["show", "--format=fuller", "--stat", GUARD],
        "short-read-retry.patch": ["show", "--format=fuller", RETRY],
        "native-cpu-changes.patch": ["diff", BASE, CURRENT, "--", "src/kernels/cpu/native_expert.cpp"],
        "upstream-head.txt": ["show", "-s", "--format=fuller", CURRENT],
    }
    for name, args in commands.items():
        raw = git(*args)
        with (OUT / name).open("xb") as stream:
            stream.write(raw)
        records.append({"path": name, "git_arguments": args, "bytes": len(raw),
                        "sha256": hashlib.sha256(raw).hexdigest()})
    result = {"status": "read_only_source_audit", "observed_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
              "previous_frozen_upstream": BASE, "fetched_upstream": CURRENT,
              "current_tag": "v0.2.4.2", "working_source_changed": False, "model_run": False,
              "guard_commit": GUARD, "short_read_retry_commit": RETRY, "artifacts": records,
              "findings": [
                  "Upstream guard detects nonfinite head logits, drops contaminated session/cache state and supports one server retry.",
                  "The later retry change reads a prompt fully batched after nonfinite logits in its short prompt window.",
                  "Its reported short-read-zero success is external evidence, not a verified result on this host.",
                  "The native CPU Q8_1 from_float/dot route has no observed new sum-overflow fix in the retained diff.",
                  "Neither guard nor retry establishes a numerical root-cause repair.",
                  "Run75 retains its prior frozen source. A new current-upstream baseline needs separate build, admission and quality gates."
              ]}
    with (OUT / "audit.json").open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2)
    print(json.dumps({"status": result["status"], "artifacts": len(records), "upstream": CURRENT}))


if __name__ == "__main__":
    main()
