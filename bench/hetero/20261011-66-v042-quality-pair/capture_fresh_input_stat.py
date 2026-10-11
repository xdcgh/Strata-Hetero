from __future__ import annotations

import datetime as dt
import json
import os
from pathlib import Path

RUN = Path(__file__).resolve().parent
REPO = RUN.parents[2]
INTEGRITY = REPO / "bench/hetero/20261008-00-admission/model-integrity-02.json"
OUT = RUN / "fresh-input-identities-01.json"


def stat_record(path: Path) -> tuple[dict, dict]:
    before = path.stat(follow_symlinks=False)
    attrs = getattr(before, "st_file_attributes", 0)
    if path.is_symlink() or attrs & 0x400 or not path.is_file():
        raise RuntimeError(f"refusing non-regular or reparse input: {path}")
    with path.open("rb") as handle:
        opened = os.fstat(handle.fileno())
        # Metadata only: deliberately make no read() call on the handle.
    after = path.stat(follow_symlinks=False)
    file_id = {"volume": after.st_dev, "index": after.st_ino}
    before_key = (before.st_size, before.st_mtime_ns, before.st_ctime_ns, before.st_dev, before.st_ino)
    after_key = (after.st_size, after.st_mtime_ns, after.st_ctime_ns, after.st_dev, after.st_ino)
    fd_key = (opened.st_size, opened.st_mtime_ns, opened.st_dev, opened.st_ino)
    if before_key != after_key or fd_key != (after.st_size, after.st_mtime_ns, after.st_dev, after.st_ino):
        raise RuntimeError(f"stat/fstat identity changed while observing: {path}")
    identity = {
        "path": str(path),
        "size_bytes": after.st_size,
        "mtime_ns": after.st_mtime_ns,
        "ctime_ns": after.st_ctime_ns,
        "device": after.st_dev,
        "file_index": after.st_ino,
        "fstat": {
            "size_bytes": opened.st_size,
            "mtime_ns": opened.st_mtime_ns,
            "device": opened.st_dev,
            "file_index": opened.st_ino,
        },
    }
    identity_recheck = {
        "size_bytes": after.st_size,
        "mtime_ns": after.st_mtime_ns,
        "ctime_ns": after.st_ctime_ns,
        "file_id": file_id,
    }
    return identity, identity_recheck


prior = json.loads(INTEGRITY.read_text(encoding="utf-8"))
if prior.get("status") != "pass":
    raise RuntimeError("prior full-integrity receipt is not passing")

model_entries = []
for item in prior["model_shards"]:
    path = Path(item["path"])
    current, stat = stat_record(path)
    expected = item["after"]
    expected_file_id = expected["file_id"]
    matches = (
        stat["size_bytes"] == expected["size_bytes"]
        and stat["mtime_ns"] == expected["mtime_ns"]
        and stat["ctime_ns"] == expected["ctime_ns"]
        and stat["file_id"] == expected_file_id
        and item["sha256"] == item["expected_sha256"]
        and item["matches_expected"] is True
    )
    model_entries.append({
        "path": str(path),
        "role": "original F model shard; shared native-dense source in both arms",
        "current_identity": current,
        "identity_matches_prior_full_hash_receipt": matches,
        "prior_sha256": item["sha256"],
        "full_rehash_performed_now": False,
        "identity_source": str(INTEGRITY),
    })

runtime_entries = []
for category, group in prior["runtime_identity"].items():
    for item in group["files"]:
        path = Path(item["path"])
        _, stat = stat_record(path)
        expected = item["after"]
        matches = (
            stat["size_bytes"] == expected["size_bytes"]
            and stat["mtime_ns"] == expected["mtime_ns"]
            and stat["ctime_ns"] == expected["ctime_ns"]
            and stat["file_id"] == expected["file_id"]
        )
        runtime_entries.append({
            "category": category,
            "path": str(path),
            "stat": stat,
            "prior_full_sha256": item["sha256"],
            "matches_prior_stat": matches,
        })

if len(model_entries) != 4 or len(runtime_entries) != 14:
    raise RuntimeError(f"expected 4 model shards and 14 pack/MTP files; got {len(model_entries)} and {len(runtime_entries)}")

all_match = all(x["identity_matches_prior_full_hash_receipt"] for x in model_entries) and all(
    x["matches_prior_stat"] for x in runtime_entries
)
result = {
    "schema_version": 1,
    "observed_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
    "method": "fresh path stat plus opened-handle fstat; zero payload bytes read; compared with prior full-SHA/stat receipts",
    "all_current_identities_match": all_match,
    "input_files": model_entries,
    "runtime_identity_files": runtime_entries,
    "selected_ple_source": str(Path(r"F:\Strata-data\models\unsloth-UD-Q4_K_XL\Qwen3.8-Flash-Next-UD-Q4_K_XL-00002-of-00004.gguf")),
    "source_full_hash_receipt": str(INTEGRITY),
    "full_model_or_runtime_hash_performed_now": False,
    "payload_bytes_read": 0,
    "counts": {"model_shards": len(model_entries), "pack_mtp_runtime_files": len(runtime_entries)},
}
raw = (json.dumps(result, indent=2, ensure_ascii=False) + "\n").encode("utf-8")
OUT.write_bytes(raw)
print(json.dumps({
    "status": "pass" if all_match else "identity_mismatch",
    "output": str(OUT),
    "model_shards": len(model_entries),
    "pack_mtp_runtime_files": len(runtime_entries),
    "all_current_identities_match": all_match,
    "payload_bytes_read": 0,
}, separators=(",", ":")))
