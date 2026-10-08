#!/usr/bin/env python3
"""Hash existing model/runtime files and verify build receipts without running inference."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

TOTAL_MODEL_BYTES = 111_334_654_784
CHUNK_BYTES = 8 * 1024 * 1024
PROGRESS_BYTES = 512 * 1024 * 1024


class VerifyFailure(RuntimeError):
    pass


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def _is_reparse(st: os.stat_result) -> bool:
    return bool(getattr(st, "st_file_attributes", 0) & 0x400)


def file_stat(path: Path) -> dict[str, Any]:
    st = path.stat(follow_symlinks=False)
    if path.is_symlink() or _is_reparse(st):
        raise VerifyFailure(f"refusing symlink/reparse point: {path}")
    if not path.is_file():
        raise VerifyFailure(f"expected regular file: {path}")
    return {"size_bytes": st.st_size, "mtime_ns": st.st_mtime_ns, "ctime_ns": st.st_ctime_ns,
            "file_id": {"volume": st.st_dev, "index": st.st_ino}}


def stat_unchanged(before: dict[str, Any], after: dict[str, Any]) -> bool:
    return all(before[k] == after[k] for k in ("size_bytes", "mtime_ns", "ctime_ns", "file_id"))


def fd_matches_path(fd_stat: os.stat_result, path_stat: dict[str, Any]) -> bool:
    same = (fd_stat.st_size == path_stat["size_bytes"]
            and fd_stat.st_mtime_ns == path_stat["mtime_ns"]
            and {"volume": fd_stat.st_dev, "index": fd_stat.st_ino} == path_stat["file_id"])
    # On Windows, this Python runtime reports an invalid zero-epoch ctime from fstat;
    # path stat before and after the read remains the authoritative ctime comparison.
    if os.name != "nt":
        same = same and fd_stat.st_ctime_ns == path_stat["ctime_ns"]
    return same


class Progress:
    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.stream = path.open("xb")
        self.started = time.monotonic()

    def write(self, stage: str, **fields: Any) -> None:
        row = {"time_utc": utc_now(), "elapsed_seconds": round(time.monotonic() - self.started, 3),
               "stage": stage, **fields}
        line = json.dumps(row, ensure_ascii=True, separators=(",", ":")) + "\n"
        self.stream.write(line.encode("utf-8"))
        self.stream.flush()
        print(line.rstrip(), flush=True)

    def close(self) -> None:
        self.stream.close()


def parse_expected_shards(doc: Path) -> tuple[list[dict[str, Any]], list[int]]:
    text = doc.read_text(encoding="utf-8")
    tick = chr(96)
    pat = re.compile(
        r"^\|\s*" + tick + r"(Qwen3\.8-Flash-Next-UD-Q4_K_XL-\d{5}-of-00004\.gguf)" + tick +
        r"\s*\|\s*([\d,]+)\s*\|\s*" + tick + r"([0-9a-f]{64})" + tick + r"\s*\|$",
        re.MULTILINE)
    found = [{"name": n, "size_bytes": int(s.replace(",", "")), "sha256": h}
             for n, s, h in pat.findall(text)]
    if len(found) != 4:
        raise VerifyFailure(f"expected four Q4 shard rows in {doc}, found {len(found)}")
    names = [f"Qwen3.8-Flash-Next-UD-Q4_K_XL-{i:05d}-of-00004.gguf" for i in range(1, 5)]
    if [x["name"] for x in found] != names:
        raise VerifyFailure("formal table shard names/order differ from expected four-shard set")
    total = sum(x["size_bytes"] for x in found)
    if total != TOTAL_MODEL_BYTES:
        raise VerifyFailure(f"formal table total {total} != expected {TOTAL_MODEL_BYTES}")
    lines = [i for i, line in enumerate(text.splitlines(), 1)
             if "Qwen3.8-Flash-Next-UD-Q4_K_XL-" in line and ".gguf" in line
             and re.search(r"\|\s*" + tick + r"[0-9a-f]{64}" + tick + r"\s*\|", line)]
    return found, lines


def hash_file(path: Path, progress: Progress, stage: str, *,
              expected: dict[str, Any] | None = None, ordinal: int | None = None,
              total_files: int | None = None) -> dict[str, Any]:
    before = file_stat(path)
    if expected and before["size_bytes"] != expected["size_bytes"]:
        raise VerifyFailure(f"size mismatch before read: {path}: {before['size_bytes']} != {expected['size_bytes']}")
    digest = hashlib.sha256()
    size = 0
    next_report = PROGRESS_BYTES
    started = time.monotonic()
    with path.open("rb", buffering=0) as stream:
        opened = os.fstat(stream.fileno())
        opened_id = {"volume": opened.st_dev, "index": opened.st_ino}
        if not fd_matches_path(opened, before):
            raise VerifyFailure(f"file changed between stat and open: {path}")
        while True:
            block = stream.read(CHUNK_BYTES)
            if not block:
                break
            size += len(block)
            digest.update(block)
            while size >= next_report:
                progress.write("hash-progress", phase=stage, file=str(path), file_index=ordinal,
                               file_count=total_files, bytes_read=size, size_bytes=before["size_bytes"],
                               sha256_prefix=digest.hexdigest())
                next_report += PROGRESS_BYTES
        final_fd = os.fstat(stream.fileno())
    after = file_stat(path)
    actual_hash = digest.hexdigest()
    fd_same = fd_matches_path(final_fd, before)
    stable = stat_unchanged(before, after) and fd_same and size == before["size_bytes"]
    matches = ((size == expected["size_bytes"] and actual_hash == expected["sha256"])
               if expected else None)
    result = {"path": str(path), "size_bytes": size, "sha256": actual_hash,
              "before": before, "after": after, "unchanged_during_read": stable,
              "read_seconds": round(time.monotonic() - started, 3),
              "expected_size_bytes": expected["size_bytes"] if expected else None,
              "expected_sha256": expected["sha256"] if expected else None,
              "matches_expected": matches}
    progress.write("hash-complete", phase=stage, file=str(path), file_index=ordinal,
                   file_count=total_files, bytes_read=size, sha256=actual_hash,
                   unchanged_during_read=stable, matches_expected=matches)
    if not stable:
        raise VerifyFailure(f"file changed while hashing: {path}")
    if matches is False:
        raise VerifyFailure(f"SHA-256 mismatch for {path}; file preserved")
    return result


def list_regular_files(root: Path) -> list[Path]:
    if not root.is_dir() or root.is_symlink():
        raise VerifyFailure(f"runtime identity directory is missing or not a directory: {root}")
    result: list[Path] = []
    for current, dirs, names in os.walk(root, topdown=True, followlinks=False):
        base = Path(current)
        safe_dirs = []
        for name in sorted(dirs):
            child = base / name
            st = child.stat(follow_symlinks=False)
            if child.is_symlink() or _is_reparse(st):
                raise VerifyFailure(f"reparse directory refused in runtime tree: {child}")
            safe_dirs.append(name)
        dirs[:] = safe_dirs
        for name in sorted(names):
            child = base / name
            st = child.stat(follow_symlinks=False)
            if child.is_symlink() or _is_reparse(st):
                raise VerifyFailure(f"reparse file refused in runtime tree: {child}")
            if not child.is_file():
                raise VerifyFailure(f"non-regular runtime file: {child}")
            result.append(child)
    return sorted(result, key=lambda p: p.relative_to(root).as_posix().casefold())


def git_read(repo: Path, *args: str) -> str:
    p = subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True,
                       encoding="utf-8", errors="replace", check=False)
    if p.returncode:
        raise VerifyFailure(f"git {' '.join(args)} failed in {repo}: {p.stderr.strip()}")
    return p.stdout.strip()


def verify_build_receipt(receipt_path: Path, progress: Progress, ordinal: int) -> dict[str, Any]:
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if receipt.get("status") != "success" or receipt.get("exit_code") != 0:
        raise VerifyFailure(f"build receipt is not successful: {receipt_path}")
    binary = receipt.get("binary", {})
    binary_path = Path(binary.get("path", ""))
    actual = hash_file(binary_path, progress, "build-binary",
                       expected={"size_bytes": binary["size_bytes"], "sha256": binary["sha256"]},
                       ordinal=ordinal, total_files=2)
    source = receipt.get("source", {})
    item = {"receipt_path": str(receipt_path), "run": receipt.get("run"),
            "receipt_status": receipt.get("status"), "source_path": source.get("path"),
            "source_sha": source.get("sha") or source.get("head_at_receipt"), "binary": actual}
    if "compile_input_hashes" in source:
        current = []
        for record in source["compile_input_hashes"]:
            p = Path(source["path"]) / Path(record["path"])
            h = hashlib.sha256(p.read_bytes()).hexdigest() if p.is_file() else None
            expected_hash = record.get("sha256_at_receipt") or record.get("sha256")
            current.append({"path": str(p), "expected_sha256": expected_hash,
                            "current_sha256": h, "matches": h == expected_hash})
        item["compile_input_hashes_current"] = current
    elif source.get("path") and source.get("sha"):
        source_repo = Path(source["path"])
        head = git_read(source_repo, "rev-parse", "HEAD") if source_repo.exists() else None
        item["source_head_current"] = head
        item["source_head_matches_receipt"] = head == source["sha"]
    return item


def verify(args: argparse.Namespace) -> dict[str, Any]:
    progress = Progress(Path(args.progress_log))
    result: dict[str, Any] = {
        "schema_version": 1,
        "run_id": Path(args.progress_log).parent.name,
        "started_utc": utc_now(),
        "status": "running",
        "scope": {"model_dir": str(Path(args.model_dir)), "pack_dir": str(Path(args.pack_dir)),
                  "mtp_dir": str(Path(args.mtp_dir)), "documentation": str(Path(args.documentation)),
                  "single_threaded": True, "chunk_bytes": CHUNK_BYTES,
                  "progress_interval_bytes": PROGRESS_BYTES},
        "claims": {"model_weight_integrity": False, "runtime_identity_hashes_collected": False,
                   "build_receipts_and_binaries_verified": False,
                   "conversion_correctness_revalidated": False},
    }
    try:
        doc = Path(args.documentation)
        expected, table_lines = parse_expected_shards(doc)
        result["documentation"] = {"path": str(doc), "formal_table_line_numbers": table_lines,
                                   "documented_total_bytes": sum(x["size_bytes"] for x in expected),
                                   "shards": expected}
        progress.write("start", model_shards=len(expected),
                       model_total_bytes=sum(x["size_bytes"] for x in expected))
        model_dir = Path(args.model_dir)
        shards = []
        for index, item in enumerate(expected, 1):
            shards.append(hash_file(model_dir / item["name"], progress, "model-shard",
                                    expected=item, ordinal=index, total_files=len(expected)))
        result["model_shards"] = shards
        result["claims"]["model_weight_integrity"] = True

        runtime = {}
        for name, root_text in (("pack", args.pack_dir), ("mtp_runtime", args.mtp_dir)):
            root = Path(root_text)
            paths = list_regular_files(root)
            total = sum(file_stat(p)["size_bytes"] for p in paths)
            progress.write("runtime-tree-start", tree=name, file_count=len(paths), total_bytes=total)
            entries = [hash_file(p, progress, f"runtime-{name}", ordinal=i, total_files=len(paths))
                       for i, p in enumerate(paths, 1)]
            runtime[name] = {"root": str(root), "file_count": len(entries),
                             "total_bytes": sum(x["size_bytes"] for x in entries), "files": entries}
            progress.write("runtime-tree-complete", tree=name, file_count=len(entries),
                           total_bytes=runtime[name]["total_bytes"])
        result["runtime_identity"] = runtime
        result["claims"]["runtime_identity_hashes_collected"] = True

        builds = [verify_build_receipt(Path(path), progress, i)
                  for i, path in enumerate((args.baseline_receipt, args.feature_receipt), 1)]
        result["builds"] = builds
        ok = all(x["binary"]["matches_expected"] and x["binary"]["unchanged_during_read"] for x in builds)
        result["claims"]["build_receipts_and_binaries_verified"] = ok
        if not ok:
            raise VerifyFailure("one or more existing build binaries do not match their receipts")
        result["status"] = "pass"
        result["finished_utc"] = utc_now()
        progress.write("verification-complete", status="pass")
        return result
    except Exception as exc:
        result["status"] = "fail"
        result["failure"] = {"type": type(exc).__name__, "message": str(exc)}
        result["finished_utc"] = utc_now()
        progress.write("verification-failed", error_type=type(exc).__name__, message=str(exc))
        return result
    finally:
        progress.close()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--documentation", required=True)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--pack-dir", required=True)
    parser.add_argument("--mtp-dir", required=True)
    parser.add_argument("--baseline-receipt", required=True)
    parser.add_argument("--feature-receipt", required=True)
    parser.add_argument("--progress-log", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    output = Path(args.output)
    if output.exists() or output.is_symlink():
        print(f"refusing to overwrite evidence file: {output}", file=sys.stderr)
        return 2
    result = verify(args)
    try:
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("x", encoding="utf-8", newline="\n") as stream:
            json.dump(result, stream, indent=2, ensure_ascii=True)
            stream.write("\n")
        print(json.dumps({"status": result["status"], "output": str(output),
                          "failure": result.get("failure")}, ensure_ascii=True), flush=True)
    except FileExistsError:
        print(f"refusing to overwrite evidence file: {output}", file=sys.stderr)
        return 2
    return 0 if result["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
