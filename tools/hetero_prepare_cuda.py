#!/usr/bin/env python3
"""Prepare a pinned CUDA redistributable SDK tree without installing it.

Different NVIDIA packages may contain different root LICENSE or Version.json
files. Those exact package-root metadata files are retained under
component-metadata/<component>/; all other differing SDK paths are collisions.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import stat
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
import zipfile
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any

OFFICIAL_PREFIX = "https://developer.download.nvidia.com/compute/cuda/redist/"
DEFAULT_ROOT = r"E:\Strata-Hetero-data\toolchains\cuda-13.3.1"
RECEIPT_NAME = "_strata_cuda_prepare_receipt.json"
CHUNK_BYTES = 1024 * 1024
DOWNLOAD_TIMEOUT_SECONDS = 45
DOWNLOAD_ATTEMPTS = 3
SCHEMA_VERSION = 1
METADATA_POLICY = (
    "Only package-root LICENSE, LICENSE.txt, LICENSE.md, Version.json, and "
    "version.json are stored separately under component-metadata/<component>/. "
    "Every other different-content path collision fails without overwriting."
)
ROOT_METADATA_NAMES = {"LICENSE", "LICENSE.txt", "LICENSE.md", "Version.json", "version.json"}


class PrepareError(RuntimeError):
    """A validation or preparation failure that should be reported to the CLI."""


def _sha256_file(path: Path) -> tuple[int, str]:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as stream:
        while True:
            chunk = stream.read(CHUNK_BYTES)
            if not chunk:
                break
            size += len(chunk)
            digest.update(chunk)
    return size, digest.hexdigest()


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _inside(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False


def _safe_relative_path(value: Any, *, label: str) -> PurePosixPath:
    if not isinstance(value, str) or not value or "\x00" in value:
        raise PrepareError(f"{label}: expected a non-empty relative path")
    if "\\" in value:
        raise PrepareError(f"{label}: backslashes are not permitted")
    posix = PurePosixPath(value)
    win = PureWindowsPath(value)
    if posix.is_absolute() or win.is_absolute() or win.drive or value.startswith("/"):
        raise PrepareError(f"{label}: absolute or drive-qualified path refused")
    if any(part in {"", ".", ".."} for part in value.split("/")):
        raise PrepareError(f"{label}: empty, dot, or parent path component refused")
    if any(c in value for c in map(chr, (60, 62, 58, 34, 124, 63, 42, 37))):
        raise PrepareError(f"{label}: Windows-invalid, URL-control, drive, or alternate data stream name refused")
    if any(part.endswith((".", " ")) for part in value.split("/")):
        raise PrepareError(f"{label}: Windows trailing-dot or trailing-space name refused")
    reserved = {"CON", "PRN", "AUX", "NUL", *(f"COM{i}" for i in range(1, 10)), *(f"LPT{i}" for i in range(1, 10))}
    if any(part.split(".", 1)[0].upper() in reserved for part in value.split("/")):
        raise PrepareError(f"{label}: reserved Windows device name refused")
    if any(ord(char) < 32 for char in value):
        raise PrepareError(f"{label}: control characters are not permitted")
    return posix


def _validate_manifest(manifest: Any) -> list[dict[str, Any]]:
    if not isinstance(manifest, dict):
        raise PrepareError("manifest root must be a JSON object")
    if manifest.get("architecture") != "windows-x86_64":
        raise PrepareError("manifest architecture must be windows-x86_64")
    release = manifest.get("release_label")
    if not isinstance(release, str) or not re.fullmatch(r"\d+\.\d+\.\d+", release):
        raise PrepareError("manifest release_label is invalid")
    source = manifest.get("source")
    expected_source = f"https://developer.download.nvidia.com/compute/cuda/redist/redistrib_{release}.json"
    if source != expected_source:
        raise PrepareError("manifest source is not the pinned NVIDIA official redistrib URL")
    raw_components = manifest.get("components")
    if not isinstance(raw_components, list) or not raw_components:
        raise PrepareError("manifest components must be a non-empty list")
    names: set[str] = set()
    paths: set[str] = set()
    result: list[dict[str, Any]] = []
    for index, item in enumerate(raw_components):
        label = f"manifest component {index + 1}"
        if not isinstance(item, dict):
            raise PrepareError(f"{label}: expected an object")
        name = item.get("component")
        manifest_name = item.get("manifest_component", name)
        if not isinstance(name, str) or not re.fullmatch(r"[a-zA-Z0-9_.-]+", name):
            raise PrepareError(f"{label}: invalid component name")
        if not isinstance(manifest_name, str) or not re.fullmatch(r"[a-zA-Z0-9_.-]+", manifest_name):
            raise PrepareError(f"{label}: invalid manifest component name")
        if name in names:
            raise PrepareError(f"duplicate component name: {name}")
        names.add(name)
        relative = _safe_relative_path(item.get("relative_path"), label=f"{name} relative_path")
        if len(relative.parts) < 3 or relative.parts[1] != "windows-x86_64":
            raise PrepareError(f"{name}: URL path must be under windows-x86_64")
        if relative.parts[0] != manifest_name or relative.suffix.casefold() != ".zip":
            raise PrepareError(f"{name}: URL path must match its manifest component and end in .zip")
        relative_text = relative.as_posix()
        if relative_text in paths:
            raise PrepareError(f"duplicate component archive path: {relative_text}")
        paths.add(relative_text)
        size = item.get("size_bytes")
        digest = item.get("sha256")
        version = item.get("version")
        if isinstance(size, bool) or not isinstance(size, int) or size <= 0:
            raise PrepareError(f"{name}: size_bytes must be a positive integer")
        if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-fA-F]{64}", digest):
            raise PrepareError(f"{name}: sha256 must contain 64 hex digits")
        if not isinstance(version, str) or not version:
            raise PrepareError(f"{name}: version is required")
        result.append({"component": name, "manifest_component": manifest_name,
                       "version": version, "relative_path": relative_text,
                       "size_bytes": size, "sha256": digest.lower()})
    return result


def _official_url(relative_path: str) -> str:
    relative = _safe_relative_path(relative_path, label="download path")
    if len(relative.parts) < 3 or relative.parts[1] != "windows-x86_64" or relative.suffix.casefold() != ".zip":
        raise PrepareError("download path is not a windows-x86_64 ZIP path")
    url = OFFICIAL_PREFIX + relative.as_posix()
    parsed = urllib.parse.urlsplit(url)
    if (parsed.scheme != "https" or parsed.hostname != "developer.download.nvidia.com"
            or parsed.query or parsed.fragment or parsed.username or parsed.password):
        raise PrepareError("download URL is outside NVIDIA's official HTTPS host")
    return url


def _load_manifest(path: Path) -> tuple[dict[str, Any], bytes, list[dict[str, Any]]]:
    try:
        raw = path.read_bytes()
        manifest = json.loads(raw.decode("utf-8-sig"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise PrepareError(f"cannot read manifest: {exc}") from exc
    components = _validate_manifest(manifest)
    return manifest, raw, components


def _root_identity(root: Path) -> str:
    return os.path.normcase(os.path.abspath(os.fspath(root)))


def _check_existing_root(root: Path, manifest_sha256: str, components: list[dict[str, Any]]) -> dict[str, Any]:
    if root.is_symlink() or not root.is_dir():
        raise PrepareError(f"SDK root exists but is not a plain directory: {root}")
    receipt_path = root / RECEIPT_NAME
    if receipt_path.is_symlink() or not receipt_path.is_file():
        raise PrepareError(f"SDK root exists without a trusted receipt; refusing reuse: {root}")
    try:
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise PrepareError(f"SDK receipt is unreadable: {exc}") from exc
    if not isinstance(receipt, dict) or receipt.get("schema_version") != SCHEMA_VERSION:
        raise PrepareError("SDK receipt schema is invalid")
    if receipt.get("ready") is not True or receipt.get("manifest_sha256") != manifest_sha256:
        raise PrepareError("SDK receipt does not match the selected pinned manifest")
    if receipt.get("sdk_root") != _root_identity(root):
        raise PrepareError("SDK receipt root path does not match the requested root")
    if receipt.get("metadata_collision_policy") != METADATA_POLICY:
        raise PrepareError("SDK receipt does not record the current component-metadata policy")
    expected_archives = [{"component": c["component"], "size_bytes": c["size_bytes"], "sha256": c["sha256"]}
                         for c in components]
    if receipt.get("archives") != expected_archives:
        raise PrepareError("SDK receipt archive hashes do not match the selected manifest")
    files = receipt.get("files")
    if not isinstance(files, list):
        raise PrepareError("SDK receipt has no installed-file verification list")
    expected: dict[str, tuple[int, str]] = {}
    for entry in files:
        if not isinstance(entry, dict):
            raise PrepareError("SDK receipt contains an invalid file record")
        rel = _safe_relative_path(entry.get("path"), label="receipt file path").as_posix()
        if rel.casefold() == RECEIPT_NAME.casefold() or rel in expected:
            raise PrepareError("SDK receipt contains a duplicate or reserved file path")
        size, digest = entry.get("size_bytes"), entry.get("sha256")
        if isinstance(size, bool) or not isinstance(size, int) or size < 0 or not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest):
            raise PrepareError("SDK receipt contains an invalid file size or digest")
        expected[rel] = (size, digest)
    observed: set[str] = set()
    for directory, dirs, filenames in os.walk(root, followlinks=False):
        base = Path(directory)
        for dirname in dirs:
            child = base / dirname
            if child.is_symlink():
                raise PrepareError(f"SDK root contains a symbolic link: {child}")
        for filename in filenames:
            child = base / filename
            rel = child.relative_to(root).as_posix()
            if rel == RECEIPT_NAME:
                continue
            if child.is_symlink() or not child.is_file():
                raise PrepareError(f"SDK root contains a non-regular file: {child}")
            if rel not in expected:
                raise PrepareError(f"SDK root contains a file absent from its receipt: {rel}")
            size, digest = _sha256_file(child)
            if expected[rel] != (size, digest):
                raise PrepareError(f"SDK file differs from receipt: {rel}")
            observed.add(rel)
    if observed != set(expected):
        raise PrepareError("SDK root is missing one or more files listed in its receipt")
    nvcc_rel = receipt.get("nvcc_relative_path")
    safe_nvcc = _safe_relative_path(nvcc_rel, label="receipt nvcc path").as_posix()
    if safe_nvcc.casefold() != "bin/nvcc.exe" or not (root / Path(*PurePosixPath(safe_nvcc).parts)).is_file():
        raise PrepareError("SDK receipt does not point to bin/nvcc.exe")
    expected_nvcc = os.fspath(root / "bin" / "nvcc.exe")
    if receipt.get("nvcc_path") != expected_nvcc:
        raise PrepareError("SDK receipt absolute nvcc path does not match the requested root")
    if receipt.get("file_count") != len(expected):
        raise PrepareError("SDK receipt file count does not match its file list")
    return receipt


def _archive_path(root: Path, component: dict[str, Any]) -> Path:
    archive_dir = root.parent / "archive"
    name = f"{component['component']}-{component['version']}-{component['sha256']}.zip"
    return archive_dir / name


def _validate_archive_directory(root: Path) -> None:
    directory = root.parent / "archive"
    _validate_archive_path(directory)


def _validate_archive_path(directory: Path) -> None:
    if directory.is_symlink():
        raise PrepareError(f"archive directory is a symbolic link: {directory}")
    if directory.exists() and not directory.is_dir():
        raise PrepareError(f"archive path is not a directory: {directory}")


def _cached_archive_status(path: Path, component: dict[str, Any]) -> str:
    if not path.exists() and not path.is_symlink():
        return "missing"
    if path.is_symlink() or not path.is_file():
        raise PrepareError(f"cached archive is not a regular file: {path}")
    size, digest = _sha256_file(path)
    if size != component["size_bytes"] or digest != component["sha256"]:
        raise PrepareError(f"cached archive failed pinned size/SHA-256 verification: {path}")
    return "verified"


def _no_redirect(_url: str, _fp: Any, _code: int, _msg: str, _headers: Any, _newurl: str) -> None:
    raise urllib.error.HTTPError(_url, _code, "redirects are refused for pinned NVIDIA downloads", _headers, _fp)


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req: Any, fp: Any, code: int, msg: str, headers: Any, newurl: str) -> None:
        _no_redirect(req.full_url, fp, code, msg, headers, newurl)


def _download_one(component: dict[str, Any], target: Path) -> Path:
    url = _official_url(component["relative_path"])
    _validate_archive_path(target.parent)
    if _cached_archive_status(target, component) == "verified":
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    opener = urllib.request.build_opener(_NoRedirect())
    failures: list[str] = []
    for attempt in range(1, DOWNLOAD_ATTEMPTS + 1):
        part = target.parent / f"{target.name}.attempt-{attempt}-{uuid.uuid4().hex}.part"
        digest = hashlib.sha256()
        size = 0
        try:
            request = urllib.request.Request(url, headers={"User-Agent": "Strata-Hetero-CUDA-Prepare/1"})
            with opener.open(request, timeout=DOWNLOAD_TIMEOUT_SECONDS) as response, part.open("xb") as output:
                final = urllib.parse.urlsplit(response.geturl())
                if final.scheme != "https" or final.hostname != "developer.download.nvidia.com":
                    raise PrepareError("download response escaped NVIDIA's official HTTPS host")
                while True:
                    chunk = response.read(CHUNK_BYTES)
                    if not chunk:
                        break
                    size += len(chunk)
                    if size > component["size_bytes"]:
                        raise PrepareError("download exceeded the pinned byte count")
                    output.write(chunk)
                    digest.update(chunk)
                output.flush()
                os.fsync(output.fileno())
            if size != component["size_bytes"] or digest.hexdigest() != component["sha256"]:
                raise PrepareError(f"download verification failed (received {size} bytes, SHA-256 {digest.hexdigest()})")
            if target.exists() or target.is_symlink():
                if _cached_archive_status(target, component) == "verified":
                    return target
                raise PrepareError(f"refusing to overwrite archive path: {target}")
            os.rename(part, target)
            return target
        except Exception as exc:
            failures.append(str(exc))
            # Keep every attempt's .part file as evidence; the next try gets a new name.
            if attempt < DOWNLOAD_ATTEMPTS:
                time.sleep(0.25 * attempt)
    raise PrepareError(f"download failed after {DOWNLOAD_ATTEMPTS} attempts; partials retained: {'; '.join(failures)}")


def _zip_members(archive: zipfile.ZipFile, component: str) -> tuple[str, list[tuple[zipfile.ZipInfo, PurePosixPath]]]:
    mapped: list[tuple[zipfile.ZipInfo, PurePosixPath]] = []
    top_names: set[str] = set()
    for info in archive.infolist():
        if info.flag_bits & 0x1:
            raise PrepareError(f"{component}: encrypted ZIP entries are refused")
        raw_name = info.filename.rstrip("/")
        rel = _safe_relative_path(raw_name, label=f"{component} ZIP entry")
        if len(rel.parts) == 1 and info.is_dir():
            top_names.add(rel.parts[0])
            continue
        if len(rel.parts) < 2:
            raise PrepareError(f"{component}: every ZIP entry must be under a package root folder")
        top_names.add(rel.parts[0])
        mode = (info.external_attr >> 16) & 0xFFFF
        kind = stat.S_IFMT(mode)
        if kind not in (0, stat.S_IFREG, stat.S_IFDIR):
            raise PrepareError(f"{component}: special or symbolic-link ZIP entry refused: {info.filename}")
        dos_attributes = info.external_attr & 0xFFFF
        if dos_attributes & 0x400:
            raise PrepareError(f"{component}: reparse-point ZIP entry refused: {info.filename}")
        mapped.append((info, PurePosixPath(*rel.parts[1:])))
    if len(top_names) != 1:
        raise PrepareError(f"{component}: archive must contain exactly one package root folder")
    if not mapped:
        raise PrepareError(f"{component}: archive is empty")
    return next(iter(top_names)), mapped


def _extract_merge(archive_path: Path, stage: Path, component: dict[str, Any]) -> dict[str, tuple[int, str]]:
    label = component["component"]
    try:
        with zipfile.ZipFile(archive_path, "r") as archive:
            _package_root, members = _zip_members(archive, label)
            result: dict[str, tuple[int, str]] = {}
            for info, relative in members:
                if len(relative.parts) == 1 and not info.is_dir() and relative.name in ROOT_METADATA_NAMES:
                    relative = PurePosixPath("component-metadata", label, relative.name)
                target = stage.joinpath(*relative.parts)
                if not _inside(target.resolve(strict=False), stage.resolve()):
                    raise PrepareError(f"{label}: ZIP path escapes the staging root: {info.filename}")
                if info.is_dir():
                    if target.exists() and not target.is_dir():
                        raise PrepareError(f"file/directory collision at {relative.as_posix()}")
                    target.mkdir(parents=True, exist_ok=True)
                    continue
                target.parent.mkdir(parents=True, exist_ok=True)
                digest = hashlib.sha256()
                size = 0
                with archive.open(info, "r") as source:
                    while True:
                        chunk = source.read(CHUNK_BYTES)
                        if not chunk:
                            break
                        size += len(chunk)
                        digest.update(chunk)
                if size != info.file_size:
                    raise PrepareError(f"{label}: ZIP entry size mismatch: {info.filename}")
                actual = digest.hexdigest()
                if target.exists():
                    if target.is_symlink() or not target.is_file():
                        raise PrepareError(f"file collision at {relative.as_posix()}")
                    old_size, old_digest = _sha256_file(target)
                    if (old_size, old_digest) != (size, actual):
                        raise PrepareError(f"content collision at {relative.as_posix()}; existing file preserved")
                else:
                    written = hashlib.sha256()
                    written_size = 0
                    with archive.open(info, "r") as source, target.open("xb") as output:
                        while True:
                            chunk = source.read(CHUNK_BYTES)
                            if not chunk:
                                break
                            output.write(chunk)
                            written.update(chunk)
                            written_size += len(chunk)
                    if (written_size, written.hexdigest()) != (size, actual):
                        raise PrepareError(f"{label}: entry changed while extracting: {info.filename}")
                result[relative.as_posix()] = (size, actual)
            return result
    except (OSError, zipfile.BadZipFile, RuntimeError) as exc:
        if isinstance(exc, PrepareError):
            raise
        raise PrepareError(f"{label}: cannot extract verified archive: {exc}") from exc


def _build_sdk(root: Path, manifest: dict[str, Any], raw_manifest: bytes,
               components: list[dict[str, Any]]) -> dict[str, Any]:
    if root.exists() or root.is_symlink():
        raise PrepareError(f"SDK root appeared during preparation; refusing to overwrite: {root}")
    stage = root.parent / f".{root.name}.stage-{uuid.uuid4().hex}"
    stage.mkdir(parents=True, exist_ok=False)
    installed: dict[str, tuple[int, str]] = {}
    archive_records = []
    for component in components:
        archive_path = _archive_path(root, component)
        if _cached_archive_status(archive_path, component) != "verified":
            archive_path = _download_one(component, archive_path)
        archive_records.append({"component": component["component"], "size_bytes": component["size_bytes"],
                                "sha256": component["sha256"]})
        extracted = _extract_merge(archive_path, stage, component)
        for relative, info in extracted.items():
            prior = installed.get(relative)
            if prior is not None and prior != info:
                raise PrepareError(f"content collision at {relative}; existing staged file preserved")
            installed[relative] = info
    nvcc_relative = "bin/nvcc.exe"
    if nvcc_relative not in installed:
        raise PrepareError("prepared archives did not provide bin/nvcc.exe")
    files = [{"path": path, "size_bytes": item[0], "sha256": item[1]}
             for path, item in sorted(installed.items())]
    receipt = {"schema_version": SCHEMA_VERSION, "ready": True,
               "release_label": manifest["release_label"],
               "metadata_collision_policy": METADATA_POLICY,
               "manifest_sha256": _sha256_bytes(raw_manifest),
               "sdk_root": _root_identity(root), "archives": archive_records,
               "file_count": len(files), "total_file_bytes": sum(x["size_bytes"] for x in files),
               "files": files, "nvcc_relative_path": nvcc_relative,
               "nvcc_path": os.fspath(root / "bin" / "nvcc.exe")}
    receipt_path = stage / RECEIPT_NAME
    with receipt_path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(receipt, stream, indent=2, ensure_ascii=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    if root.exists() or root.is_symlink():
        raise PrepareError(f"SDK root appeared before finalization; staging preserved at {stage}")
    try:
        os.rename(stage, root)
    except OSError as exc:
        raise PrepareError(f"cannot publish SDK root; staging preserved at {stage}: {exc}") from exc
    return receipt


def prepare(manifest_path: Path, root: Path, *, validate_only: bool) -> dict[str, Any]:
    manifest, raw, components = _load_manifest(manifest_path)
    root = Path(os.path.abspath(os.fspath(root)))
    if root == Path(root.anchor) or root.name in {"", ".", ".."}:
        raise PrepareError("SDK root must be a named directory, not a filesystem root")
    _validate_archive_directory(root)
    manifest_digest = _sha256_bytes(raw)
    if root.exists() or root.is_symlink():
        receipt = _check_existing_root(root, manifest_digest, components)
        return {"status": "ready", "reused": True, "sdk_root": os.fspath(root),
                "nvcc_path": receipt["nvcc_path"], "file_count": receipt["file_count"]}
    plan = []
    for component in components:
        cached = _cached_archive_status(_archive_path(root, component), component)
        plan.append({"component": component["component"], "archive_status": cached,
                     "archive_path": os.fspath(_archive_path(root, component)),
                     "download_bytes": 0 if cached == "verified" else component["size_bytes"]})
    if validate_only:
        return {"status": "validated", "reused": False, "validate_only": True,
                "manifest_sha256": manifest_digest, "sdk_root": os.fspath(root),
                "archive_dir": os.fspath(root.parent / "archive"), "components": plan,
                "download_bytes": sum(x["download_bytes"] for x in plan)}
    root.parent.mkdir(parents=True, exist_ok=True)
    receipt = _build_sdk(root, manifest, raw, components)
    # Hash and verify the just-published receipt before reporting ready.
    verified = _check_existing_root(root, manifest_digest, components)
    return {"status": "ready", "reused": False, "sdk_root": os.fspath(root),
            "nvcc_path": verified["nvcc_path"], "file_count": verified["file_count"]}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, help="pinned NVIDIA redistrib manifest JSON")
    parser.add_argument("--root", default=DEFAULT_ROOT, help=f"prepared SDK root (default: {DEFAULT_ROOT})")
    parser.add_argument("--validate-only", action="store_true", help="validate inputs and cache without network or writes")
    args = parser.parse_args(argv)
    try:
        result = prepare(Path(args.manifest), Path(args.root), validate_only=args.validate_only)
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return 0
    except Exception as exc:
        print(f"CUDA preparation failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
