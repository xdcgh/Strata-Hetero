from __future__ import annotations

import contextlib
import hashlib
import io
import json
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch

import hetero_prepare_cuda as prepare


def _manifest(archive: bytes = b"pinned archive") -> dict:
    return {
        "release_label": "13.3.1",
        "architecture": "windows-x86_64",
        "source": "https://developer.download.nvidia.com/compute/cuda/redist/redistrib_13.3.1.json",
        "components": [{
            "component": "cuda_nvcc",
            "manifest_component": "cuda_nvcc",
            "version": "13.3.73",
            "relative_path": "cuda_nvcc/windows-x86_64/cuda_nvcc-windows-x86_64-13.3.73-archive.zip",
            "size_bytes": len(archive),
            "sha256": hashlib.sha256(archive).hexdigest(),
        }],
    }


def _write_zip(path: Path, entries: dict[str, bytes]) -> bytes:
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_STORED) as archive:
        for name, data in entries.items():
            archive.writestr(name, data)
    return path.read_bytes()


class ManifestTests(unittest.TestCase):
    def test_only_official_windows_x64_archive_paths_are_accepted(self):
        manifest = _manifest()
        prepare._validate_manifest(manifest)
        invalid_paths = [
            "../outside.zip",
            "/cuda_nvcc/windows-x86_64/file.zip",
            "C:/cuda_nvcc/windows-x86_64/file.zip",
            "cuda_nvcc/windows-x86_64/file.zip?download=1",
            "cuda_nvcc/windows-x86_64/../file.zip",
            "cuda_nvcc/windows-x86_64/file.zip:stream",
            "cuda_nvcc/windows-aarch64/file.zip",
            r"cuda_nvcc\windows-x86_64\file.zip",
        ]
        for path in invalid_paths:
            with self.subTest(path=path):
                altered = _manifest()
                altered["components"][0]["relative_path"] = path
                with self.assertRaises(prepare.PrepareError):
                    prepare._validate_manifest(altered)

    def test_manifest_source_cannot_redirect_downloads_to_another_host(self):
        manifest = _manifest()
        manifest["source"] = "https://example.invalid/redistrib_13.3.1.json"
        with self.assertRaises(prepare.PrepareError):
            prepare._validate_manifest(manifest)
        for path in ("cuda_nvcc/windows-x86_64/x.zip?x=1",
                     "cuda_nvcc/windows-x86_64/x.zip#fragment",
                     "cuda_nvcc/windows-x86_64/%2e%2e.zip"):
            with self.subTest(path=path), self.assertRaises(prepare.PrepareError):
                prepare._official_url(path)


class ReadOnlyValidationTests(unittest.TestCase):
    def test_validate_only_has_no_network_or_filesystem_side_effects(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            manifest_path = base / "manifest.json"
            manifest_path.write_text(json.dumps(_manifest()), encoding="utf-8")
            root = base / "new-parent" / "sdk"
            with patch.object(prepare.urllib.request, "urlopen",
                              side_effect=AssertionError("network access")), \
                    patch.object(prepare, "_download_one",
                                 side_effect=AssertionError("download attempted")):
                result = prepare.prepare(manifest_path, root, validate_only=True)
            self.assertEqual(result["status"], "validated")
            self.assertFalse(root.exists())
            self.assertFalse(root.parent.exists())
            self.assertFalse(root.parent / "archive" in list(base.rglob("archive")))

    def test_validate_only_refuses_an_unreceipted_existing_root_without_mutation(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            manifest_path = base / "manifest.json"
            manifest_path.write_text(json.dumps(_manifest()), encoding="utf-8")
            root = base / "sdk"
            root.mkdir()
            sentinel = root / "keep.txt"
            sentinel.write_text("leave me", encoding="utf-8")
            with self.assertRaises(prepare.PrepareError):
                prepare.prepare(manifest_path, root, validate_only=True)
            self.assertEqual(sentinel.read_text(encoding="utf-8"), "leave me")
            self.assertFalse((root / prepare.RECEIPT_NAME).exists())


class DownloadTests(unittest.TestCase):
    class Response:
        def __init__(self, payload: bytes, url: str):
            self._stream = io.BytesIO(payload)
            self._url = url

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            self._stream.close()

        def read(self, count: int = -1):
            return self._stream.read(count)

        def geturl(self):
            return self._url

    class Opener:
        def __init__(self, payload: bytes):
            self.payload = payload

        def open(self, request, timeout):
            self.timeout = timeout
            return DownloadTests.Response(
                self.payload,
                "https://developer.download.nvidia.com" + request.full_url.split(
                    "https://developer.download.nvidia.com", 1
                )[1],
            )

    def test_hash_mismatch_keeps_each_attempt_partial_and_never_publishes_archive(self):
        manifest = _manifest(b"expected")
        component = prepare._validate_manifest(manifest)[0]
        target = Path("unused.zip")
        with tempfile.TemporaryDirectory() as temporary:
            target = Path(temporary) / "archive" / "nvcc.zip"
            with patch.object(prepare.urllib.request, "build_opener",
                              return_value=self.Opener(b"wrong")), \
                    patch.object(prepare.time, "sleep"):
                with self.assertRaises(prepare.PrepareError):
                    prepare._download_one(component, target)
            self.assertFalse(target.exists())
            partials = sorted(target.parent.glob("*.part"))
            self.assertEqual(len(partials), prepare.DOWNLOAD_ATTEMPTS)
            self.assertEqual([p.read_bytes() for p in partials], [b"wrong"] * prepare.DOWNLOAD_ATTEMPTS)

    def test_download_accepts_only_exact_pinned_bytes_then_publishes_archive(self):
        payload = b"expected pinned archive bytes"
        component = prepare._validate_manifest(_manifest(payload))[0]
        with tempfile.TemporaryDirectory() as temporary:
            target = Path(temporary) / "archive" / "nvcc.zip"
            with patch.object(prepare.urllib.request, "build_opener",
                              return_value=self.Opener(payload)):
                result = prepare._download_one(component, target)
            self.assertEqual(result, target)
            self.assertEqual(prepare._cached_archive_status(target, component), "verified")
            self.assertEqual(list(target.parent.glob("*.part")), [])


class ZipSafetyTests(unittest.TestCase):
    def test_traversal_entry_is_rejected_before_any_file_is_written(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            archive_path = base / "bad.zip"
            _write_zip(archive_path, {
                "pkg/bin/nvcc.exe": b"ok",
                "pkg/../../escape.txt": b"escape",
            })
            stage = base / "stage"
            stage.mkdir()
            component = prepare._validate_manifest(_manifest())[0]
            with self.assertRaises(prepare.PrepareError):
                prepare._extract_merge(archive_path, stage, component)
            self.assertEqual(list(stage.rglob("*")), [])
            self.assertFalse((base / "escape.txt").exists())

    def test_drive_ads_and_symlink_entries_are_rejected(self):
        bad_names = [
            "pkg/C:/outside.txt",
            "pkg/bin/nvcc.exe:stream",
            "pkg/../outside.txt",
        ]
        for bad_name in bad_names:
            with self.subTest(name=bad_name), tempfile.TemporaryDirectory() as temporary:
                base = Path(temporary)
                archive_path = base / "bad.zip"
                _write_zip(archive_path, {bad_name: b"bad"})
                with zipfile.ZipFile(archive_path, "a") as archive:
                    info = zipfile.ZipInfo("pkg/bin/link")
                    info.create_system = 3
                    info.external_attr = (0o120777 << 16)
                    archive.writestr(info, "target")
                stage = base / "stage"
                stage.mkdir()
                component = prepare._validate_manifest(_manifest())[0]
                with self.assertRaises(prepare.PrepareError):
                    prepare._extract_merge(archive_path, stage, component)
                self.assertEqual(list(stage.rglob("*")), [])

    def test_different_content_collision_fails_without_overwriting_existing_file(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            archive_path = base / "collision.zip"
            _write_zip(archive_path, {"pkg/bin/tool.dll": b"new bytes"})
            stage = base / "stage"
            target = stage / "bin" / "tool.dll"
            target.parent.mkdir(parents=True)
            target.write_bytes(b"existing bytes")
            component = prepare._validate_manifest(_manifest())[0]
            with self.assertRaisesRegex(prepare.PrepareError, "content collision"):
                prepare._extract_merge(archive_path, stage, component)
            self.assertEqual(target.read_bytes(), b"existing bytes")

    def test_different_component_root_licenses_are_preserved_separately(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            stage = base / "stage"
            stage.mkdir()
            first = base / "first.zip"
            second = base / "second.zip"
            _write_zip(first, {"pkg-one/LICENSE": b"license for one"})
            _write_zip(second, {"pkg-two/LICENSE": b"license for two"})
            component1 = {"component": "cuda_alpha"}
            component2 = {"component": "cuda_beta"}
            prepare._extract_merge(first, stage, component1)
            prepare._extract_merge(second, stage, component2)
            self.assertEqual((stage / "component-metadata/cuda_alpha/LICENSE").read_bytes(), b"license for one")
            self.assertEqual((stage / "component-metadata/cuda_beta/LICENSE").read_bytes(), b"license for two")

    def test_different_component_sdk_headers_still_fail_closed(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            stage = base / "stage"
            stage.mkdir()
            first = base / "first.zip"
            second = base / "second.zip"
            _write_zip(first, {"pkg-one/include/same.h": b"header one"})
            _write_zip(second, {"pkg-two/include/same.h": b"header two"})
            prepare._extract_merge(first, stage, {"component": "cuda_alpha"})
            with self.assertRaisesRegex(prepare.PrepareError, "content collision"):
                prepare._extract_merge(second, stage, {"component": "cuda_beta"})
            self.assertEqual((stage / "include/same.h").read_bytes(), b"header one")

    def test_identical_content_collision_is_reused(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            archive_path = base / "same.zip"
            _write_zip(archive_path, {"pkg/bin/tool.dll": b"same"})
            stage = base / "stage"
            target = stage / "bin" / "tool.dll"
            target.parent.mkdir(parents=True)
            target.write_bytes(b"same")
            component = prepare._validate_manifest(_manifest())[0]
            self.assertEqual(prepare._extract_merge(archive_path, stage, component)["bin/tool.dll"],
                             (4, hashlib.sha256(b"same").hexdigest()))
            self.assertEqual(target.read_bytes(), b"same")


class ReceiptTests(unittest.TestCase):
    def test_existing_sdk_reuse_requires_matching_manifest_and_all_file_hashes(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "sdk"
            root.mkdir()
            nvcc = root / "bin" / "nvcc.exe"
            nvcc.parent.mkdir()
            nvcc.write_bytes(b"test compiler")
            manifest = _manifest()
            raw = json.dumps(manifest).encode()
            components = prepare._validate_manifest(manifest)
            size, digest = prepare._sha256_file(nvcc)
            receipt = {
                "schema_version": prepare.SCHEMA_VERSION,
                "ready": True,
                "metadata_collision_policy": prepare.METADATA_POLICY,
                "manifest_sha256": hashlib.sha256(raw).hexdigest(),
                "sdk_root": prepare._root_identity(root),
                "archives": [{"component": "cuda_nvcc", "size_bytes": components[0]["size_bytes"],
                              "sha256": components[0]["sha256"]}],
                "files": [{"path": "bin/nvcc.exe", "size_bytes": size, "sha256": digest}],
                "file_count": 1,
                "nvcc_relative_path": "bin/nvcc.exe",
                "nvcc_path": str(nvcc),
            }
            (root / prepare.RECEIPT_NAME).write_text(json.dumps(receipt), encoding="utf-8")
            verified = prepare._check_existing_root(root, hashlib.sha256(raw).hexdigest(), components)
            self.assertEqual(verified["file_count"], 1)
            nvcc.write_bytes(b"changed")
            with self.assertRaisesRegex(prepare.PrepareError, "differs from receipt"):
                prepare._check_existing_root(root, hashlib.sha256(raw).hexdigest(), components)


if __name__ == "__main__":
    unittest.main()
