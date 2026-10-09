"""Small CPU-only copy/receipt fixtures; never open a production GGUF payload."""

from __future__ import annotations

import hashlib
import io
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from tools import hetero_copy_verified as copier


class VerifiedCopyTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="hetero-copy-fixture-")
        self.root = Path(self.temp.name)
        self.source = (self.root / "source.gguf").resolve()
        self.destination = (self.root / "out" / "copy.gguf").resolve()
        self.destination.parent.mkdir()
        self.payload = bytes(range(251)) * 13
        self.source.write_bytes(self.payload)
        self.manifest = self.root / "integrity.json"
        self.inventory = self.root / "inventory.json"
        self.receipt_path = self.root / "copy-receipt.json"
        self._write_manifest()
        self.inventory.write_text(json.dumps({"schema_version": 1, "collected_at_utc": "fixture",
                                               "storage": []}), encoding="utf-8")

    def tearDown(self):
        self.temp.cleanup()

    def _identity(self):
        st = self.source.stat()
        return {"size_bytes": st.st_size, "mtime_ns": st.st_mtime_ns, "ctime_ns": st.st_ctime_ns,
                "file_id": {"volume": st.st_dev, "index": st.st_ino}}

    def _write_manifest(self, matches=True, prior_sha=None):
        digest = hashlib.sha256(self.payload).hexdigest()
        stat = self.source.stat()
        after = {"size_bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns, "ctime_ns": stat.st_ctime_ns,
                 "file_id": {"volume": stat.st_dev, "index": stat.st_ino}}
        value = {"schema_version": 1, "run_id": "fixture-run", "status": "pass",
                 "claims": {"model_weight_integrity": True},
                 "model_shards": [{"path": str(self.source), "size_bytes": stat.st_size,
                                   "sha256": prior_sha or digest, "after": after,
                                   "expected_size_bytes": stat.st_size, "expected_sha256": digest,
                                   "matches_expected": matches}]}
        self.manifest.write_text(json.dumps(value), encoding="utf-8")

    def _plan(self, reserve=0):
        return copier.build_plan(self.source, self.destination, self.manifest, self.inventory,
                                 reserve_bytes=reserve, free_bytes=10**9)

    def test_validate_only_is_metadata_only_and_does_not_write(self):
        before = self.source.read_bytes()
        with mock.patch.object(copier, "_stream_hash", side_effect=AssertionError("payload hash called")), \
                mock.patch.object(copier, "_stream_copy_and_hash", side_effect=AssertionError("copy called")):
            plan = self._plan(reserve=12)
        self.assertEqual(plan["expected_size_bytes"], len(self.payload))
        self.assertFalse(self.destination.exists())
        self.assertFalse(self.receipt_path.exists())
        self.assertEqual(self.source.read_bytes(), before)

    def test_default_hardware_inventory_maps_e_and_f_without_guessing(self):
        self.assertTrue(copier.DEFAULT_INVENTORY.is_file())
        e = copier._inventory_mapping(copier.DEFAULT_INVENTORY, "E:\\")
        f = copier._inventory_mapping(copier.DEFAULT_INVENTORY, "F:\\")
        self.assertEqual(e["status"], "mapped_in_inventory")
        self.assertEqual(e["physical_disks"][0]["bus_type"], "NVMe")
        self.assertEqual(e["physical_disks"][0]["model"], "Micron_2450_MTFDKBA512TFK")
        self.assertEqual(f["status"], "mapped_in_inventory")
        self.assertEqual(f["physical_disks"][0]["bus_type"], "USB")
        self.assertEqual(f["physical_disks"][0]["model"], "Realtek RTL9201")
        self.assertIn("not a live", e["scope"])

    @unittest.skipUnless(os.name == "nt", "actual publication is intentionally Windows-only")
    def test_small_copy_uses_fresh_destination_readback_and_exclusive_receipt(self):
        plan = self._plan(reserve=0)
        result = copier.execute_copy(plan, "fixture-001", reserve_bytes=0, buffer_bytes=97)
        self.assertEqual(result["status"], "pass")
        self.assertEqual(result["source_stream_sha256"], plan["expected_sha256"])
        self.assertEqual(result["destination_readback_sha256"], plan["expected_sha256"])
        self.assertEqual(result["destination_readback_path"], result["temporary_path"])
        self.assertTrue(result["final_is_same_verified_file"])
        self.assertEqual(result["destination_identity"]["file_id"],
                         result["part_identity_after_readback"]["file_id"])
        self.assertEqual(self.destination.read_bytes(), self.payload)
        copier.write_receipt_exclusive(self.receipt_path, result)
        with self.assertRaises(FileExistsError):
            copier.write_receipt_exclusive(self.receipt_path, result)

    def test_short_writes_are_completed_and_hashed_by_written_chunks(self):
        class ShortWriter:
            def __init__(self):
                self.data = bytearray()

            def write(self, value):
                chunk = bytes(value[:7])
                self.data.extend(chunk)
                return len(chunk)

        writer = ShortWriter()
        observed = bytearray()
        count = copier._write_all(writer, b"short-write-fixture", lambda chunk: observed.extend(chunk))
        self.assertEqual(count, len(b"short-write-fixture"))
        self.assertEqual(bytes(writer.data), b"short-write-fixture")
        self.assertEqual(bytes(observed), b"short-write-fixture")

    def test_receipt_with_failed_hash_match_is_rejected_before_copy(self):
        self._write_manifest(matches=False)
        with self.assertRaisesRegex(copier.CopyError, "passing expected SHA"):
            self._plan()
        self.assertFalse(self.destination.exists())

    def test_inconsistent_prior_expected_digest_is_rejected(self):
        self._write_manifest(matches=True, prior_sha="0" * 64)
        with self.assertRaisesRegex(copier.CopyError, "expected size/SHA"):
            self._plan()

    @unittest.skipUnless(os.name == "nt", "actual publication is intentionally Windows-only")
    def test_stream_digest_failure_preserves_temporary_file(self):
        plan = self._plan()
        plan["expected_sha256"] = "0" * 64
        result = copier.execute_copy(plan, "wrong-digest", reserve_bytes=0, buffer_bytes=101)
        part = Path(result["temporary_path"])
        self.assertEqual(result["status"], "failed")
        self.assertTrue(part.is_file())
        self.assertFalse(self.destination.exists())
        self.assertIn("differs from prior expected", result["copy_error"]["message"])
        failure_receipt = self.root / "failed-copy.json"
        copier.write_receipt_exclusive(failure_receipt, result)
        saved = json.loads(failure_receipt.read_text(encoding="utf-8"))
        self.assertEqual(saved["status"], "failed")
        self.assertTrue(Path(saved["temporary_path"]).is_file())

    @unittest.skipUnless(os.name == "nt", "actual publication is intentionally Windows-only")
    def test_fresh_readback_mismatch_does_not_publish_final(self):
        plan = self._plan()
        with mock.patch.object(copier, "_stream_hash", return_value=("0" * 64, len(self.payload))):
            result = copier.execute_copy(plan, "readback-corrupt", reserve_bytes=0, buffer_bytes=91)
        self.assertEqual(result["status"], "failed")
        self.assertTrue(Path(result["temporary_path"]).is_file())
        self.assertFalse(self.destination.exists())
        self.assertIn("fresh temporary-file readback", result["copy_error"]["message"])

    @unittest.skipUnless(os.name == "nt", "actual publication is intentionally Windows-only")
    def test_source_change_during_copy_fails_and_preserves_partial(self):
        plan = self._plan()
        original = copier._stream_copy_and_hash

        def copy_then_mutate(source, part, buffer_bytes, progress=None):
            result = original(source, part, buffer_bytes, progress)
            source.write_bytes(b"changed while the copy was in progress")
            return result

        with mock.patch.object(copier, "_stream_copy_and_hash", side_effect=copy_then_mutate):
            result = copier.execute_copy(plan, "source-change", reserve_bytes=0, buffer_bytes=83)
        self.assertEqual(result["status"], "failed")
        self.assertTrue(Path(result["temporary_path"]).is_file())
        self.assertFalse(self.destination.exists())
        self.assertIn("source path/fd identity changed", result["copy_error"]["message"])

    @unittest.skipUnless(os.name == "nt", "actual publication is intentionally Windows-only")
    def test_interrupted_stream_receipt_keeps_partial_hash_and_byte_count(self):
        plan = self._plan()
        original = copier._stream_copy_and_hash

        def copy_then_fail(source, part, buffer_bytes, progress=None):
            original(source, part, buffer_bytes, progress)
            raise OSError("fixture interrupted after partial file was flushed")

        with mock.patch.object(copier, "_stream_copy_and_hash", side_effect=copy_then_fail):
            result = copier.execute_copy(plan, "interrupted", reserve_bytes=0, buffer_bytes=89)
        self.assertEqual(result["status"], "failed")
        self.assertTrue(Path(result["temporary_path"]).is_file())
        self.assertEqual(result["copied_bytes"], len(self.payload))
        self.assertEqual(result["source_stream_sha256"], plan["expected_sha256"])
        failpath = self.root / "interrupted-receipt.json"
        copier.write_receipt_exclusive(failpath, result)
        self.assertTrue(json.loads(failpath.read_text(encoding="utf-8"))["partial_preserved"])

    def test_existing_target_is_preserved(self):
        self.destination.write_bytes(b"keep this file")
        with self.assertRaisesRegex(copier.CopyError, "destination already exists"):
            self._plan()
        self.assertEqual(self.destination.read_bytes(), b"keep this file")

    def test_insufficient_free_space_is_rejected(self):
        with self.assertRaisesRegex(copier.CopyError, "free-space gate failed"):
            copier.build_plan(self.source, self.destination, self.manifest, self.inventory,
                              reserve_bytes=1000, free_bytes=len(self.payload) + 999)
        self.assertFalse(self.destination.exists())

    def test_receipt_path_failures_precede_plan_and_copy(self):
        invalid_receipts = [Path("relative-receipt.json"), self.root / "missing" / "receipt.json",
                            self.source, self.destination,
                            self.destination.with_name(self.destination.name + ".part.fixture-run")]
        for receipt in invalid_receipts:
            with self.subTest(receipt=receipt):
                with self.assertRaises(copier.CopyError):
                    copier.validate_receipt_path(receipt, self.source, self.destination, "fixture-run")

    def test_cli_rejects_relative_receipt_before_build_or_copy(self):
        invalid = ["relative.json", str(self.root / "missing" / "receipt.json")]
        for receipt in invalid:
            with self.subTest(receipt=receipt), \
                    mock.patch.object(copier, "build_plan", side_effect=AssertionError("plan must not run")) as plan, \
                    mock.patch.object(copier, "execute_copy", side_effect=AssertionError("copy must not run")) as copy, \
                    mock.patch("sys.stderr", io.StringIO()):
                rc = copier.main(["--copy", "--source", str(self.source), "--destination", str(self.destination),
                                  "--integrity-receipt", str(self.manifest), "--inventory", str(self.inventory),
                                  "--receipt", receipt, "--run-id", "fixture-run"])
            self.assertEqual(rc, 1)
            plan.assert_not_called()
            copy.assert_not_called()

    def test_cli_enforces_buffer_range(self):
        for invalid in ("0", str(64 * 1024 * 1024 + 1)):
            with self.subTest(invalid=invalid):
                with mock.patch("sys.stderr", io.StringIO()):
                    with self.assertRaises(SystemExit):
                        copier.main(["--source", str(self.source), "--destination", str(self.destination),
                                     "--buffer-bytes", invalid])

    def test_unsupported_publication_platform_is_explicitly_rejected(self):
        part = self.root / "candidate.part.test"
        part.write_bytes(b"candidate")
        with self.assertRaisesRegex(copier.CopyError, "only on Windows"):
            copier.publish_verified_part(part, self.destination, platform="posix")
        self.assertTrue(part.exists())
        self.assertFalse(self.destination.exists())

    def test_non_windows_copy_mode_refuses_before_payload_io(self):
        plan = self._plan()
        with mock.patch.object(copier, "_stream_copy_and_hash", side_effect=AssertionError("payload I/O happened")), \
                mock.patch.object(copier, "_stream_hash", side_effect=AssertionError("payload I/O happened")):
            result = copier.execute_copy(plan, "unsupported-platform", reserve_bytes=0, platform="posix")
        self.assertEqual(result["status"], "failed")
        self.assertFalse(self.destination.exists())
        self.assertFalse(Path(result["temporary_path"]).exists())

    def test_no_replace_publication_preserves_racing_existing_target(self):
        part = self.destination.with_name(self.destination.name + ".part.race")
        part.write_bytes(b"candidate")
        self.destination.write_bytes(b"target created concurrently")
        if os.name == "nt":
            error = "appeared before final rename"
        else:
            error = "only on Windows"
        with self.assertRaisesRegex(copier.CopyError, error):
            copier.publish_verified_part(part, self.destination, platform="nt")
        self.assertEqual(self.destination.read_bytes(), b"target created concurrently")
        self.assertTrue(part.exists())


if __name__ == "__main__":
    unittest.main()
