"""Small CPU fixtures for source identity, original payload export and no-overwrite."""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from tools import hetero_native_inputs as tool


class NativeInputTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="hetero-native-input-fixture-")
        self.root = Path(self.temp.name)
        self.source = self.root / "source.gguf"
        self.parts = {"gate": b"\x13" * 4608, "up": b"\x24" * 4608, "down": b"\x35" * 6144}
        self.source.write_bytes(b"header" + b"".join(self.parts.values()))
        hashes = {role: tool.digest(data) for role, data in self.parts.items()}
        sizes = {role: len(data) for role, data in self.parts.items()}
        identity = {"layer": 0, "expert": 0, "source": {
            "raw_selected_payload_sha256": hashes, "raw_selected_payload_bytes": sizes,
            "source_file_stat_before_after": {str(self.source): tool.expert.file_stat(self.source)}}}
        self.identity = self.root / "identity.json"
        self.identity.write_text(json.dumps(identity), encoding="utf-8")
        tensors, offset = {}, 6
        for role, data in self.parts.items():
            tensors[role] = {"type_id": 7 if role == "down" else 12,
                             "shard": self.source.name, "data_offset": offset}
            offset += len(data)
        receipt = {"status": "extracted", "source_file_stat_unchanged": True,
                   "identity_path": str(self.identity), "selection": {"layer": 0, "expert": 0},
                   "dimensions": {"hidden": 256, "intermediate": 32}, "profile": {"experts": 1},
                   "raw_selected_payload_sha256": hashes, "raw_selected_payload_bytes": sizes,
                   "tensors": tensors}
        self.receipt = self.root / "extraction.json"
        self.receipt.write_text(json.dumps(receipt), encoding="utf-8")

    def tearDown(self):
        self.temp.cleanup()

    def test_metadata_does_not_read_payload_or_write_outputs(self):
        result, _ = tool.inspect(self.receipt)
        self.assertFalse(result["payload_read"])
        self.assertFalse(result["files_written"])
        self.assertEqual(len(list(self.root.iterdir())), 3)

    def test_exact_quantized_concat_and_input_recipe(self):
        result, _ = tool.inspect(self.receipt)
        target = self.root / "prepared"
        original = self.source.read_bytes()
        prepared = tool.prepare(result, target, seed=42)
        self.assertEqual((target / "expert.native.bin").read_bytes(), b"".join(self.parts.values()))
        x = np.random.default_rng(42 + 8).standard_normal((8, 256), dtype=np.float32) * np.float32(0.25)
        self.assertEqual((target / "input-008.f32").read_bytes(), x.astype("<f4").tobytes())
        self.assertEqual(self.source.read_bytes(), original)
        self.assertEqual(len(prepared["inputs"]), 9)

    def test_source_changed_since_receipt_is_refused(self):
        self.source.write_bytes(b"changed")
        with self.assertRaisesRegex(ValueError, "identity changed"):
            tool.inspect(self.receipt)

    def test_changed_payload_between_inspection_and_prepare_creates_nothing(self):
        result, _ = tool.inspect(self.receipt)
        self.source.write_bytes(b"tampered")
        target = self.root / "prepared"
        with self.assertRaisesRegex(ValueError, "identity changed"):
            tool.prepare(result, target)
        self.assertFalse(target.exists())

    def test_wrong_scoped_hash_creates_nothing(self):
        result, _ = tool.inspect(self.receipt)
        result["payloads"][0]["sha256"] = "0" * 64
        target = self.root / "prepared"
        with self.assertRaisesRegex(ValueError, "payload hash/length differs"):
            tool.prepare(result, target)
        self.assertFalse(target.exists())

    def test_existing_directory_is_preserved(self):
        result, _ = tool.inspect(self.receipt)
        target = self.root / "prepared"
        target.mkdir()
        sentinel = target / "sentinel"
        sentinel.write_bytes(b"retain")
        with self.assertRaises(FileExistsError):
            tool.prepare(result, target)
        self.assertEqual(sentinel.read_bytes(), b"retain")

    def test_out_of_range_payload_offset_is_refused(self):
        receipt = json.loads(self.receipt.read_text())
        receipt["tensors"]["down"]["data_offset"] = 2**63
        self.receipt.write_text(json.dumps(receipt))
        with self.assertRaisesRegex(ValueError, "outside source file"):
            tool.inspect(self.receipt)


if __name__ == "__main__":
    unittest.main()
