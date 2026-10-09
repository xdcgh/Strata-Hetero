"""CPU-only fixtures for the real-expert extractor; no model payload or accelerator is used."""

from __future__ import annotations

import struct
import tempfile
import unittest
from types import SimpleNamespace
from pathlib import Path
from unittest import mock

import numpy as np

from tools import hetero_extract_expert as extractor


class ExpertExtractorTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="hetero-extract-fixture-")
        self.root = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def test_profile_parser_reads_rank_without_touching_model(self):
        # A complete 1x2 profile with one ranked pair and its slot table.
        profile = self.root / "profile.bin"
        profile.write_bytes(b"STRP" + struct.pack("<5I", 1, 1, 2, 1, 1) +
                            struct.pack("<HH", 0, 1) + struct.pack("<2i", -1, 0))
        meta, pairs = extractor.read_profile(profile)
        self.assertEqual(meta["ranked_count"], 1)
        self.assertEqual(pairs, [(0, 1)])

    def test_native_v4_parser_preserves_per_role_shards_and_offsets(self):
        native = self.root / "native_experts.txt"
        native.write_text("# strata native experts v4: fixture\n"
                          "0 12 7 0 3072000 101 202 303 a.gguf,b.gguf,c.gguf\n", encoding="utf-8")
        entry = extractor.read_native_entry(native, 0)
        self.assertEqual(entry["shards"], {"gate": "a.gguf", "up": "b.gguf", "down": "c.gguf"})
        self.assertEqual((entry["gate_offset"], entry["up_offset"], entry["down_offset"]), (101, 202, 303))

    def test_path_ctime_survives_windows_fstat_zero_ctime(self):
        stat_fields = SimpleNamespace(st_size=100, st_mtime_ns=200, st_ctime_ns=300, st_dev=4, st_ino=5)
        fstat_fields = SimpleNamespace(st_size=100, st_mtime_ns=200, st_ctime_ns=0, st_dev=4, st_ino=5)
        with mock.patch.object(Path, "stat", return_value=stat_fields), \
                mock.patch.object(extractor.os, "open", return_value=10), \
                mock.patch.object(extractor.os, "fstat", return_value=fstat_fields), \
                mock.patch.object(extractor.os, "close"):
            result = extractor.file_stat(self.root / "fixture.gguf")
        self.assertEqual(result["ctime_ns"], 300)
        self.assertEqual(result["fstat_file_index"], 5)

    def test_file_stat_rejects_fstat_inode_or_mtime_mismatch(self):
        stat_fields = SimpleNamespace(st_size=100, st_mtime_ns=200, st_ctime_ns=300, st_dev=4, st_ino=5)
        fstat_fields = SimpleNamespace(st_size=100, st_mtime_ns=201, st_ctime_ns=0, st_dev=4, st_ino=6)
        with mock.patch.object(Path, "stat", return_value=stat_fields), \
                mock.patch.object(extractor.os, "open", return_value=10), \
                mock.patch.object(extractor.os, "fstat", return_value=fstat_fields), \
                mock.patch.object(extractor.os, "close"):
            with self.assertRaisesRegex(extractor.ExtractError, "path stat/fstat differ"):
                extractor.file_stat(self.root / "fixture.gguf")

    def test_pinned_official_dequantizer_handles_synthetic_small_rows(self):
        gguf, quants, _, _, _ = extractor._load_gguf(extractor.DEFAULT_GGUF_PY)
        for qtype, width in ((gguf.GGMLQuantizationType.Q4_K, 256),
                             (gguf.GGMLQuantizationType.Q5_1, 32)):
            block, size = gguf.GGML_QUANT_SIZES[qtype]
            packed = np.zeros((1, width // block * size), dtype=np.uint8)
            decoded = quants.dequantize(packed, qtype)
            self.assertEqual(decoded.shape, (1, width))
            self.assertTrue(np.isfinite(decoded).all())


if __name__ == "__main__":
    unittest.main()
