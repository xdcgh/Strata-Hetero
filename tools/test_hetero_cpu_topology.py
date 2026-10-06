from __future__ import annotations

import json
import struct
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import hetero_cpu_topology as topo


def cpu_set(group=0, lp=0, core=0, *, cpu_id=0, eff=0, flags=0, size=32, tag=0x1122334455667788):
    if size < 32:
        return struct.pack("<II", size, 0) + bytes(max(0, size - 8))
    b = bytearray(size)
    struct.pack_into("<II", b, 0, size, 0)
    struct.pack_into("<IH6BIQ", b, 8, cpu_id, group, lp, core, 3, 4, eff, flags, 5, tag)
    return bytes(b)


class CpuTopologyTests(unittest.TestCase):
    def test_variable_size_entries_and_unknown_type_skip(self):
        unknown = struct.pack("<II", 40, 91) + bytes(32)
        row = topo.parse_windows_cpu_set_buffer(unknown + cpu_set(group=2, lp=9, core=4, size=40))[0]
        self.assertEqual((row["group"], row["logical_processor_index"], row["core_index"]), (2, 9, 4))
        self.assertEqual(row["allocation_tag"], 0x1122334455667788)
        self.assertEqual(row["efficiency_class"], 0)

    def test_malformed_or_truncated_records_rejected(self):
        for payload in (b"123", struct.pack("<II", 0, 0), struct.pack("<II", 7, 0),
                        struct.pack("<II", 40, 1), cpu_set(size=31), cpu_set()[:-1]):
            with self.subTest(payload=payload[:8]), self.assertRaises(ValueError):
                topo.parse_windows_cpu_set_buffer(payload)

    def test_scheduling_class_is_byte_not_reserved_union_word(self):
        payload = bytearray(cpu_set())
        struct.pack_into("<I", payload, 20, 0xAABBCC05)
        self.assertEqual(topo.parse_windows_cpu_set_buffer(bytes(payload))[0]["scheduling_class"], 5)

    def test_linux_cpu_zero_keeps_valid_candidate_identity(self):
        rows = [{"logical_processor": 0, "physical_core": {"package_id": 0, "core_id": 0},
                 "allowed_cpuset": True, "efficiency_class": None, "linux_cpu_capacity": None},
                {"logical_processor": 1, "physical_core": {"package_id": 0, "core_id": 1},
                 "allowed_cpuset": True, "efficiency_class": None, "linux_cpu_capacity": None}]
        candidates = topo.candidate_pool_workers(rows, "linux", Path("missing-cpu-sysfs-fixture"))
        self.assertEqual(candidates["pool-affinity-all"]["host_core_candidate"], 0)
        self.assertEqual(candidates["pool-affinity-all"]["worker_candidates"], [1])

    def test_core_identity_includes_processor_group(self):
        rows = topo.parse_windows_cpu_set_buffer(cpu_set(group=0, lp=0, core=0, cpu_id=10, eff=5) +
                                                 cpu_set(group=1, lp=0, core=0, cpu_id=20, eff=5))
        topo.apply_windows_allowed_masks(rows, {0: 1, 1: 1}, {0, 1})
        candidates = topo.candidate_pool_workers(rows, "windows")
        all_candidates = candidates["pool-affinity-all"]["worker_candidates"]
        self.assertEqual(len(all_candidates), 1)
        self.assertNotEqual(all_candidates[0], candidates["pool-affinity-all"]["host_core_candidate"])

    def test_process_allowed_mask_and_cpu_set_allocation_restrict_candidates(self):
        rows = topo.parse_windows_cpu_set_buffer(cpu_set(group=0, lp=0, core=0, cpu_id=1, eff=4) +
                                                 cpu_set(group=0, lp=1, core=1, cpu_id=2, eff=2) +
                                                 cpu_set(group=1, lp=0, core=0, cpu_id=3, eff=4) +
                                                 cpu_set(group=0, lp=2, core=2, cpu_id=4, eff=4, flags=0x02))
        topo.apply_windows_allowed_masks(rows, {0: 0b001}, {0})
        self.assertIs(rows[0]["allowed_cpuset"], True)
        self.assertIs(rows[1]["allowed_cpuset"], False)
        self.assertIsNone(rows[2]["allowed_cpuset"])
        self.assertIs(rows[3]["allowed_cpuset"], False)

    def test_process_default_cpu_set_selection_restricts_allowed_set(self):
        rows = topo.parse_windows_cpu_set_buffer(cpu_set(group=0, lp=0, core=0, cpu_id=100, eff=3) +
                                                 cpu_set(group=0, lp=1, core=1, cpu_id=101, eff=3))
        topo.apply_windows_allowed_masks(rows, {0: 0b11}, {0}, {100}, defaults_known=True)
        self.assertIs(rows[0]["allowed_cpuset"], True)
        self.assertIs(rows[1]["allowed_cpuset"], False)
        self.assertEqual(rows[1]["allowed_status"], "excluded_process_default_cpu_sets")

    def test_efficiency_classes_are_numeric_only_no_cpu_brand_claim(self):
        rows = topo.parse_windows_cpu_set_buffer(cpu_set(lp=0, core=0, cpu_id=1, eff=7) +
                                                 cpu_set(lp=1, core=1, cpu_id=2, eff=2))
        topo.apply_windows_allowed_masks(rows, {0: 0b11}, {0})
        for r in rows:
            r.update({"logical_processor": r["logical_identity"],
                      "physical_core": r["core_identity"], "core_type": "unknown",
                      "classify_confidence": "unknown"})
        candidates = topo.candidate_pool_workers(rows, "windows")
        self.assertEqual([r["efficiency_class"] for r in rows], [7, 2])
        self.assertTrue(all(r["core_type"] == "unknown" for r in rows))
        self.assertEqual(candidates["classification_confidence"], "unknown")
        self.assertIn("P/E/LP labels", candidates["core_type_claims"])

    def test_linux_fixture_respects_sched_affinity_and_package_core_pair(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td) / "cpu"
            root.mkdir()
            (root / "online").write_text("0-3\n", encoding="ascii")
            for cpu, package, core, capacity, node in ((0, 0, 0, 100, 0), (1, 0, 0, 100, 0),
                                                       (2, 1, 0, 60, 1), (3, 1, 0, 60, 1)):
                cpu_root = root / f"cpu{cpu}"
                (cpu_root / "topology").mkdir(parents=True)
                (cpu_root / "topology" / "physical_package_id").write_text(str(package), encoding="ascii")
                (cpu_root / "topology" / "core_id").write_text(str(core), encoding="ascii")
                (cpu_root / "cpu_capacity").write_text(str(capacity), encoding="ascii")
                (cpu_root / f"node{node}").mkdir()
            inventory = topo.linux_inventory(root, allowed_cpus={1, 2, 3})
            rows = inventory["logical_processors"]
            self.assertEqual([r["allowed_cpuset"] for r in rows], [False, True, True, True])
            self.assertEqual(rows[1]["physical_core"], rows[0]["physical_core"])
            self.assertNotEqual(rows[1]["physical_core"], rows[2]["physical_core"])
            self.assertEqual(rows[2]["numa_node"], 1)
            self.assertTrue(all(r["core_type"] == "unknown" for r in rows))

    def test_validate_only_performs_no_observation_or_write(self):
        with tempfile.TemporaryDirectory() as td:
            out = Path(td) / "new" / "cpu.json"
            with mock.patch.object(topo, "collect_inventory", side_effect=AssertionError("hardware observed")):
                self.assertEqual(topo.main(["--output", str(out), "--validate-only"]), 0)
            self.assertFalse(out.parent.exists())

    def test_output_is_exclusive(self):
        with tempfile.TemporaryDirectory() as td:
            out = Path(td) / "cpu.json"
            out.write_text("keep", encoding="utf-8")
            self.assertEqual(topo.main(["--output", str(out), "--validate-only"]), 2)
            self.assertEqual(out.read_text(encoding="utf-8"), "keep")


if __name__ == "__main__":
    unittest.main()
