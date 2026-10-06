import unittest
import contextlib
import io
import tempfile
from pathlib import Path
from unittest.mock import patch

from hetero_inventory import decode_windows_payload, main, map_windows_storage, parse_nvidia_gpus


class InventoryHelpersTests(unittest.TestCase):
    def test_shared_physical_disk_keeps_each_volume_partition_mapping(self):
        disks = [{"number": 2, "model": "NVMe A", "bus_type": "NVMe"}]
        partitions = [
            {"disk_number": 2, "partition_number": 1, "drive_letter": "C", "size_bytes": 1000},
            {"disk_number": 2, "partition_number": 2, "drive_letter": "D", "size_bytes": 2000},
        ]
        volumes = [
            {"drive_letter": "C", "label": "System", "filesystem": "NTFS", "total_bytes": 900, "free_bytes": 100},
            {"drive_letter": "D", "label": "Data", "filesystem": "NTFS", "total_bytes": 1800, "free_bytes": 300},
        ]
        mapped = map_windows_storage(volumes, partitions, disks)
        self.assertEqual([x["physical_disks"][0]["disk_number"] for x in mapped], [2, 2])
        self.assertEqual([x["physical_disks"][0]["partition_number"] for x in mapped], [1, 2])
        self.assertEqual([x["root"] for x in mapped], ["C:\\", "D:\\"])

    def test_unknown_devices_are_preserved_and_section_errors_do_not_erase_data(self):
        decoded = decode_windows_payload('{"data":{"gpus":[{"name":"Unknown Display Adapter","driver_version":null}],"memory":[{"total_bytes":1024,"available_bytes":512}],"storage":[{"disks":[{"number":0,"model":"NVMe","bus_type":"NVMe"}],"partitions":[{"disk_number":0,"partition_number":1,"drive_letter":"C","size_bytes":2048}],"volumes":[{"drive_letter":"C","label":"System","filesystem":"NTFS","total_bytes":2000,"free_bytes":800}]}]},"errors":[{"section":"accelerators","error":"access denied"}]}')
        self.assertEqual(decoded["gpus"][0]["name"], "Unknown Display Adapter")
        self.assertEqual(decoded["memory"]["available_bytes"], 512)
        self.assertEqual(decoded["storage"][0]["physical_disks"][0]["model"], "NVMe")
        self.assertEqual(decoded["errors"][0]["section"], "accelerators")
        self.assertEqual(decoded["errors"][0]["error"], "access denied")

    def test_letterless_volume_joins_by_volume_path(self):
        mapped = map_windows_storage(
            [{"drive_letter": None, "path": "\\\\?\\Volume{unit}\\", "total_bytes": 100, "free_bytes": 50}],
            [{"disk_number": 0, "partition_number": 3, "drive_letter": None,
              "access_paths": ["\\\\?\\Volume{unit}\\"], "size_bytes": 200}],
            [{"number": 0, "model": "System NVMe", "bus_type": "NVMe"}],
        )
        self.assertEqual(mapped[0]["root"], None)
        self.assertEqual(mapped[0]["physical_disks"][0]["partition_number"], 3)

    def test_nvidia_csv_parses_numbers_strictly(self):
        sample = "name, driver_version, memory.total [MiB], memory.used [MiB], memory.free [MiB], utilization.gpu [%]\nRTX 4090 D, 610.88, 49140, 765, 47864, 0\n"
        parsed = parse_nvidia_gpus(sample)
        self.assertEqual(parsed[0]["name"], "RTX 4090 D")
        self.assertEqual(parsed[0]["vram_total_bytes"], 49140 * 1024**2)
        self.assertEqual(parsed[0]["utilization_percent"], 0)
        with self.assertRaises(ValueError):
            parse_nvidia_gpus("RTX, 610.88, 49140, unknown, 2, 0\n")
        with self.assertRaises(ValueError):
            parse_nvidia_gpus("RTX, 610.88, 49140, 2, 3\n")
        with self.assertRaises(ValueError):
            parse_nvidia_gpus("RTX, 610.88, 49140, -2, 3, 0\n")
        with self.assertRaises(ValueError):
            parse_nvidia_gpus("RTX, 610.88, 49140, 2, 3, 101\n")

    def test_powershell_bom_does_not_erase_partial_results(self):
        data = decode_windows_payload('\ufeff{"data":{"cpu":[]},"errors":[{"section":"memory","error":"denied"}]}')
        self.assertEqual(data["cpu"], [])
        self.assertEqual(data["errors"][0]["section"], "memory")

    def test_existing_snapshot_is_preserved_without_collecting(self):
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "hardware.json"
            target.write_text("original evidence", encoding="utf-8")
            with patch("hetero_inventory.collect") as collect_mock, contextlib.redirect_stderr(io.StringIO()):
                self.assertEqual(main(["--output", str(target)]), 1)
            collect_mock.assert_not_called()
            self.assertEqual(target.read_text(encoding="utf-8"), "original evidence")


if __name__ == "__main__":
    unittest.main()
