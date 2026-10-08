import datetime as dt
import io
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

from hetero_resources import (
    RAM_RESERVE_BYTES,
    ResourceSampler,
    main,
    memory_snapshot,
    parse_nvidia_base,
    parse_nvidia_optional,
    run_sampling,
)


class FakeTime:
    def __init__(self):
        self.now = 100.0
        self.utc = dt.datetime(2026, 10, 8, tzinfo=dt.timezone.utc)

    def monotonic(self):
        return self.now

    def sleep(self, seconds):
        self.now += seconds
        self.utc += dt.timedelta(seconds=seconds)

    def utc_now(self):
        return self.utc


class FakeProcess:
    def __init__(self, pid, parent, name, exe, rss, read, write, user, system, children=None):
        self.pid_value, self.parent, self.name_value, self.exe_value = pid, parent, name, exe
        self.rss, self.read, self.write, self.user, self.system = rss, read, write, user, system
        self.child_processes = children or []

    @property
    def pid(self): return self.pid_value
    def create_time(self): return 1000.0
    def ppid(self): return self.parent
    def name(self): return self.name_value
    def exe(self): return self.exe_value
    def memory_info(self): return SimpleNamespace(rss=self.rss)
    def io_counters(self): return SimpleNamespace(read_bytes=self.read, write_bytes=self.write)
    def cpu_times(self): return SimpleNamespace(user=self.user, system=self.system)
    def children(self, recursive=True): return self.child_processes


class FakePsutil:
    def __init__(self):
        child = FakeProcess(202, 101, "worker.exe", r"C:\Strata\engine\worker.exe", 20, 30, 40, 1.5, .5)
        self.root = FakeProcess(101, 4, "strata.exe", r"C:\Strata\engine\strata.exe", 100, 200, 300, 2.5, 1.0, [child])
        self.cpu_calls = 0

    def cpu_percent(self, interval=None):
        self.cpu_calls += 1
        return 0.0 if self.cpu_calls == 1 else 21.5
    def Process(self, pid):
        if pid != 101: raise ProcessLookupError()
        return self.root
    def disk_io_counters(self, perdisk=True):
        return {"PhysicalDrive0": SimpleNamespace(read_bytes=500, write_bytes=600, read_count=7, write_count=8),
                "PhysicalDrive3": SimpleNamespace(read_bytes=900, write_bytes=1000, read_count=9, write_count=10)}


class FakeRunner:
    def __init__(self, base=None, optional=None, optional_rc=0):
        self.base = base if base is not None else "RTX 4090 D, 0, 48000, 0, N/A, N/A\n"
        self.optional = optional if optional is not None else "2100, 10001, 5, 16\n"
        self.optional_rc = optional_rc
        self.calls = []

    def __call__(self, args, **kwargs):
        self.calls.append((args, kwargs))
        is_optional = "clocks.current.graphics" in args[1]
        return SimpleNamespace(returncode=self.optional_rc if is_optional else 0,
                               stdout=self.optional if is_optional else self.base, stderr="")


class ResourceSamplerTests(unittest.TestCase):
    def test_nvidia_zero_is_real_and_na_optional_fields_stay_unknown(self):
        devices, errors = parse_nvidia_base("RTX, 0, 48000, 0, N/A, N/A\n")
        self.assertTrue(devices[0]["telemetry_known"])
        self.assertEqual(devices[0]["memory_used_mib"], 0)
        self.assertEqual(devices[0]["utilization_percent"], 0)
        self.assertEqual(devices[0]["power_watts"], None)
        self.assertEqual(errors, [])

    def test_bad_gpu_csv_field_count_or_required_na_is_unknown(self):
        with self.assertRaises(ValueError):
            parse_nvidia_base("GPU,0,1,2,3\n")
        devices, errors = parse_nvidia_base("GPU, N/A, 100, 0, 0, N/A\n")
        self.assertFalse(devices[0]["telemetry_known"])
        self.assertTrue(errors)

    def test_optional_gpu_csv_n_a_and_bad_row_count(self):
        rows, error = parse_nvidia_optional("N/A, 1000, 3, 8\n", 1)
        self.assertIsNone(error)
        self.assertIsNone(rows[0]["clocks.current.graphics"])
        _, error = parse_nvidia_optional("1,2,3,4\n", 2)
        self.assertIsNotNone(error)

    def test_memory_global_and_psutil_fallback_have_explicit_sources(self):
        global_fields = {"source": "fake GlobalMemoryStatusEx", "physical_total_bytes": 10,
                         "physical_available_bytes": 4, "physical_used_bytes": 6,
                         "commit_limit_bytes": 20, "commit_available_bytes": 8, "commit_used_bytes": 12}
        got, errors = memory_snapshot(lambda: global_fields, None)
        self.assertEqual(got["source"], "fake GlobalMemoryStatusEx")
        self.assertEqual(got["commit_used_bytes"], 12)
        self.assertEqual(errors, [])
        fallback = SimpleNamespace(
            virtual_memory=lambda: SimpleNamespace(total=100, available=60),
            swap_memory=lambda: SimpleNamespace(total=50, used=7))
        got, errors = memory_snapshot(lambda: (_ for _ in ()).throw(OSError("gmem")), fallback)
        self.assertIn("psutil", got["source"])
        self.assertEqual(got["physical_available_bytes"], 60)
        self.assertIsNone(got["commit_limit_bytes"])
        self.assertEqual(got["pagefile_used_bytes"], 7)
        self.assertEqual(errors[0]["section"], "memory_global")

    def test_sample_contains_pid_tree_disk_labels_memory_gate_and_timestamps(self):
        now = FakeTime()
        runner = FakeRunner()
        ps = FakePsutil()
        sampler = ResourceSampler(101, psutil_module=ps, nvidia_runner=runner,
                                  memory_global=lambda: {"source": "GlobalMemoryStatusEx", "physical_total_bytes": 64*1024**3,
                                      "physical_available_bytes": 16*1024**3, "physical_used_bytes": 48*1024**3,
                                      "commit_limit_bytes": 80*1024**3, "commit_available_bytes": 30*1024**3,
                                      "commit_used_bytes": 50*1024**3, "pagefile_total_bytes": None, "pagefile_used_bytes": None},
                                  monotonic=now.monotonic, utc_now=now.utc_now, nvidia_path="nvidia-smi.exe")
        sample = sampler.sample()
        self.assertTrue(sample["observed_at_utc"].endswith("Z"))
        self.assertEqual(sample["monotonic_seconds"], 100.0)
        self.assertEqual(sample["ram_gate"]["status"], "pass")
        self.assertEqual(sample["cpu"]["percent"], 21.5)
        self.assertEqual([p["pid"] for p in sample["owner_process_tree"]], [101, 202])
        self.assertNotIn("command_line", sample["owner_process_tree"][0])
        self.assertEqual(sample["disk_io"]["PhysicalDrive3"]["volume_mapping"], "not_inferred")
        self.assertEqual(sample["nvidia"]["devices"][0]["memory_used_mib"], 0)
        now.sleep(1.25)
        second = sampler.sample()
        self.assertAlmostEqual(second["sample_gap_seconds"], 1.25)
        self.assertEqual(second["nvidia"]["telemetry_known"], True)

    def test_low_ram_and_failed_nvidia_have_explicit_alerts_not_zero_gpu(self):
        now = FakeTime()
        runner = FakeRunner(base="GPU, N/A, N/A, N/A, N/A, N/A\n")
        sampler = ResourceSampler(None, psutil_module=FakePsutil(), nvidia_runner=runner,
                                  memory_global=lambda: {"source": "test", "physical_total_bytes": 20,
                                      "physical_available_bytes": 10, "physical_used_bytes": 10,
                                      "commit_limit_bytes": None, "commit_available_bytes": None, "commit_used_bytes": None,
                                      "pagefile_total_bytes": None, "pagefile_used_bytes": None},
                                  monotonic=now.monotonic, utc_now=now.utc_now, nvidia_path="smi")
        sample = sampler.sample()
        self.assertEqual(sample["ram_gate"]["status"], "block")
        self.assertIn("available_ram_below_12_gib", sample["alerts"])
        self.assertIn("nvidia_telemetry_unknown", sample["alerts"])
        self.assertFalse(sample["nvidia"]["telemetry_known"])

    def test_stop_file_interrupts_sampler_and_keeps_utc_monotonic_records(self):
        now = FakeTime()

        class TinySampler:
            calls = 0
            def sample(self):
                self.calls += 1
                return {"record_type": "resource_sample", "observed_at_utc": now.utc_now().isoformat(),
                        "monotonic_seconds": now.monotonic(), "value": self.calls}

        sampler = TinySampler()
        check_count = {"n": 0}
        def stop_exists(_path):
            check_count["n"] += 1
            return check_count["n"] > 1
        stream = io.StringIO()
        reason = run_sampling(stream, sampler, interval_seconds=1, max_seconds=5, stop_file="stop.flag",
                              monotonic=now.monotonic, sleep=now.sleep, stop_exists=stop_exists)
        records = [json.loads(line) for line in stream.getvalue().splitlines()]
        self.assertEqual(reason, "stop_file")
        self.assertEqual(sampler.calls, 1)
        self.assertEqual(records[0]["record_type"], "resource_sample")
        self.assertEqual(records[1]["reason"], "stop_file_exists")
        self.assertEqual(records[0]["monotonic_seconds"], 100.0)

    def test_validate_only_and_output_guard_do_not_construct_sampler_or_overwrite(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td) / "resource_test_newdir"
            root.mkdir()
            output = root / "new.jsonl"
            factory = Mock(side_effect=AssertionError("validate-only must not query resources"))
            self.assertEqual(main(["--output", str(output), "--validate-only"], sampler_factory=factory), 0)
            factory.assert_not_called()
            self.assertFalse(output.exists())
            protected = root / "keep.jsonl"
            protected.write_text("keep\n", encoding="utf-8")
            self.assertEqual(main(["--output", str(protected)], sampler_factory=factory), 2)
            factory.assert_not_called()
            self.assertEqual(protected.read_text(encoding="utf-8"), "keep\n")


if __name__ == "__main__":
    unittest.main()
