"""Pure local SVE1 parser and no-spawn client validation tests."""
from __future__ import annotations

import contextlib
import hashlib
import io
import json
import queue
import struct
import tempfile
import unittest
from pathlib import Path
from unittest import mock
from unittest.mock import patch
import importlib.util

SOURCE = Path(__file__).with_name("oracle_client.py")
SPEC = importlib.util.spec_from_file_location("vision_oracle_client_fixture", SOURCE)
oracle_client = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(oracle_client)


class FakeStdout:
    def __init__(self): self.lines = queue.Queue()
    def feed(self, value): self.lines.put(value)
    def readline(self, _limit): return self.lines.get(timeout=3)


class FakeStdin:
    def __init__(self, owner): self.owner = owner
    def write(self, data):
        line = data.decode().strip()
        if line == "QUIT":
            self.owner.returncode = self.owner.quit_code
            self.owner.stdout.feed(b"")
        else:
            _, _image, out = line.split()
            payload = oracle_client.HEADER.pack(0x31455653, 1, 1, 1, oracle_client.HIDDEN)
            Path(out).write_bytes(payload + bytes(oracle_client.HIDDEN * 4))
            self.owner.stdout.feed(b"OK 1 1 1 0\n")
        return len(data)
    def flush(self): pass


class FakeProcess:
    def __init__(self, quit_code):
        self.pid, self._handle, self.quit_code, self.returncode = 4321, 123, quit_code, None
        self.stdout = FakeStdout(); self.stdout.feed(b"READY 2560\n"); self.stdin = FakeStdin(self)
    def poll(self): return self.returncode
    def wait(self, timeout=None): return self.returncode
    def terminate(self): self.returncode = -15
    def kill(self): self.returncode = -9


class LongLineStream:
    def __init__(self): self.called = False
    def readline(self, limit):
        if self.called: return b""
        self.called = True
        return b"x" * limit


class OracleClientTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="sve1-fixture-")
        self.path = Path(self.temp.name) / "tiny.sve"

    def tearDown(self):
        self.temp.cleanup()

    def write_sve(self, body=None):
        body = body if body is not None else bytes(oracle_client.HIDDEN * 4)
        self.path.write_bytes(oracle_client.HEADER.pack(0x31455653, 1, 1, 1, oracle_client.HIDDEN) + body)

    def test_sve1_fixture_parses_header_exact_body_hash_and_time(self):
        self.write_sve()
        result = oracle_client.validate_sve(self.path, "OK 1 1 1 0")
        self.assertEqual((result["n_tokens"], result["nx"], result["ny"], result["n_embd"]), (1, 1, 1, 2560))
        self.assertEqual(result["size_bytes"], 20 + 2560 * 4)
        self.assertEqual(len(result["sha256"]), 64)

    def test_sve1_rejects_malformed_response_header_and_nonfinite(self):
        self.write_sve()
        with self.assertRaises(ValueError): oracle_client.validate_sve(self.path, "OK 2 1 1 0")
        self.path.write_bytes(oracle_client.HEADER.pack(0x31455653, 1, 1, 1, oracle_client.HIDDEN) + b"short")
        with self.assertRaisesRegex(ValueError, "body size"):
            oracle_client.validate_sve(self.path, "OK 1 1 1 0")
        bad = oracle_client.HEADER.pack(0x31455653, 1, 1, 1, 1280) + bytes(oracle_client.HIDDEN * 4)
        self.path.write_bytes(bad)
        with self.assertRaisesRegex(ValueError, "header"):
            oracle_client.validate_sve(self.path, "OK 1 1 1 0")
        body = struct.pack("<f", float("nan")) + bytes((oracle_client.HIDDEN - 1) * 4)
        self.write_sve(body)
        with self.assertRaisesRegex(ValueError, "NaN"):
            oracle_client.validate_sve(self.path, "OK 1 1 1 0")

    def test_default_client_validation_never_spawns_helper(self):
        stdout = io.StringIO()
        with mock.patch.object(oracle_client.subprocess, "Popen", side_effect=AssertionError("spawn forbidden")), \
             contextlib.redirect_stdout(stdout):
            rc = oracle_client.main([])
        self.assertEqual(rc, 0)
        self.assertIn('"helper_spawned":false', stdout.getvalue())

    def test_stdout_line_has_strict_16kib_cap(self):
        q = queue.Queue(maxsize=8)
        oracle_client.reader(LongLineStream(), q)
        with self.assertRaisesRegex(RuntimeError, "16 KiB"):
            oracle_client.read_line(q, 1)

    def test_nonzero_quit_writes_exclusive_failure_receipt_and_progress(self):
        root = Path(self.temp.name); mm = root / "mm.gguf"; model = root / "model.gguf"
        mm.write_bytes(b"mmproj fixture"); model.write_bytes(b"model header fixture")
        mm_sha = hashlib.sha256(mm.read_bytes()).hexdigest(); model_sha = hashlib.sha256(model.read_bytes()).hexdigest()
        runs = root / "runs"; child = FakeProcess(9)
        memory = {"source": "fake GlobalMemoryStatusEx+PSAPI", "physical_available_bytes": 100 * 1024**3,
                  "commit_available_bytes": 100 * 1024**3}
        with patch.object(oracle_client, "validate_only", return_value={}), \
             patch.object(oracle_client, "MM", mm), patch.object(oracle_client, "MM_SHA", mm_sha), \
             patch.object(oracle_client, "MODEL", model), patch.object(oracle_client, "MODEL_SHA", model_sha), \
             patch.object(oracle_client, "MODEL_SIZE", len(model.read_bytes())), patch.object(oracle_client, "RUNS", runs), \
             patch.object(oracle_client, "memory_snapshot", return_value=(memory, [])) as resource, \
             patch.object(oracle_client, "process_created_utc", return_value="2026-10-09T00:00:00+00:00"), \
             patch.object(oracle_client, "idle"), patch.object(oracle_client.subprocess, "Popen", return_value=child):
            with self.assertRaisesRegex(RuntimeError, "exited 9"):
                oracle_client.run()
        run_dir = next(runs.iterdir())
        failure = json.loads((run_dir / "failed-oracle-receipt.json").read_text(encoding="utf-8"))
        progress = [json.loads(line) for line in (run_dir / "progress.jsonl").read_text(encoding="utf-8").splitlines()]
        self.assertEqual(failure["phase"], "quit")
        self.assertEqual((failure["pid"], failure["process_create_utc"], len(failure["partial_results"])),
                         (4321, "2026-10-09T00:00:00+00:00", 16))
        self.assertEqual(failure["exit_code"], 9)
        self.assertEqual(resource.call_count, 34)  # both startup gates and before/after every ENC
        self.assertEqual(sum(event["event"] == "encoder_response" for event in progress), 16)
        self.assertEqual(sum(event["event"] == "resource_gate" for event in progress), 34)
        self.assertFalse((run_dir / "oracle-receipt.json").exists())
