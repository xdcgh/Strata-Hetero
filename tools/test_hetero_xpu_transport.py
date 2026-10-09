"""Pure stdlib fixtures for bounded SXPU framing and explicitly owned fake workers."""
from __future__ import annotations

import contextlib
import io
import json
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest import mock

from tools import hetero_xpu_transport as transport

ECHO_CODE = "\n".join([
    "import struct, sys, time",
    "h = struct.Struct('<4sHHQI')",
    "def exact(n):",
    "    out = bytearray()",
    "    while len(out) < n:",
    "        part = sys.stdin.buffer.read(n - len(out))",
    "        if not part: raise SystemExit(0)",
    "        out.extend(part)",
    "    return bytes(out)",
    "while True:",
    "    first = sys.stdin.buffer.read(h.size)",
    "    if not first: break",
    "    raw = first + exact(h.size - len(first))",
    "    magic, version, kind, rid, n = h.unpack(raw)",
    "    payload = exact(n)",
    "    time.sleep(0.06)",
    "    sys.stdout.buffer.write(h.pack(magic, version, kind | 0x8000, rid, n))",
    "    sys.stdout.buffer.write(payload)",
    "    sys.stdout.buffer.flush()",
])


class ShortReader:
    def __init__(self, data: bytes, cap: int = 3):
        self.data = data
        self.cap = cap
        self.bytes_read = 0

    def read(self, count: int) -> bytes:
        n = min(count, self.cap, len(self.data))
        chunk, self.data = self.data[:n], self.data[n:]
        self.bytes_read += n
        return chunk


class ShortWriter:
    def __init__(self, cap: int = 2):
        self.cap = cap
        self.data = bytearray()
        self.flushes = 0

    def write(self, view) -> int:
        n = min(len(view), self.cap)
        self.data.extend(view[:n])
        return n

    def flush(self) -> None:
        self.flushes += 1


class FakePopen:
    def __init__(self):
        self.pid = 900001
        self.stdin = None
        self.stdout = None
        self.returncode = None
        self.terminated = False

    def poll(self):
        return self.returncode

    def terminate(self):
        self.terminated = True
        self.returncode = -15

    def kill(self):
        self.terminated = True
        self.returncode = -9

    def wait(self, timeout=None):
        if self.returncode is None:
            raise subprocess.TimeoutExpired("fake", timeout)
        return self.returncode


class HeteroXpuTransportTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="hetero-xpu-transport-")
        self.root = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def make_worker(self, name: str, code: str = ECHO_CODE, **kwargs):
        return transport.OwnedPipeWorker([sys.executable, "-u", "-c", code],
                                         self.root / f"{name}.stderr", **kwargs)

    def test_header_is_20_bytes_and_short_read_write_are_completed(self):
        self.assertEqual(transport.HEADER.size, 20)
        data = bytes(range(32))
        reader = ShortReader(data, cap=3)
        self.assertEqual(transport.read_exact(reader, len(data)), data)
        self.assertEqual(reader.bytes_read, len(data))
        writer = ShortWriter(cap=2)
        transport.write_all(writer, data)
        self.assertEqual(bytes(writer.data), data)
        self.assertEqual(writer.flushes, 1)

    def test_eof_and_invalid_lengths_are_rejected(self):
        with self.assertRaisesRegex(transport.ProtocolError, "EOF"):
            transport.read_exact(ShortReader(b"abc", cap=2), 4)
        with self.assertRaisesRegex(transport.ProtocolError, "read length"):
            transport.read_exact(ShortReader(b""), transport.MAX_FRAME_BYTES + 1)
        with self.assertRaisesRegex(transport.ProtocolError, "no progress"):
            transport.write_all(ShortWriter(cap=0), b"x")

    def test_oversize_header_is_rejected_before_payload_read(self):
        header = transport.HEADER.pack(transport.MAGIC, transport.VERSION, 0x8001, 42,
                                       transport.MAX_PAYLOAD_BYTES + 1)
        reader = ShortReader(header, cap=transport.HEADER_BYTES)
        with self.assertRaisesRegex(transport.ProtocolError, "exceeds"):
            transport.read_response(reader, 1, 42)
        self.assertEqual(reader.bytes_read, transport.HEADER_BYTES)

    def test_magic_version_kind_and_request_id_mismatches_rejected_from_header(self):
        cases = [
            (b"NOPE", 1, 0x8001, 9, "magic"),
            (transport.MAGIC, 2, 0x8001, 9, "version"),
            (transport.MAGIC, 1, 0x8002, 9, "kind"),
            (transport.MAGIC, 1, 0x8001, 10, "request_id"),
        ]
        for magic, version, kind, rid, message in cases:
            with self.subTest(message=message):
                header = transport.HEADER.pack(magic, version, kind, rid, 4)
                reader = ShortReader(header + b"data", cap=transport.HEADER_BYTES)
                with self.assertRaisesRegex(transport.ProtocolError, message):
                    transport.read_response(reader, 1, 9)
                self.assertEqual(reader.bytes_read, transport.HEADER_BYTES)

    def test_default_cli_and_import_only_validate_schema(self):
        stdout = io.StringIO()
        with mock.patch.object(transport.subprocess, "Popen", side_effect=AssertionError("spawn forbidden")):
            with contextlib.redirect_stdout(stdout):
                self.assertEqual(transport.main([]), 0)
        result = json.loads(stdout.getvalue())
        self.assertEqual(result["status"], "schema_valid")
        self.assertFalse(result["worker_started"])
        self.assertNotIn("openvino", sys.modules)
        self.assertNotIn("numpy", sys.modules)

    def test_json_cli_validation_never_spawns(self):
        request = {"schema": transport.REQUEST_SCHEMA, "version": 1, "kind": 2,
                   "request_id": 17, "payload_len": 4}
        path = self.root / "request.json"
        path.write_text(json.dumps(request), encoding="utf-8")
        stdout = io.StringIO()
        with mock.patch.object(transport.subprocess, "Popen", side_effect=AssertionError("spawn forbidden")):
            with contextlib.redirect_stdout(stdout):
                self.assertEqual(transport.main(["--validate-json", str(path)]), 0)
        self.assertEqual(json.loads(stdout.getvalue())["request"]["request_id"], 17)
        request["payload_len"] = transport.MAX_PAYLOAD_BYTES + 1
        with self.assertRaises(transport.ProtocolError):
            transport.validate_request_schema(request)

    def test_constructing_worker_does_not_spawn_or_create_stderr(self):
        path = self.root / "lazy.stderr"
        with mock.patch.object(transport.subprocess, "Popen", side_effect=AssertionError("spawn forbidden")) as popen:
            worker = self.make_worker("lazy")
            self.assertIsNone(worker.pid)
            self.assertFalse(worker.in_flight)
            worker.close()
            popen.assert_not_called()
        self.assertFalse(path.exists())

    def test_start_and_close_are_serialized_until_owned_popen_is_published(self):
        external = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"], shell=False,
                                    stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                                    stderr=subprocess.DEVNULL)
        worker = self.make_worker("start-close-race", shutdown_timeout=0.5)
        create_entered = threading.Event()
        allow_create_to_return = threading.Event()
        close_entered = threading.Event()
        fake_proc = FakePopen()
        start_errors, close_errors = [], []

        def delayed_popen(*_args, **_kwargs):
            create_entered.set()
            if not allow_create_to_return.wait(timeout=3):
                raise TimeoutError("fixture did not release delayed Popen")
            return fake_proc

        def start_worker():
            try:
                worker.start()
            except BaseException as exc:
                start_errors.append(exc)

        def close_worker():
            close_entered.set()
            try:
                worker.close()
            except BaseException as exc:
                close_errors.append(exc)

        start_thread = threading.Thread(target=start_worker)
        close_thread = threading.Thread(target=close_worker)
        try:
            with mock.patch.object(transport.subprocess, "Popen", side_effect=delayed_popen):
                start_thread.start()
                self.assertTrue(create_entered.wait(timeout=2))
                close_thread.start()
                self.assertTrue(close_entered.wait(timeout=2))
                time.sleep(0.05)
                self.assertTrue(close_thread.is_alive(), "close must wait for start to publish its child")
                self.assertFalse(fake_proc.terminated)
                allow_create_to_return.set()
                start_thread.join(timeout=2)
                close_thread.join(timeout=2)
                self.assertFalse(start_thread.is_alive())
                self.assertFalse(close_thread.is_alive())
            self.assertEqual(start_errors, [])
            self.assertEqual(close_errors, [])
            self.assertTrue(fake_proc.terminated)
            self.assertEqual(worker.returncode, -15)
            self.assertIsNone(external.poll(), "close must not affect the unrelated Popen object")
        finally:
            allow_create_to_return.set()
            if start_thread.is_alive():
                start_thread.join(timeout=2)
            if close_thread.is_alive():
                close_thread.join(timeout=2)
            worker.close()
            if external.poll() is None:
                external.terminate()
                external.wait(timeout=2)

    def test_echo_roundtrip_is_sequential_and_checks_response_bit(self):
        worker = self.make_worker("echo")
        worker.start()
        try:
            for rid, payload in ((21, b"one"), (22, bytes(range(128)))):
                response = worker.request(5, rid, payload)
                self.assertEqual(response.kind, 5 | transport.RESPONSE_BIT)
                self.assertEqual(response.request_id, rid)
                self.assertEqual(response.payload, payload)
                self.assertEqual(response.version, transport.VERSION)
            self.assertFalse(worker.unusable)
        finally:
            worker.close()
        self.assertIsNotNone(worker.returncode)

    def test_bad_response_kills_only_owned_child_and_marks_worker_unusable(self):
        code = "\n".join([
            "import struct, sys, time",
            "h=struct.Struct('<4sHHQI')",
            "raw=sys.stdin.buffer.read(h.size)",
            "magic, version, kind, rid, n=h.unpack(raw)",
            "payload=sys.stdin.buffer.read(n)",
            "sys.stdout.buffer.write(h.pack(magic, version, kind|0x8000, rid+1, 0))",
            "sys.stdout.buffer.flush()",
            "time.sleep(20)",
        ])
        worker = self.make_worker("bad-response", code)
        worker.start()
        try:
            with self.assertRaisesRegex(transport.ProtocolError, "request_id"):
                worker.request(1, 30, b"x")
            self.assertTrue(worker.unusable)
            self.assertIsNotNone(worker.returncode)
            with self.assertRaises(transport.WorkerStateError):
                worker.request(1, 31, b"")
        finally:
            worker.close()

    def test_exit_before_response_is_reaped_and_not_reused(self):
        code = "import sys; sys.stdin.buffer.read(20); sys.exit(7)"
        worker = self.make_worker("early-exit", code)
        worker.start()
        try:
            with self.assertRaises(transport.ProtocolError):
                worker.request(1, 70, b"x")
            self.assertEqual(worker.returncode, 7)
            self.assertTrue(worker.unusable)
        finally:
            worker.close()

    def test_timeout_includes_blocked_stdin_write_and_only_kills_owned_popen(self):
        worker = self.make_worker("blocked-write", "import time; time.sleep(30)",
                                  request_timeout=0.15, shutdown_timeout=1.0)
        unrelated = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"], shell=False,
                                     stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                                     stderr=subprocess.DEVNULL)
        try:
            worker.start()
            with self.assertRaisesRegex(transport.WorkerTimeout, "reaped"):
                worker.request(1, 81, b"x" * transport.MAX_PAYLOAD_BYTES, timeout=0.15)
            self.assertTrue(worker.unusable)
            self.assertIsNotNone(worker.returncode)
            self.assertIsNone(unrelated.poll())
        finally:
            worker.close()
            if unrelated.poll() is None:
                unrelated.terminate()
                unrelated.wait(timeout=2)

    def test_stderr_path_is_exclusive_and_existing_file_is_preserved(self):
        path = self.root / "owned.stderr"
        path.write_bytes(b"preserve")
        worker = transport.OwnedPipeWorker([sys.executable, "-c", "pass"], path)
        with mock.patch.object(transport.subprocess, "Popen") as popen:
            with self.assertRaises(FileExistsError):
                worker.start()
            popen.assert_not_called()
        self.assertEqual(path.read_bytes(), b"preserve")

    def test_concurrent_request_is_rejected_while_first_finishes(self):
        worker = self.make_worker("single-flight")
        worker.start()
        results, errors = [], []

        def first():
            try:
                results.append(worker.request(4, 101, b"first", timeout=2))
            except BaseException as exc:
                errors.append(exc)

        thread = threading.Thread(target=first)
        thread.start()
        try:
            deadline = time.monotonic() + 1
            while not worker.in_flight and time.monotonic() < deadline:
                time.sleep(0.002)
            self.assertTrue(worker.in_flight)
            with self.assertRaises(transport.WorkerBusy):
                worker.request(4, 102, b"nested", timeout=1)
            thread.join(timeout=3)
            self.assertFalse(thread.is_alive())
            self.assertEqual(errors, [])
            self.assertEqual(results[0].payload, b"first")
            self.assertFalse(worker.unusable)
        finally:
            worker.close()


if __name__ == "__main__":
    unittest.main()
