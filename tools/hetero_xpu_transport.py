#!/usr/bin/env python3
"""Bounded stdio framing and explicitly started owned-pipe workers.

Importing this module and running its default CLI path only validate protocol metadata. No
hardware/runtime libraries are imported and no subprocess is started unless a caller invokes
OwnedPipeWorker.start().
"""
from __future__ import annotations

import argparse
import json
import math
import os
import queue
import struct
import subprocess
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, BinaryIO, Sequence

MAGIC = b"SXPU"
VERSION = 1
HEADER = struct.Struct("<4sHHQI")
HEADER_BYTES = 20
MAX_PAYLOAD_BYTES = 8 * 1024 * 1024
MAX_FRAME_BYTES = HEADER_BYTES + MAX_PAYLOAD_BYTES
RESPONSE_BIT = 0x8000
REQUEST_SCHEMA = "strata-hetero-xpu-request-v1"
MAX_REQUEST_JSON_BYTES = 64 * 1024


class TransportError(RuntimeError):
    """Base error for protocol or owned-worker failures."""


class ProtocolError(TransportError):
    """The byte stream is truncated, malformed, oversized, or mismatched."""


class FrameTooLarge(ProtocolError):
    """A validated header advertised a payload that exceeds the receiver cap."""

    def __init__(self, kind: int, request_id: int, payload_len: int, limit: int):
        super().__init__(f"frame payload {payload_len} exceeds {limit} byte limit")
        self.kind = kind
        self.request_id = request_id
        self.payload_len = payload_len
        self.limit = limit


class WorkerStateError(TransportError):
    """The worker is not started, is unusable, or is already closed."""


class WorkerBusy(WorkerStateError):
    """A second request was attempted while one is in flight."""


class WorkerTimeout(TransportError):
    """A request timed out, including time blocked while writing stdin."""


class WorkerCleanupError(TransportError):
    """The owned child could not be confirmed reaped after termination."""


@dataclass(frozen=True)
class Frame:
    kind: int
    request_id: int
    payload: bytes
    version: int = VERSION
    magic: bytes = MAGIC


def _uint(value: Any, maximum: int, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value <= maximum:
        raise ProtocolError(f"{label} must be an integer in 0..{maximum}")
    return value


def validate_request_schema(value: Any) -> dict[str, int | str]:
    """Validate a small JSON request descriptor without opening a worker or runtime."""
    if not isinstance(value, dict):
        raise ProtocolError("request descriptor must be a JSON object")
    expected = {"schema", "version", "kind", "request_id", "payload_len"}
    if set(value) != expected:
        raise ProtocolError(f"request descriptor keys must be exactly {sorted(expected)}")
    if value["schema"] != REQUEST_SCHEMA:
        raise ProtocolError(f"unsupported request schema: {value['schema']!r}")
    version = _uint(value["version"], 0xFFFF, "version")
    if version != VERSION:
        raise ProtocolError(f"unsupported protocol version: {version}")
    kind = _uint(value["kind"], 0x7FFF, "request kind")
    request_id = _uint(value["request_id"], 0xFFFFFFFFFFFFFFFF, "request_id")
    payload_len = _uint(value["payload_len"], MAX_PAYLOAD_BYTES, "payload_len")
    return {"schema": REQUEST_SCHEMA, "version": version, "kind": kind,
            "request_id": request_id, "payload_len": payload_len}


def encode_frame(kind: int, request_id: int, payload: bytes | bytearray | memoryview,
                 *, version: int = VERSION) -> bytes:
    kind = _uint(kind, 0x7FFF, "request kind")
    request_id = _uint(request_id, 0xFFFFFFFFFFFFFFFF, "request_id")
    version = _uint(version, 0xFFFF, "version")
    if version != VERSION:
        raise ProtocolError(f"unsupported protocol version: {version}")
    try:
        view = memoryview(payload).cast("B")
    except (TypeError, ValueError) as exc:
        raise ProtocolError("payload must be a contiguous bytes-like object") from exc
    payload_len = view.nbytes
    if payload_len > MAX_PAYLOAD_BYTES:
        raise ProtocolError(f"payload exceeds {MAX_PAYLOAD_BYTES} byte limit")
    header = HEADER.pack(MAGIC, version, kind, request_id, payload_len)
    return header + view.tobytes()


def encode_response_frame(request_kind: int, request_id: int, payload: bytes | bytearray | memoryview,
                          *, version: int = VERSION) -> bytes:
    """Encode the response for a request kind, setting its high response bit."""
    request_kind = _uint(request_kind, 0x7FFF, "request kind")
    request_id = _uint(request_id, 0xFFFFFFFFFFFFFFFF, "request_id")
    version = _uint(version, 0xFFFF, "version")
    if version != VERSION:
        raise ProtocolError(f"unsupported protocol version: {version}")
    try:
        view = memoryview(payload).cast("B")
    except (TypeError, ValueError) as exc:
        raise ProtocolError("payload must be a contiguous bytes-like object") from exc
    payload_len = view.nbytes
    if payload_len > MAX_PAYLOAD_BYTES:
        raise ProtocolError(f"response payload exceeds {MAX_PAYLOAD_BYTES} byte limit")
    return HEADER.pack(MAGIC, version, request_kind | RESPONSE_BIT, request_id, payload_len) + view.tobytes()


def read_exact(stream: BinaryIO, length: int) -> bytes:
    """Read exactly length bytes, tolerating short reads but bounding every allocation."""
    if isinstance(length, bool) or not isinstance(length, int) or not 0 <= length <= MAX_FRAME_BYTES:
        raise ProtocolError(f"read length must be in 0..{MAX_FRAME_BYTES}")
    result = bytearray()
    while len(result) < length:
        remaining = length - len(result)
        chunk = stream.read(remaining)
        if chunk is None:
            raise ProtocolError("stream returned None during blocking read")
        if not chunk:
            raise ProtocolError(f"unexpected EOF: needed {length} bytes, received {len(result)}")
        if len(chunk) > remaining:
            raise ProtocolError("stream returned more bytes than requested")
        result.extend(chunk)
    return bytes(result)


def write_all(stream: BinaryIO, data: bytes | bytearray | memoryview) -> None:
    """Write a bounded frame completely, handling streams that accept short writes."""
    try:
        view = memoryview(data).cast("B")
    except (TypeError, ValueError) as exc:
        raise ProtocolError("write data must be a contiguous bytes-like object") from exc
    if view.nbytes > MAX_FRAME_BYTES:
        raise ProtocolError(f"frame exceeds {MAX_FRAME_BYTES} byte limit")
    offset = 0
    while offset < view.nbytes:
        written = stream.write(view[offset:])
        if written is None:
            raise ProtocolError("stream returned None during blocking write")
        if isinstance(written, bool) or not isinstance(written, int) or written <= 0:
            raise ProtocolError("stream made no progress while writing")
        if written > view.nbytes - offset:
            raise ProtocolError("stream reported writing more bytes than supplied")
        offset += written
    flush = getattr(stream, "flush", None)
    if callable(flush):
        flush()


def read_frame(stream: BinaryIO, *, max_payload_bytes: int = MAX_PAYLOAD_BYTES) -> Frame | None:
    """Read one generic frame; None denotes clean EOF before any header byte."""
    max_payload_bytes = _uint(max_payload_bytes, MAX_PAYLOAD_BYTES, "max_payload_bytes")
    first = stream.read(HEADER_BYTES)
    if first is None:
        raise ProtocolError("stream returned None while reading frame header")
    if not first:
        return None
    if len(first) > HEADER_BYTES:
        raise ProtocolError("stream returned more bytes than requested for frame header")
    raw_header = first if len(first) == HEADER_BYTES else first + read_exact(stream, HEADER_BYTES - len(first))
    magic, version, kind, request_id, payload_len = HEADER.unpack(raw_header)
    if magic != MAGIC:
        raise ProtocolError(f"bad frame magic: {magic!r}")
    if version != VERSION:
        raise ProtocolError(f"unsupported protocol version: {version}")
    if payload_len > max_payload_bytes:
        raise FrameTooLarge(kind, request_id, payload_len, max_payload_bytes)
    payload = read_exact(stream, payload_len)
    return Frame(kind=kind, request_id=request_id, payload=payload, version=version, magic=magic)


def read_response(stream: BinaryIO, request_kind: int, request_id: int,
                  *, version: int = VERSION,
                  max_payload_bytes: int = MAX_PAYLOAD_BYTES) -> Frame:
    """Read a response, validating its full header before allocating its payload."""
    request_kind = _uint(request_kind, 0x7FFF, "request kind")
    request_id = _uint(request_id, 0xFFFFFFFFFFFFFFFF, "request_id")
    version = _uint(version, 0xFFFF, "version")
    if version != VERSION:
        raise ProtocolError(f"unsupported expected version: {version}")
    max_payload_bytes = _uint(max_payload_bytes, MAX_PAYLOAD_BYTES, "max_payload_bytes")
    raw_header = read_exact(stream, HEADER_BYTES)
    magic, got_version, got_kind, got_id, payload_len = HEADER.unpack(raw_header)
    if magic != MAGIC:
        raise ProtocolError(f"bad frame magic: {magic!r}")
    if got_version != version:
        raise ProtocolError(f"frame version mismatch: expected {version}, got {got_version}")
    expected_kind = request_kind | RESPONSE_BIT
    if got_kind != expected_kind:
        raise ProtocolError(f"response kind mismatch: expected {expected_kind}, got {got_kind}")
    if got_id != request_id:
        raise ProtocolError(f"request_id mismatch: expected {request_id}, got {got_id}")
    # The size cap is checked before read_exact allocates or reads any payload bytes.
    if payload_len > max_payload_bytes:
        raise ProtocolError(f"response payload exceeds {max_payload_bytes} byte limit")
    payload = read_exact(stream, payload_len)
    return Frame(kind=got_kind, request_id=got_id, payload=payload,
                 version=got_version, magic=magic)


class OwnedPipeWorker:
    """A single-flight stdio worker owned by this exact subprocess.Popen object."""

    def __init__(self, argv: Sequence[str], stderr_path: str | os.PathLike[str],
                 *, request_timeout: float = 30.0, shutdown_timeout: float = 1.0,
                 max_payload_bytes: int = MAX_PAYLOAD_BYTES) -> None:
        if isinstance(argv, (str, bytes)) or not isinstance(argv, Sequence) or not argv:
            raise ValueError("argv must be a non-empty sequence of command arguments")
        if any(not isinstance(part, str) or not part for part in argv):
            raise ValueError("every argv item must be a non-empty string")
        if not math.isfinite(request_timeout) or request_timeout <= 0:
            raise ValueError("request_timeout must be finite and positive")
        if not math.isfinite(shutdown_timeout) or shutdown_timeout <= 0:
            raise ValueError("shutdown_timeout must be finite and positive")
        if isinstance(max_payload_bytes, bool) or not isinstance(max_payload_bytes, int) or not 0 <= max_payload_bytes <= MAX_PAYLOAD_BYTES:
            raise ValueError(f"max_payload_bytes must be in 0..{MAX_PAYLOAD_BYTES}")
        self.argv = tuple(argv)
        self.stderr_path = Path(stderr_path)
        self.request_timeout = float(request_timeout)
        self.shutdown_timeout = float(shutdown_timeout)
        self.max_payload_bytes = max_payload_bytes
        self._proc: subprocess.Popen[bytes] | None = None
        self._stderr_file: BinaryIO | None = None
        self._state_lock = threading.Lock()
        self._cleanup_lock = threading.Lock()
        self._started = False
        self._closed = False
        self._unusable = False
        self._in_flight = False

    @property
    def pid(self) -> int | None:
        return self._proc.pid if self._proc is not None else None

    @property
    def in_flight(self) -> bool:
        with self._state_lock:
            return self._in_flight

    @property
    def unusable(self) -> bool:
        with self._state_lock:
            return self._unusable

    @property
    def returncode(self) -> int | None:
        return self._proc.poll() if self._proc is not None else None

    def start(self) -> "OwnedPipeWorker":
        """Explicitly spawn the worker; stderr is an exclusive caller-supplied file."""
        with self._state_lock:
            if self._started or self._closed:
                raise WorkerStateError("worker can be started exactly once")
            self._started = True
            stderr_file: BinaryIO | None = None
            try:
                stderr_file = self.stderr_path.open("xb", buffering=0)
                creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0) if os.name == "nt" else 0
                proc = subprocess.Popen(
                    list(self.argv), shell=False, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                    stderr=stderr_file, bufsize=0, close_fds=True, creationflags=creationflags,
                )
            except BaseException:
                if stderr_file is not None:
                    stderr_file.close()
                self._unusable = True
                raise
            # Publish both owned handles before releasing the same lock that close() uses.
            self._stderr_file = stderr_file
            self._proc = proc
        return self

    def request(self, kind: int, request_id: int, payload: bytes | bytearray | memoryview,
                *, timeout: float | None = None) -> Frame:
        # Validate caller data before taking a worker slot or starting any I/O.
        kind = _uint(kind, 0x7FFF, "request kind")
        request_id = _uint(request_id, 0xFFFFFFFFFFFFFFFF, "request_id")
        try:
            payload_view = memoryview(payload).cast("B")
        except (TypeError, ValueError) as exc:
            raise ProtocolError("payload must be a contiguous bytes-like object") from exc
        if payload_view.nbytes > self.max_payload_bytes:
            raise ProtocolError(f"payload exceeds worker limit {self.max_payload_bytes}")
        frame_bytes = encode_frame(kind, request_id, payload_view)
        if len(frame_bytes) > HEADER_BYTES + self.max_payload_bytes:
            raise ProtocolError("encoded frame exceeds worker limit")
        wait_seconds = self.request_timeout if timeout is None else timeout
        if not math.isfinite(wait_seconds) or wait_seconds <= 0:
            raise ValueError("timeout must be finite and positive")

        with self._state_lock:
            if not self._started or self._proc is None:
                raise WorkerStateError("worker has not been explicitly started")
            if self._closed or self._unusable:
                raise WorkerStateError("worker is closed or unusable")
            if self._in_flight:
                raise WorkerBusy("one request is already in flight")
            if self._proc.poll() is not None:
                self._unusable = True
                raise WorkerStateError(f"owned worker exited with code {self._proc.returncode}")
            self._in_flight = True
            proc = self._proc
            result_queue: queue.Queue[tuple[bool, Frame | BaseException]] = queue.Queue(maxsize=1)

        def io_job() -> None:
            try:
                if proc.stdin is None or proc.stdout is None:
                    raise WorkerStateError("owned worker pipes are unavailable")
                write_all(proc.stdin, frame_bytes)
                response = read_response(proc.stdout, kind, request_id, max_payload_bytes=self.max_payload_bytes)
                result_queue.put_nowait((True, response))
            except BaseException as exc:
                try:
                    result_queue.put_nowait((False, exc))
                except queue.Full:
                    pass

        io_thread = threading.Thread(target=io_job, name="owned-xpu-stdio", daemon=True)
        try:
            io_thread.start()
        except BaseException as exc:
            with self._state_lock:
                self._in_flight = False
                self._unusable = True
            reaped, cleanup_error = self._reap_owned()
            if not reaped:
                raise WorkerCleanupError(f"I/O thread failed and owned child PID {proc.pid} could not be reaped: {cleanup_error}") from exc
            raise

        try:
            ok, result = result_queue.get(timeout=wait_seconds)
        except queue.Empty as exc:
            with self._state_lock:
                self._in_flight = False
                self._unusable = True
            reaped, cleanup_error = self._reap_owned()
            if not reaped:
                raise WorkerCleanupError(
                    f"request timed out and owned child PID {proc.pid} could not be confirmed reaped: {cleanup_error}"
                ) from exc
            raise WorkerTimeout(f"request timed out after {wait_seconds:.3f}s; owned child PID {proc.pid} was reaped") from exc

        with self._state_lock:
            self._in_flight = False
            if not ok:
                self._unusable = True

        if not ok:
            reaped, cleanup_error = self._reap_owned()
            if not reaped:
                raise WorkerCleanupError(
                    f"worker protocol/pipe failure; owned child PID {proc.pid} could not be confirmed reaped: {cleanup_error}"
                ) from result
            if isinstance(result, TransportError):
                raise result
            raise TransportError(f"owned worker I/O failed: {result}") from result

        if not isinstance(result, Frame):
            with self._state_lock:
                self._unusable = True
            reaped, cleanup_error = self._reap_owned()
            if not reaped:
                raise WorkerCleanupError(f"invalid worker result and child PID {proc.pid} not reaped: {cleanup_error}")
            raise TransportError("I/O thread returned an invalid result object")

        if proc.poll() is not None:
            with self._state_lock:
                self._unusable = True
            reaped, cleanup_error = self._reap_owned()
            if not reaped:
                raise WorkerCleanupError(f"worker exited after its response and could not be reaped: {cleanup_error}")
            raise WorkerStateError(f"owned worker exited after response with code {proc.returncode}")
        return result

    def _reap_owned(self) -> tuple[bool, str | None]:
        """Terminate/wait/kill only this object's Popen child, then close owned handles."""
        proc = self._proc
        if proc is None:
            return True, None
        errors: list[str] = []
        with self._cleanup_lock:
            try:
                if proc.poll() is None:
                    try:
                        proc.terminate()
                    except OSError as exc:
                        errors.append(f"terminate: {exc}")
                    try:
                        proc.wait(timeout=self.shutdown_timeout)
                    except subprocess.TimeoutExpired:
                        try:
                            proc.kill()
                        except OSError as exc:
                            errors.append(f"kill: {exc}")
                        try:
                            proc.wait(timeout=self.shutdown_timeout)
                        except subprocess.TimeoutExpired:
                            errors.append("wait timed out after kill")
                    except OSError as exc:
                        errors.append(f"wait: {exc}")
            finally:
                for pipe in (proc.stdin, proc.stdout):
                    if pipe is not None:
                        try:
                            pipe.close()
                        except (OSError, ValueError) as exc:
                            errors.append(f"pipe close: {exc}")
                if self._stderr_file is not None:
                    try:
                        self._stderr_file.close()
                    except OSError as exc:
                        errors.append(f"stderr close: {exc}")
                    self._stderr_file = None
        reaped = proc.poll() is not None
        return reaped, ("; ".join(errors) if errors else None)

    def close(self) -> None:
        with self._state_lock:
            self._closed = True
            self._unusable = True
        reaped, error = self._reap_owned()
        if not reaped:
            raise WorkerCleanupError(f"owned worker PID {self.pid} may remain unreaped: {error or 'unknown wait failure'}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validate-json", type=Path,
                        help="validate a small request descriptor; this CLI never starts a worker")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.validate_json is None:
            request = {"schema": REQUEST_SCHEMA, "version": VERSION, "kind": 0, "request_id": 0, "payload_len": 0}
        else:
            size = args.validate_json.stat().st_size
            if size > MAX_REQUEST_JSON_BYTES:
                raise ProtocolError(f"request JSON exceeds {MAX_REQUEST_JSON_BYTES} bytes")
            request = json.loads(args.validate_json.read_text(encoding="utf-8"))
        validated = validate_request_schema(request)
        print(json.dumps({"status": "schema_valid", "worker_started": False,
                          "header_bytes": HEADER_BYTES, "max_payload_bytes": MAX_PAYLOAD_BYTES,
                          "request": validated}, sort_keys=True))
        return 0
    except (OSError, UnicodeError, json.JSONDecodeError, ProtocolError) as exc:
        parser.exit(2, f"{parser.prog}: {exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())
