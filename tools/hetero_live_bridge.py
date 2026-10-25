"""Explicit live-model SXPU bridge. Importing this module never starts a worker.

Kind 10 carries five LE u32 fields (layer, expert, rows, hidden, blob bytes),
the actual engine expert blob, then actual model LE F32 input rows. Response
uses the existing status:u32, worker_ns:u64 service envelope and LE F32 rows.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import secrets
import struct
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from tools import hetero_xpu_transport as transport

META = struct.Struct("<IIIII")
RESULT = struct.Struct("<IQ")
SEGMENTS = {"gate": (0, 921600), "up": (921600, 921600), "down": (1843200, 1228800)}
BLOB_BYTES = 3072000


def validate_payload(payload: bytes, identity: dict) -> tuple[int, bytes]:
    if len(payload) < META.size:
        raise ValueError("short live request")
    layer, expert, rows, hidden, blob_bytes = META.unpack_from(payload)
    if (layer, expert, hidden, blob_bytes) != (28, 288, 2560, BLOB_BYTES) or not 1 <= rows <= 8:
        raise ValueError("unsupported live expert/shape")
    if (identity.get("layer"), identity.get("expert")) != (layer, expert):
        raise ValueError("identity layer/expert mismatch")
    if len(payload) != META.size + blob_bytes + rows * hidden * 4:
        raise ValueError("live request length mismatch")
    source = identity.get("source", {})
    if source.get("quantized_source_types") != {"gate": "Q4_K", "up": "Q4_K", "down": "Q5_1"}:
        raise ValueError("identity quantization mismatch")
    blob = memoryview(payload)[META.size:META.size + blob_bytes]
    expected = source.get("raw_selected_payload_sha256", {})
    for role, (offset, count) in SEGMENTS.items():
        if hashlib.sha256(blob[offset:offset+count]).hexdigest() != expected.get(role):
            raise ValueError(f"actual engine {role} hash mismatch")
    raw = payload[META.size + blob_bytes:]
    # Pure stdlib validation: no NumPy or device imports in the bridge.
    import math
    if any(not math.isfinite(v[0]) for v in struct.iter_unpack("<f", raw)):
        raise ValueError("nonfinite model input")
    return rows, raw


class LiveBridge:
    def __init__(self, identity, identity_sha, worker):
        self.identity, self.identity_sha, self.worker = identity, identity_sha, worker
        self.rows = None
        self.request_id = 0

    def request(self, kind, payload):
        self.request_id += 1
        frame = self.worker.request(kind, self.request_id, payload)
        if frame.kind != kind | 0x8000 or frame.request_id != self.request_id or len(frame.payload) < RESULT.size:
            raise ValueError("worker response mismatch")
        status, elapsed = RESULT.unpack_from(frame.payload)
        if status:
            raise ValueError("Arc service rejected request: " + frame.payload[RESULT.size:].decode("utf-8", "replace")[:512])
        return frame.payload[RESULT.size:], elapsed

    def infer(self, payload):
        rows, raw = validate_payload(payload, self.identity)
        if rows != self.rows:
            nonce = secrets.token_hex(16)
            body, _ = self.request(1, json.dumps({"schema": "strata-xpu-init-v1", "nonce": nonce,
                "rows": rows, "identity_sha256": self.identity_sha}, separators=(",", ":")).encode("ascii"))
            ready = json.loads(body)
            binding = ready.get("native_blob_binding", {})
            checks = (ready.get("status") == "ready", ready.get("rows") == rows,
                      ready.get("nonce") == nonce, ready.get("identity_sha256") == self.identity_sha,
                      ready.get("execution_devices") == ["GPU.0"], ready.get("device") == "GPU.0",
                      ready.get("operator_signature") == "Q4_K_Q8_K_SILU_Q5_1_Q8_1",
                      ready.get("weights_loaded_and_hash_verified") is True,
                      ready.get("weights_sha256") == self.identity["weights_sha256"],
                      binding.get("segment_sha256_actual") == self.identity["source"]["raw_selected_payload_sha256"])
            if not all(checks):
                raise ValueError("Arc READY identity/device/operator mismatch")
            self.rows = rows
        output, worker_ns = self.request(2, raw)
        if len(output) != rows * 2560 * 4:
            raise ValueError("Arc output shape mismatch")
        import math
        if any(not math.isfinite(v[0]) for v in struct.iter_unpack("<f", output)):
            raise ValueError("nonfinite Arc output")
        return RESULT.pack(0, worker_ns) + output


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--identity", required=True, type=Path)
    parser.add_argument("--identity-sha256", required=True)
    parser.add_argument("--weights", required=True, type=Path)
    parser.add_argument("--native-blob", required=True, type=Path)
    parser.add_argument("--stderr", required=True, type=Path, help="exclusive Arc diagnostic file")
    parser.add_argument("--python", default=sys.executable)
    args = parser.parse_args(argv)
    for path in (args.identity, args.weights, args.native_blob, args.stderr):
        if not path.is_absolute():
            parser.error("all artifact/log paths must be absolute")
    with args.identity.open("rb") as stream:
        raw_identity = stream.read(65537)
    if len(raw_identity) > 65536 or hashlib.sha256(raw_identity).hexdigest() != args.identity_sha256:
        parser.error("identity hash/size mismatch")
    identity = json.loads(raw_identity)
    # The engine owns the bridge in a Windows kill-on-close Job Object; this
    # owned child inherits that job and is also reaped by OwnedPipeWorker.
    worker = transport.OwnedPipeWorker([args.python, str(ROOT / "tools/hetero_xpu_service.py"),
        "--serve", "--device", "GPU.0", "--precision", "f32", "--operator", "native-q4k-q5_1",
        "--weights", str(args.weights), "--identity", str(args.identity),
        "--native-blob", str(args.native_blob)], args.stderr, request_timeout=110)
    bridge = LiveBridge(identity, args.identity_sha256, worker)
    started = False
    try:
        while True:
            frame = transport.read_frame(sys.stdin.buffer)
            if frame is None:
                return 0
            try:
                if frame.kind != 10:
                    raise ValueError("unsupported live request kind")
                # Validate actual model identity before creating any worker/device.
                validate_payload(frame.payload, identity)
                if not started:
                    worker.start(); started = True
                body = bridge.infer(frame.payload)
            except Exception as exc:
                body = RESULT.pack(1, 0) + json.dumps({"message": str(exc)[:512]}).encode("utf-8")
                transport.write_all(sys.stdout.buffer, transport.encode_response_frame(frame.kind, frame.request_id, body))
                return 2  # Poisoned bridge cannot accidentally reuse stale service state.
            transport.write_all(sys.stdout.buffer, transport.encode_response_frame(frame.kind, frame.request_id, body))
    finally:
        worker.close()


if __name__ == "__main__":
    raise SystemExit(main())
