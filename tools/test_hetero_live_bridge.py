"""Pure-host fixtures. No subprocess, NumPy, OpenVINO, model or device access."""
import hashlib
import json
import struct
import unittest
from tools import hetero_live_bridge as live
from tools import hetero_xpu_transport as transport


class MockWorker:
    def __init__(self, identity, identity_sha, *, wrong_device=False, fail=False):
        self.identity, self.identity_sha = identity, identity_sha
        self.wrong_device, self.fail, self.calls = wrong_device, fail, []

    def request(self, kind, request_id, payload):
        self.calls.append((kind, payload))
        if kind == 1:
            init = json.loads(payload)
            ready = {"status": "ready", "rows": init["rows"], "nonce": init["nonce"],
                "identity_sha256": self.identity_sha, "execution_devices": ["CPU" if self.wrong_device else "GPU.0"],
                "device": "GPU.0", "operator_signature": "Q4_K_Q8_K_SILU_Q5_1_Q8_1",
                "weights_loaded_and_hash_verified": True, "weights_sha256": self.identity["weights_sha256"],
                "native_blob_binding": {"segment_sha256_actual": self.identity["source"]["raw_selected_payload_sha256"]}}
            body = json.dumps(ready).encode()
        else:
            body = b"" if self.fail else payload
        return transport.Frame(kind | 0x8000, request_id, live.RESULT.pack(0, 123) + body)


class LiveTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.blob = bytes(live.BLOB_BYTES)
        cls.identity = {"layer": 28, "expert": 288, "weights_sha256": {"gate": "mock"},
            "source": {"quantized_source_types": {"gate": "Q4_K", "up": "Q4_K", "down": "Q5_1"},
            "raw_selected_payload_sha256": {role: hashlib.sha256(cls.blob[o:o+n]).hexdigest()
                for role, (o, n) in live.SEGMENTS.items()}}}

    def payload(self, rows=1):
        return live.META.pack(28, 288, rows, 2560, live.BLOB_BYTES) + self.blob + struct.pack("<f", 0.25) * (rows*2560)

    def test_actual_payload_roundtrip_and_static_rows_reuse(self):
        mock = MockWorker(self.identity, "a"*64)
        bridge = live.LiveBridge(self.identity, "a"*64, mock)
        for rows in (1, 1, 2):
            result = bridge.infer(self.payload(rows))
            self.assertEqual(result[12:], struct.pack("<f", 0.25) * (rows*2560))
        self.assertEqual([kind for kind, _ in mock.calls], [1, 2, 2, 1, 2])

    def test_changed_engine_blob_never_reaches_worker(self):
        mock = MockWorker(self.identity, "a"*64)
        payload = bytearray(self.payload()); payload[live.META.size] ^= 1
        with self.assertRaisesRegex(ValueError, "hash mismatch"):
            live.LiveBridge(self.identity, "a"*64, mock).infer(bytes(payload))
        self.assertEqual(mock.calls, [])

    def test_device_and_output_shape_rejection(self):
        for kwargs in ({"wrong_device": True}, {"fail": True}):
            with self.assertRaises(ValueError):
                live.LiveBridge(self.identity, "a"*64, MockWorker(self.identity, "a"*64, **kwargs)).infer(self.payload())

    def test_nonfinite_and_unsupported_rows_rejected(self):
        payload = self.payload()[:-4] + struct.pack("<f", float("nan"))
        with self.assertRaisesRegex(ValueError, "nonfinite"):
            live.validate_payload(payload, self.identity)
        with self.assertRaisesRegex(ValueError, "unsupported"):
            live.validate_payload(self.payload(9), self.identity)


if __name__ == "__main__":
    unittest.main()
