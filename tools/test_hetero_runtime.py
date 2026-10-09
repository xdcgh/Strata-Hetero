"""Profile provenance, complete dispatch boundaries and failure quarantine."""
import hashlib
import json
import tempfile
import unittest
from dataclasses import asdict
from datetime import datetime, timedelta, timezone
from pathlib import Path

from tools.hetero_planner import Acceptance, HeteroCostModel, JobKey, Observation, identity_hash
from tools.hetero_runtime import (ACCEPTANCE_SCHEMA, HeteroRuntimeScheduler, RuntimeBusy,
                                  load_profile, profile_bytes, write_profile)

IDENTITY = {"hardware": "fixture-host", "runtime": "fixture-runtime", "model": "fixture-model", "engine": "fixture-engine"}
NOW = datetime(2026, 10, 9, tzinfo=timezone.utc)
JOB = JobKey("fixture-native-expert-weights-sha", "prefill", 64)
DIGEST = "a" * 64


def model(*, acceptance=True, digest=DIGEST, helper_digest=DIGEST):
    observations = (Observation("cpu", JOB, (1000, 1000, 1000), "input_ready_to_output_ready", True, digest),
                    Observation("arc", JOB, (600, 600, 600), "input_ready_to_output_ready", True, digest))
    accepted = (Acceptance("arc", True, 3, .09, .01, helper_digest),) if acceptance else ()
    return HeteroCostModel(IDENTITY, observations, accepted, expires_utc=NOW + timedelta(days=1))


class Clock:
    def __init__(self):
        self.ns = 10000

    def __call__(self):
        self.ns += 100
        return self.ns


class RuntimeTests(unittest.TestCase):
    def scheduler(self, measured=None, *, arc=None, verifier=None, cleanup=None, identity=IDENTITY, clock=None):
        self.calls, self.closed = [], []

        def cpu(job, payload):
            self.calls.append(("cpu", job, payload))
            return b"cpu-output"

        def helper(job, payload):
            self.calls.append(("arc", job, payload))
            return arc(job, payload) if arc else b"arc-output"

        return HeteroRuntimeScheduler(measured or model(), identity=identity, primary="cpu",
                                     handlers={"cpu": cpu, "arc": helper},
                                     verify_output=verifier or (lambda job, output: None),
                                     close_helper={"arc": cleanup or (lambda: self.closed.append("arc"))},
                                     clock_ns=clock or Clock(), utc_now=lambda: NOW)

    def dispatch(self, runtime, **kwargs):
        return runtime.dispatch(JOB, b"immutable-input", weights_ready_ns=kwargs.pop("ready", {"cpu": 0, "arc": 0}), **kwargs)

    def test_accepted_helper_dispatches_and_reports_complete_boundary(self):
        runtime = self.scheduler()
        result = self.dispatch(runtime)
        self.assertEqual(result.output, b"arc-output")
        self.assertEqual(result.backend, "arc")
        self.assertGreater(result.elapsed_ns, result.attempts[0]["elapsed_ns"])
        self.assertEqual(self.calls[0][2], b"immutable-input")
        self.assertNotEqual(runtime.model.online_ns[("arc", JOB)], 600)

    def test_microbench_only_expired_or_mismatched_profile_keeps_primary(self):
        self.assertEqual(self.dispatch(self.scheduler(model(acceptance=False))).backend, "cpu")
        self.assertEqual(self.dispatch(self.scheduler(identity=dict(IDENTITY, engine="other"))).backend, "cpu")
        measured = model()
        measured.expires = NOW
        self.assertEqual(self.dispatch(self.scheduler(measured)).backend, "cpu")

    def test_output_validation_failure_closes_then_falls_back_with_original_input(self):
        order = []

        def verify(job, output):
            order.append(output)
            if output == b"arc-output":
                raise ValueError("invalid helper tensor")

        runtime = self.scheduler(verifier=verify, cleanup=lambda: order.append(b"closed"))
        result = self.dispatch(runtime)
        self.assertEqual(order, [b"arc-output", b"closed", b"cpu-output"])
        self.assertEqual([item["succeeded"] for item in result.attempts], [False, True])
        self.assertEqual([item[2] for item in self.calls], [b"immutable-input"] * 2)
        self.assertIn("arc", runtime.model.quarantined)
        self.assertEqual(self.dispatch(runtime).backend, "cpu")

    def test_cleanup_failure_prevents_fallback_and_primary_failure_propagates(self):
        def fail(*args):
            raise RuntimeError("owned close failed")

        runtime = self.scheduler(arc=fail, cleanup=fail)
        with self.assertRaisesRegex(RuntimeError, "owned close failed"):
            self.dispatch(runtime)
        self.assertEqual(len(self.calls), 1)
        runtime = self.scheduler(model(acceptance=False), verifier=fail)
        with self.assertRaisesRegex(RuntimeError, "owned close failed"):
            self.dispatch(runtime)

    def test_unready_or_busy_helper_and_mutable_input_are_refused(self):
        runtime = self.scheduler()
        self.assertEqual(self.dispatch(runtime, ready={"cpu": 0, "arc": 100000}).backend, "cpu")
        self.assertEqual(self.dispatch(runtime, busy_until_ns={"arc": 100000}).backend, "cpu")
        with self.assertRaises(ValueError):
            runtime.dispatch(JOB, bytearray(b"x"), weights_ready_ns={})
        self.assertTrue(runtime._lock.acquire(blocking=False))
        try:
            with self.assertRaises(RuntimeBusy):
                self.dispatch(runtime)
        finally:
            runtime._lock.release()

    def test_profile_roundtrip_hashes_receipts_and_requires_matching_model_report(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            kernel = root / "kernel.json"
            kernel.write_bytes(b'{"scope":"kernel_fixture"}')
            digest = hashlib.sha256(kernel.read_bytes()).hexdigest()
            accepted = asdict(Acceptance("arc", True, 3, .09, .01, DIGEST))
            del accepted["evidence_sha256"]
            report = root / "model.json"
            report.write_text(json.dumps({"schema": ACCEPTANCE_SCHEMA, "scope": "model_end_to_end",
                                          "identity_sha256": identity_hash(IDENTITY), "acceptance": accepted}), encoding="utf-8")
            report_digest = hashlib.sha256(report.read_bytes()).hexdigest()
            path = root / "hetero-profile.json"
            measured = model(digest=digest, helper_digest=report_digest)
            write_profile(path, IDENTITY, measured)
            loaded = load_profile(path, evidence_by_sha256={digest: kernel, report_digest: report})
            self.assertEqual(profile_bytes(IDENTITY, loaded), path.read_bytes())
            self.assertEqual(self.dispatch(self.scheduler(loaded)).backend, "arc")
            with self.assertRaises(FileExistsError):
                write_profile(path, IDENTITY, measured)
            with self.assertRaisesRegex(ValueError, "unavailable"):
                load_profile(path, evidence_by_sha256={})
            kernel.write_bytes(b"changed")
            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                load_profile(path, evidence_by_sha256={digest: kernel, report_digest: report})

    def test_kernel_receipt_cannot_authorize_model_route_and_duplicate_json_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            evidence = root / "kernel.json"
            evidence.write_bytes(b'{"scope":"kernel_fixture"}')
            digest = hashlib.sha256(evidence.read_bytes()).hexdigest()
            path = root / "profile.json"
            write_profile(path, IDENTITY, model(digest=digest, helper_digest=digest))
            with self.assertRaisesRegex(ValueError, "matching model report"):
                load_profile(path, evidence_by_sha256={digest: evidence})
            path.write_bytes(b'{"schema":"a","schema":"b"}')
            with self.assertRaisesRegex(ValueError, "duplicate"):
                load_profile(path, evidence_by_sha256={})
            with self.assertRaises(ValueError):
                profile_bytes(dict(IDENTITY, engine="other"), model())
        with self.assertRaises(ValueError):
            HeteroCostModel(IDENTITY, (), (), expires_utc=NOW, ema=True)


if __name__ == "__main__":
    unittest.main()
