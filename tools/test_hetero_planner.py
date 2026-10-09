"""Planner gates and observed completion costs without assets or workers."""
import unittest
from datetime import datetime, timedelta, timezone

from tools.hetero_planner import Acceptance, HeteroCostModel, JobKey, Observation

IDENTITY = {"hardware": "host-a", "runtime": "runtime-a", "model": "model-a", "engine": "engine-a"}
NOW = datetime(2026, 10, 9, tzinfo=timezone.utc)
JOB = JobKey("native-Q4_K-Q5_1", "prefill", 64)
SHA = "a" * 64


def measurement(backend, times, *, job=JOB, quality=True, resources=(), contention=None):
    return Observation(backend, job, tuple(times), "input_ready_to_output_ready", quality, SHA,
                       resources, contention)


def accepted(backend="arc", gain=.09, regression=.01):
    return Acceptance(backend, True, 3, gain, regression, SHA)


class PlannerTests(unittest.TestCase):
    def model(self, extra=None, acceptance=()):
        observations = [measurement("cpu", [1000, 1100, 900])]
        if extra:
            observations.append(extra)
        return HeteroCostModel(IDENTITY, tuple(observations), acceptance, expires_utc=NOW + timedelta(days=1), ema=.5)

    def choose(self, model, **kwargs):
        if "weights_ready_ns" in kwargs:
            kwargs["weights_ready_ns"] = {"cpu": 0, **kwargs["weights_ready_ns"]}
        else:
            kwargs["weights_ready_ns"] = {"cpu": 0}
        return model.choose(JOB, primary="cpu", identity=IDENTITY, now_utc=NOW, now_ns=0, **kwargs)

    def test_queue_and_weight_wait_can_make_faster_compute_slower(self):
        model = self.model(measurement("arc", [600]*3), (accepted(),))
        self.assertEqual(self.choose(model, weights_ready_ns={"arc": 0}).backend, "arc")
        self.assertEqual(self.choose(model, weights_ready_ns={"arc": 600}).backend, "cpu")
        self.assertEqual(self.choose(model, weights_ready_ns={"arc": 0}, busy_until_ns={"arc": 600}).backend, "cpu")

    def test_f32_microbench_cannot_replace_native_operator_or_model_gate(self):
        f32 = JobKey("dense-F32", "prefill", 64)
        self.assertEqual(self.choose(self.model(measurement("arc", [100]*3, job=f32))).backend, "cpu")
        decision = self.choose(self.model(measurement("arc", [100]*3)), weights_ready_ns={"arc": 0})
        self.assertEqual(decision.rejected["arc"], "model_acceptance_missing")

    def test_measured_contention_and_unknown_cache_state_gate(self):
        model = self.model(measurement("arc", [600]*3, resources=("system-RAM",)), (accepted(),))
        decision = self.choose(model, weights_ready_ns={"arc": 0}, active_memory_resources=("system-RAM",))
        self.assertEqual(decision.rejected["arc"], "contention_measurement_missing")
        measured = self.model(measurement("arc", [600]*3, resources=("system-RAM",), contention=2), (accepted(),))
        self.assertEqual(self.choose(measured, weights_ready_ns={"arc": 0}, active_memory_resources=("system-RAM",)).backend, "cpu")
        self.assertEqual(self.choose(model).rejected["arc"], "weight_readiness_missing")

    def test_identity_expiry_and_unmeasured_shape_preserve_primary(self):
        model = self.model()
        bad = dict(IDENTITY, model="different-model")
        self.assertEqual(model.choose(JOB, primary="cpu", identity=bad, now_utc=NOW, now_ns=0).reason, "profile_identity_mismatch")
        self.assertEqual(model.choose(JOB, primary="cpu", identity=IDENTITY, now_utc=NOW+timedelta(days=2), now_ns=0).reason, "profile_expired")
        self.assertEqual(model.choose(JobKey(JOB.operator, "prefill", 65), primary="cpu", identity=IDENTITY, now_utc=NOW, now_ns=0).reason, "primary_cost_unavailable")

    def test_online_update_moves_choice_and_failure_quarantines(self):
        model = self.model(measurement("arc", [600]*3), (accepted(),))
        model.observe("arc", JOB, elapsed_ns=2000, succeeded=True)
        self.assertEqual(self.choose(model, weights_ready_ns={"arc": 0}).backend, "cpu")
        model.observe("arc", JOB, elapsed_ns=1, succeeded=False)
        self.assertEqual(self.choose(model, weights_ready_ns={"arc": 0}).rejected["arc"], "runtime_failure_quarantined")

    def test_end_to_end_threshold_and_margin(self):
        model = self.model(measurement("arc", [100]*3), (accepted(gain=.079),))
        self.assertEqual(self.choose(model, weights_ready_ns={"arc": 0}).rejected["arc"], "main_workload_gain_gate_failed")
        model = self.model(measurement("arc", [100]*3), (accepted(regression=.021),))
        self.assertEqual(self.choose(model, weights_ready_ns={"arc": 0}).rejected["arc"], "stable_regression_gate_failed")
        almost = self.model(measurement("arc", [990]*3), (accepted(),))
        self.assertEqual(self.choose(almost, weights_ready_ns={"arc": 0}).backend, "cpu")

    def test_invalid_evidence_duplicates_and_clock_values_are_refused(self):
        with self.assertRaises(ValueError):
            measurement("cpu", [1, 2])
        with self.assertRaises(ValueError):
            measurement("cpu", [1, True, 3])
        with self.assertRaises(ValueError):
            self.choose(self.model(), busy_until_ns={"cpu": -1})
        obs = measurement("cpu", [1]*3)
        with self.assertRaises(ValueError):
            HeteroCostModel(IDENTITY, (obs, obs), (), expires_utc=NOW)

    def test_distinct_user_acceptance_thresholds_and_npu_memory_benefit(self):
        self.assertIsNone(accepted("cpu", gain=.05).reason())
        self.assertEqual(accepted("storage", gain=.099).reason(), "main_workload_gain_gate_failed")
        self.assertIsNone(accepted("storage", gain=.10).reason())
        self.assertIsNone(accepted("npu", gain=.001).reason())
        memory = Acceptance("npu", True, 3, -.1, .1, SHA, vram_savings_bytes=1 << 30)
        self.assertIsNone(memory.reason())
        self.assertEqual(accepted("npu", gain=0).reason(), "vision_gain_or_vram_savings_missing")
        self.assertEqual(accepted(gain=True).reason(), "model_performance_evidence_missing")

    def test_unknown_primary_weights_are_not_a_fictitious_zero_cost_hit(self):
        model = self.model(measurement("arc", [600]*3), (accepted(),))
        result = model.choose(JOB, primary="cpu", identity=IDENTITY, now_utc=NOW,
                              now_ns=0, weights_ready_ns={"arc": 0})
        self.assertEqual(result.backend, "cpu")
        self.assertEqual(result.reason, "primary_cost_unavailable")
        self.assertEqual(result.rejected["cpu"], "weight_readiness_missing")


if __name__ == "__main__":
    unittest.main()
