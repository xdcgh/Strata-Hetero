"""Measured heterogeneous job costs; callers retain dispatch and state ownership.

Full job observations already include transport and synchronization. Component
timers are explanatory and must not be added to those observations again.
"""
from __future__ import annotations

import hashlib
import json
import math
import statistics
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Mapping


def identity_hash(identity: Mapping) -> str:
    required = {"hardware", "runtime", "model", "engine"}
    if set(identity) != required or any(not identity[k] for k in required):
        raise ValueError("profile identity requires hardware/runtime/model/engine")
    return hashlib.sha256(json.dumps(identity, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def positive_ns(value: int, name: str) -> int:
    if type(value) is not int or not 0 < value <= 2**63 - 1:
        raise ValueError(f"{name} must be positive bounded integer nanoseconds")
    return value


@dataclass(frozen=True)
class JobKey:
    operator: str
    phase: str
    rows: int

    def __post_init__(self):
        if not isinstance(self.operator, str) or not self.operator or self.phase not in ("prefill", "decode", "vision"):
            raise ValueError("explicit operator and supported phase required")
        if type(self.rows) is not int or self.rows <= 0:
            raise ValueError("rows must be positive; shape extrapolation is not implicit")


@dataclass(frozen=True)
class Observation:
    backend: str
    job: JobKey
    full_job_ns: tuple[int, ...]
    scope: str
    quality_pass: bool
    evidence_sha256: str
    shared_memory_resources: tuple[str, ...] = ()
    contention_multiplier: float | None = None

    def __post_init__(self):
        if not isinstance(self.backend, str) or not self.backend or self.scope != "input_ready_to_output_ready":
            raise ValueError("cost must cover complete input-ready to output-ready boundary")
        if len(self.full_job_ns) < 3:
            raise ValueError("three formal observations required; exclude warmup")
        for value in self.full_job_ns:
            positive_ns(value, "full_job_ns")
        if type(self.quality_pass) is not bool:
            raise ValueError("quality gate must be known")
        if len(self.evidence_sha256) != 64 or any(c not in "0123456789abcdef" for c in self.evidence_sha256):
            raise ValueError("measured evidence digest required")
        factor = self.contention_multiplier
        if factor is not None and (type(factor) not in (int, float) or not math.isfinite(factor) or factor < 1):
            raise ValueError("contention multiplier must be measured, finite and at least one")

    @property
    def median_ns(self) -> float:
        return float(statistics.median(self.full_job_ns))


@dataclass(frozen=True)
class Acceptance:
    backend: str
    model_quality_pass: bool
    model_formals: int
    main_workload_gain: float | None
    stable_regression: float | None
    evidence_sha256: str
    vram_savings_bytes: int = 0

    def reason(self) -> str | None:
        if self.model_quality_pass is not True or type(self.model_formals) is not int or self.model_formals < 3:
            return "model_quality_or_repeated_model_evidence_missing"
        if len(self.evidence_sha256) != 64 or any(c not in "0123456789abcdef" for c in self.evidence_sha256):
            return "model_evidence_digest_missing"
        gain, regression = self.main_workload_gain, self.stable_regression
        if any(type(x) not in (int, float) or not math.isfinite(x) for x in (gain, regression)):
            return "model_performance_evidence_missing"
        if type(self.vram_savings_bytes) is not int or self.vram_savings_bytes < 0:
            return "vram_evidence_invalid"
        kind = self.backend.split(":", 1)[0]
        if kind not in ("arc", "cpu", "storage", "npu"):
            return "unsupported_acceptance_kind"
        if regression < 0 or (kind == "arc" and regression > .02):
            return "stable_regression_gate_failed"
        if kind == "npu":
            return None if gain > 0 or self.vram_savings_bytes > 0 else "vision_gain_or_vram_savings_missing"
        threshold = {"arc": .08, "cpu": .05, "storage": .10}[kind]
        if gain < threshold:
            return "main_workload_gain_gate_failed"
        return None


@dataclass(frozen=True)
class Decision:
    backend: str
    completion_eta_ns: int | None
    reason: str
    rejected: Mapping[str, str]


class HeteroCostModel:
    """Exact measured shapes, identity/age checks and bounded online updates.

    This selector has no worker launch, process mutation, graph execution or
    persistent configuration side effect. The engine's existing primary route
    remains its fallback when evidence is missing or a helper is quarantined.
    """

    def __init__(self, identity: Mapping, observations: tuple[Observation, ...],
                 acceptance: tuple[Acceptance, ...], *, expires_utc: datetime, ema: float = .2):
        self.identity = identity_hash(identity)
        if expires_utc.tzinfo is None or not 0 < ema <= 1:
            raise ValueError("timezone-aware expiry and EMA in (0,1] required")
        self.expires = expires_utc.astimezone(timezone.utc)
        self.ema = ema
        self.observations = {}
        for observation in observations:
            key = (observation.backend, observation.job)
            if key in self.observations:
                raise ValueError("duplicate measured route/shape")
            self.observations[key] = observation
        self.acceptance = {}
        for item in acceptance:
            if item.backend in self.acceptance:
                raise ValueError("duplicate acceptance route")
            self.acceptance[item.backend] = item
        self.online_ns: dict[tuple[str, JobKey], float] = {}
        self.quarantined: set[str] = set()

    def observe(self, backend: str, job: JobKey, *, elapsed_ns: int, succeeded: bool) -> None:
        key = (backend, job)
        if key not in self.observations:
            raise ValueError("cannot calibrate an unidentified operator/shape")
        if type(succeeded) is not bool:
            raise ValueError("completion status must be known")
        if not succeeded:
            self.quarantined.add(backend)
            return
        positive_ns(elapsed_ns, "elapsed_ns")
        previous = self.online_ns.get(key, self.observations[key].median_ns)
        self.online_ns[key] = (1 - self.ema) * previous + self.ema * elapsed_ns

    def choose(self, job: JobKey, *, primary: str, identity: Mapping, now_utc: datetime,
               now_ns: int, busy_until_ns: Mapping[str, int] | None = None,
               weights_ready_ns: Mapping[str, int] | None = None,
               active_memory_resources: tuple[str, ...] = (), minimum_job_gain: float = .02) -> Decision:
        if now_utc.tzinfo is None or type(now_ns) is not int or now_ns < 0 or not 0 <= minimum_job_gain < 1:
            raise ValueError("valid explicit clocks and gain margin required")
        if identity_hash(identity) != self.identity:
            return Decision(primary, None, "profile_identity_mismatch", {})
        if now_utc.astimezone(timezone.utc) >= self.expires:
            return Decision(primary, None, "profile_expired", {})
        busy, ready = busy_until_ns or {}, weights_ready_ns or {}
        for value in (*busy.values(), *ready.values()):
            if type(value) is not int or value < 0:
                raise ValueError("queue/weight readiness must be known nonnegative clocks")
        rejected = {}
        estimates = {}
        for (backend, key), observation in self.observations.items():
            if key != job:
                continue
            reason = None
            if not observation.quality_pass:
                reason = "operator_quality_gate_failed"
            elif backend in self.quarantined:
                reason = "runtime_failure_quarantined"
            elif backend != primary:
                accepted = self.acceptance.get(backend)
                reason = accepted.reason() if accepted else "model_acceptance_missing"
                if accepted and accepted.backend != backend:
                    reason = "acceptance_backend_mismatch"
            contended = bool(set(observation.shared_memory_resources) & set(active_memory_resources))
            if reason is None and contended and observation.contention_multiplier is None:
                reason = "contention_measurement_missing"
            if reason:
                rejected[backend] = reason
                continue
            duration = self.online_ns.get((backend, job), observation.median_ns)
            if contended:
                duration *= observation.contention_multiplier
            # File-backed primary weights also have a readiness cost.
            if backend not in ready:
                rejected[backend] = "weight_readiness_missing"
                continue
            start = max(now_ns, busy.get(backend, now_ns), ready.get(backend, now_ns))
            estimates[backend] = start + math.ceil(duration)
        if primary not in estimates:
            return Decision(primary, None, "primary_cost_unavailable", rejected)
        primary_eta = estimates[primary]
        helper = min(estimates, key=lambda backend: (estimates[backend], backend != primary, backend))
        if helper != primary and estimates[helper] - now_ns < (primary_eta - now_ns) * (1 - minimum_job_gain):
            return Decision(helper, estimates[helper], "minimum_measured_completion_eta", rejected)
        return Decision(primary, primary_eta, "primary_with_gain_margin", rejected)
