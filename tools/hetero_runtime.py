"""Explicit measured profiles and synchronous dispatch for stateless helper jobs.

Importing this module does not open assets, start workers or import a runtime.
Engine integration supplies handlers and output validation; this module owns
only its dispatch lock, timing observations and failure quarantine.
"""
from __future__ import annotations

import hashlib
import json
import threading
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Mapping

from tools.hetero_planner import Acceptance, Decision, HeteroCostModel, JobKey, Observation, identity_hash

PROFILE_SCHEMA = "strata-hetero-profile-v1"
ACCEPTANCE_SCHEMA = "strata-hetero-route-acceptance-v1"
MAX_PROFILE_BYTES = 2 * 1024**2
MAX_EVIDENCE_BYTES = 16 * 1024**2


def _object(pairs):
    out = {}
    for key, value in pairs:
        if key in out:
            raise ValueError(f"duplicate JSON key: {key}")
        out[key] = value
    return out


def _json(raw: bytes):
    return json.loads(raw, object_pairs_hook=_object,
                      parse_constant=lambda value: (_ for _ in ()).throw(ValueError(f"nonfinite JSON: {value}")))


def _read(path: Path, limit: int) -> bytes:
    path = Path(path)
    if not path.is_absolute() or path.is_symlink() or not path.is_file():
        raise ValueError("explicit absolute regular profile/evidence path required")
    before = path.stat()
    if before.st_size > limit:
        raise ValueError("profile/evidence exceeds byte limit")
    with path.open("rb") as stream:
        raw = stream.read(limit + 1)
    after = path.stat()
    fields = ("st_size", "st_mtime_ns", "st_ctime_ns", "st_dev", "st_ino")
    if len(raw) > limit or len(raw) != before.st_size or any(getattr(before, k) != getattr(after, k) for k in fields):
        raise ValueError("profile/evidence changed during bounded read")
    return raw


def profile_bytes(identity: Mapping, model: HeteroCostModel) -> bytes:
    if identity_hash(identity) != model.identity:
        raise ValueError("profile identity differs from measured model")
    value = {"schema": PROFILE_SCHEMA, "identity": dict(identity),
             "expires_utc": model.expires.isoformat(), "ema": model.ema,
             "observations": [asdict(item) for item in model.observations.values()],
             "acceptance": [asdict(item) for item in model.acceptance.values()]}
    raw = (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()
    if len(raw) > MAX_PROFILE_BYTES:
        raise ValueError("serialized profile exceeds byte limit")
    return raw


def write_profile(path: Path, identity: Mapping, model: HeteroCostModel) -> str:
    """Explicit exclusive creation; no setup configuration is changed."""
    path = Path(path)
    if not path.is_absolute():
        raise ValueError("absolute new profile path required")
    raw = profile_bytes(identity, model)
    with path.open("xb") as stream:
        stream.write(raw)
    return hashlib.sha256(raw).hexdigest()


def load_profile(path: Path, *, evidence_by_sha256: Mapping[str, Path]) -> HeteroCostModel:
    """Hash-check explicit small receipts; model acceptance needs its own report.

    A kernel/IPC receipt cannot stand in for a model acceptance report. The
    producer of that report still owns the actual benchmark/correctness checks.
    Loading never treats unavailable evidence as a successful measurement.
    """
    value = _json(_read(path, MAX_PROFILE_BYTES))
    required = {"schema", "identity", "expires_utc", "ema", "observations", "acceptance"}
    if not isinstance(value, dict) or set(value) != required or value["schema"] != PROFILE_SCHEMA:
        raise ValueError("unsupported profile schema or fields")
    identity = identity_hash(value["identity"])
    verified = {}

    def evidence(digest):
        if digest not in verified:
            if digest not in evidence_by_sha256:
                raise ValueError("profile measurement evidence is unavailable")
            raw = _read(evidence_by_sha256[digest], MAX_EVIDENCE_BYTES)
            if hashlib.sha256(raw).hexdigest() != digest:
                raise ValueError("profile measurement evidence hash mismatch")
            verified[digest] = raw
        return verified[digest]

    observations = []
    for raw in value["observations"]:
        item = dict(raw)
        item["job"] = JobKey(**item["job"])
        item["full_job_ns"] = tuple(item["full_job_ns"])
        item["shared_memory_resources"] = tuple(item["shared_memory_resources"])
        observation = Observation(**item)
        evidence(observation.evidence_sha256)
        observations.append(observation)
    acceptance = []
    for raw in value["acceptance"]:
        item = Acceptance(**raw)
        report = _json(evidence(item.evidence_sha256))
        expected = asdict(item)
        del expected["evidence_sha256"]
        if not isinstance(report, dict) or report != {
                "schema": ACCEPTANCE_SCHEMA, "scope": "model_end_to_end",
                "identity_sha256": identity, "acceptance": expected}:
            raise ValueError("helper acceptance requires the matching model report")
        acceptance.append(item)
    return HeteroCostModel(value["identity"], tuple(observations), tuple(acceptance),
                          expires_utc=datetime.fromisoformat(value["expires_utc"]), ema=value["ema"])


class RuntimeBusy(RuntimeError):
    pass


@dataclass(frozen=True)
class DispatchResult:
    output: bytes
    backend: str
    decision: Decision
    attempts: tuple[Mapping, ...]
    elapsed_ns: int


class HeteroRuntimeScheduler:
    """One caller-owned stateless job in flight; helpers require model gates.

    Handlers include serialization, transport, synchronization and their output
    copy. A verifier checks shape/finite/result policy before output is returned.
    Cleanup is required for every helper so failed transport is closed before
    the unchanged input is sent to the primary. KV/session mutation is outside
    this API. Primary failures are reported to the caller, never hidden.
    """

    def __init__(self, model: HeteroCostModel, *, identity: Mapping, primary: str,
                 handlers: Mapping[str, Callable[[JobKey, bytes], bytes]],
                 verify_output: Callable[[JobKey, bytes], None],
                 close_helper: Mapping[str, Callable[[], None]],
                 clock_ns: Callable[[], int] = time.perf_counter_ns,
                 utc_now: Callable[[], datetime] = lambda: datetime.now(timezone.utc)):
        if primary not in handlers or not callable(verify_output):
            raise ValueError("primary handler and explicit output verifier required")
        if any(not callable(h) for h in handlers.values()):
            raise ValueError("each backend requires a callable handler")
        if any(b != primary and (b not in close_helper or not callable(close_helper[b])) for b in handlers):
            raise ValueError("each helper requires exact owned cleanup")
        self.model, self.identity, self.primary = model, dict(identity), primary
        self.handlers, self.verify = dict(handlers), verify_output
        self.cleanup, self.clock, self.utc_now = dict(close_helper), clock_ns, utc_now
        self._lock = threading.Lock()
        self._unavailable: set[str] = set()

    def dispatch(self, job: JobKey, payload: bytes, *, weights_ready_ns: Mapping[str, int],
                 busy_until_ns: Mapping[str, int] | None = None,
                 active_memory_resources: tuple[str, ...] = ()) -> DispatchResult:
        if type(payload) is not bytes:
            raise ValueError("stateless input must be immutable bytes")
        if not self._lock.acquire(blocking=False):
            raise RuntimeBusy("a stateless job is already in flight")
        try:
            start = self.clock()
            # This synchronous dispatcher accepts idle, resident helpers only.
            # Forecasts for queued/pending loads remain useful in the cost model,
            # but do not authorize calling an unready helper here.
            ready = {b: t for b, t in weights_ready_ns.items() if b in self.handlers and b not in self._unavailable
                     and (b == self.primary or (t <= start and (busy_until_ns or {}).get(b, start) <= start))}
            decision = self.model.choose(job, primary=self.primary, identity=self.identity,
                                         now_utc=self.utc_now(), now_ns=start, weights_ready_ns=ready,
                                         busy_until_ns=busy_until_ns,
                                         active_memory_resources=active_memory_resources)
            attempts = []
            backend = decision.backend
            while True:
                entered = self.clock()
                try:
                    output = self.handlers[backend](job, payload)
                    if type(output) is not bytes:
                        raise ValueError("backend output must be owned immutable bytes")
                    self.verify(job, output)
                except Exception as exc:
                    elapsed = self.clock() - entered
                    attempts.append({"backend": backend, "elapsed_ns": elapsed, "succeeded": False,
                                     "error": f"{type(exc).__name__}: {exc}"})
                    self._observe(backend, job, max(1, elapsed), False)
                    if backend == self.primary:
                        raise
                    self._unavailable.add(backend)
                    self.cleanup[backend]()  # A cleanup failure prevents fallback.
                    backend = self.primary
                    continue
                elapsed = self.clock() - entered
                attempts.append({"backend": backend, "elapsed_ns": elapsed, "succeeded": True})
                self._observe(backend, job, max(1, elapsed), True)
                return DispatchResult(output, backend, decision, tuple(attempts), self.clock() - start)
        finally:
            self._lock.release()

    def _observe(self, backend, job, elapsed, success):
        if (backend, job) in self.model.observations:
            self.model.observe(backend, job, elapsed_ns=elapsed, succeeded=success)
