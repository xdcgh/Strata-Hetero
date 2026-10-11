#!/usr/bin/env python3
"""Small, auditable loopback SSE benchmark client for heterogeneous Strata runs."""
from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import math
import os
import re
import statistics
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

DEFAULT_BASE = "http://127.0.0.1:8081"
_SECRET_KEYS = {"api_key", "apikey", "x_api_key", "authorization", "access_token",
                "refresh_token", "auth_token", "client_secret", "api_secret", "password", "secret", "token"}


class _LoopbackNoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


_NO_PROXY_HANDLER = urllib.request.ProxyHandler({})
_NO_REDIRECT_HANDLER = _LoopbackNoRedirect()
_LOOPBACK_OPENER = urllib.request.build_opener(_NO_PROXY_HANDLER, _NO_REDIRECT_HANDLER)


def safe_open(req: urllib.request.Request, timeout: float):
    """Never use environment proxies and never follow a redirect away from the loopback target."""
    validate_base_url(urllib.parse.urlunsplit((urllib.parse.urlsplit(req.full_url).scheme,
                                              urllib.parse.urlsplit(req.full_url).netloc,
                                              "", "", "")))
    return _LOOPBACK_OPENER.open(req, timeout=timeout)


def redact(value: Any, secret: str = "") -> Any:
    if isinstance(value, dict):
        return {k: ("[redacted]" if str(k).lower().replace("-", "_") in _SECRET_KEYS else redact(v, secret)) for k, v in value.items()}
    if isinstance(value, list):
        return [redact(v, secret) for v in value]
    if isinstance(value, str):
        return value.replace(secret, "[redacted]") if secret else value
    return value


def validate_base_url(url: str) -> str:
    p = urllib.parse.urlsplit(url)
    if p.scheme != "http" or p.hostname not in ("127.0.0.1", "localhost", "::1"):
        raise ValueError("base URL must use http and a loopback hostname (127.0.0.1, localhost, or ::1)")
    if p.username or p.password or p.query or p.fragment:
        raise ValueError("base URL must not contain credentials, query, or fragment")
    return url.rstrip("/")


def load_identity(path: str | None) -> tuple[dict[str, Any], bool]:
    if not path:
        return {"verification": "unverified", "accepted_speedup": False}, False
    raw = Path(path).read_bytes()
    obj = json.loads(raw.decode("utf-8-sig"))
    if not isinstance(obj, dict):
        raise ValueError("identity JSON must be an object")
    required = ("schema", "source", "config", "model", "engine")
    absent = [k for k in required if k not in obj]
    if absent:
        raise ValueError("identity JSON missing fields: " + ", ".join(absent))
    return redact(obj), True


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--prompt-file", required=True)
    ap.add_argument("--output", required=True, help="root directory for a new unique run directory")
    ap.add_argument("--repeats", type=int, default=3, help="formal requests per worker; minimum 3")
    ap.add_argument("--warmup", type=int, default=1)
    ap.add_argument("--max-tokens", type=int, required=True)
    ap.add_argument("--concurrency", type=int, default=1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--reasoning-effort", default="none")
    ap.add_argument("--base-url", default=DEFAULT_BASE)
    ap.add_argument("--validate-only", action="store_true")
    ap.add_argument("--identity-json")
    ap.add_argument("--allow-capped-performance", action="store_true",
                    help="include clean finish_reason=length samples in timing statistics; quality remains pending")
    ap.add_argument("--model", default="")
    ap.add_argument("--timeout", type=float, default=1800)
    ap.add_argument("--run-id", default="")
    return ap.parse_args(argv)


def validate_args(a: argparse.Namespace) -> tuple[Path, Path, dict[str, Any], bool]:
    if a.repeats < 3 or a.warmup < 1 or a.max_tokens < 1 or a.concurrency < 1 or a.timeout <= 0:
        raise ValueError("repeats >= 3, warmup >= 1, max-tokens/concurrency/timeout > 0 are required")
    base = validate_base_url(a.base_url)
    prompt_path = Path(a.prompt_file).expanduser().resolve(strict=True)
    if not prompt_path.is_file() or prompt_path.stat().st_size == 0:
        raise ValueError("prompt file must exist and be non-empty")
    prompt = prompt_path.read_bytes()
    if not prompt:
        raise ValueError("prompt file must be non-empty")
    root = Path(a.output).expanduser().resolve()
    run_id = a.run_id or (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:8])
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,79}", run_id):
        raise ValueError("run-id may contain only letters, digits, dot, underscore, hyphen")
    run_dir = root / run_id
    if run_dir.exists():
        raise FileExistsError(f"run directory already exists: {run_dir}")
    identity, supplied = load_identity(a.identity_json)
    return prompt_path, run_dir, identity, supplied


def _read_sse(resp, secret: str, started: float) -> dict[str, Any]:
    events: list[dict[str, Any]] = []
    raw_lines: list[str] = []
    data_lines: list[str] = []
    content_parts: list[str] = []
    reasoning_parts: list[str] = []
    first_generated = first_content = None
    usage = timings = None
    finish_reason = None
    api_error = None
    done = False

    def dispatch() -> None:
        nonlocal first_generated, first_content, usage, timings, finish_reason, api_error, done
        if not data_lines:
            return
        payload = "\n".join(data_lines)
        data_lines.clear()
        if payload.strip() == "[DONE]":
            done = True
            return
        try:
            event = json.loads(payload)
        except json.JSONDecodeError as e:
            raise RuntimeError("invalid JSON in SSE data event") from e
        if not isinstance(event, dict):
            raise RuntimeError("SSE JSON event is not an object")
        events.append(redact(event, secret))
        if "error" in event:
            api_error = redact(event["error"], secret)
        if isinstance(event.get("usage"), dict):
            usage = event["usage"]
        if isinstance(event.get("timings"), dict):
            timings = event["timings"]
        choices = event.get("choices") or []
        if choices:
            choice = choices[0] or {}
            delta = choice.get("delta") or {}
            # Role-only/empty deltas do not count as generated output.
            reasoning = delta.get("reasoning_content")
            content = delta.get("content")
            has_reason = isinstance(reasoning, str) and bool(reasoning)
            has_content = isinstance(content, str) and bool(content)
            now = time.perf_counter()
            if first_generated is None and (has_reason or has_content):
                first_generated = now - started
            if first_content is None and has_content:
                first_content = now - started
            if has_reason:
                reasoning_parts.append(reasoning)
            if has_content:
                content_parts.append(content)
            if choice.get("finish_reason") is not None:
                finish_reason = choice["finish_reason"]

    # SSE uses UTF-8. utf-8-sig safely consumes a BOM on the first line.
    stream_error = None
    try:
        for raw in resp:
            line = raw.decode("utf-8-sig", errors="strict").rstrip("\r\n")
            raw_lines.append(line.replace(secret, "[redacted]") if secret else line)
            if line == "":
                dispatch()
                if done:
                    break
            elif line.startswith(":"):
                continue
            elif ":" in line:
                field, value = line.split(":", 1)
                if value.startswith(" "):
                    value = value[1:]
                if field == "data":
                    data_lines.append(value)
    except Exception as e:
        stream_error = f"{type(e).__name__}: {redact(str(e), secret)}"
    if not done and data_lines:
        try:
            dispatch()
        except Exception as e:
            stream_error = f"{type(e).__name__}: {redact(str(e), secret)}"
    if not done and stream_error is None:
        stream_error = "truncated SSE stream: missing data: [DONE]"
    wall = time.perf_counter() - started
    if api_error:
        status = "server_error"
    elif stream_error == "truncated SSE stream: missing data: [DONE]":
        status = "truncated"
    elif stream_error:
        status = "timeout" if "Timeout" in stream_error else "error"
    else:
        status = None
    return {"events": events, "raw_lines": raw_lines, "content": "".join(content_parts),
            "reasoning": "".join(reasoning_parts), "first_generated_s": first_generated,
            "first_visible_content_s": first_content, "finish_reason": finish_reason,
            "usage": usage, "timings": timings, "client_e2e_s": wall, "status": status,
            "error": "server returned an SSE error event" if api_error else stream_error}


def request_once(base: str, prompt: str, a: argparse.Namespace, run_dir: Path, label: str,
                 index: int, secret: str) -> dict[str, Any]:
    body: dict[str, Any] = {"messages": [{"role": "user", "content": prompt}], "stream": True,
                            "stream_options": {"include_usage": True}, "max_tokens": a.max_tokens,
                            "seed": a.seed, "temperature": 0, "reasoning_effort": a.reasoning_effort}
    if a.model:
        body["model"] = a.model
    data = json.dumps(body, ensure_ascii=False).encode("utf-8")
    headers = {"Content-Type": "application/json", "Accept": "text/event-stream"}
    if secret:
        headers["Authorization"] = "Bearer " + secret
    req = urllib.request.Request(base + "/v1/chat/completions", data=data, headers=headers, method="POST")
    start = time.perf_counter()
    record_start_ns = time.monotonic_ns()
    try:
        with safe_open(req, timeout=a.timeout) as response:
            record = _read_sse(response, secret, start)
    except (urllib.error.URLError, TimeoutError, OSError, RuntimeError) as e:
        elapsed = time.perf_counter() - start
        record = {"error_type": type(e).__name__, "error": redact(str(e), secret), "client_e2e_s": elapsed,
                  "status": "error"}
    if "status" not in record or record["status"] is None:
        finish = record.get("finish_reason")
        record["status"] = {"stop": "completed", "length": "truncated_by_length"}.get(finish, "incomplete_or_unknown_finish")
    record["request_start_monotonic_ns"] = record_start_ns
    record["request_end_monotonic_ns"] = time.monotonic_ns()
    record["reasoning_effort"] = a.reasoning_effort
    # Preserve exact raw model output separately; raw event JSON excludes headers and request bodies.
    stem = f"{label}-{index:04d}"
    content = record.pop("content", "")
    reasoning = record.pop("reasoning", "")
    (run_dir / f"{stem}.text.txt").write_bytes(content.encode("utf-8"))
    (run_dir / f"{stem}.reasoning.txt").write_bytes(reasoning.encode("utf-8"))
    record["content_sha256"] = hashlib.sha256(content.encode("utf-8")).hexdigest()
    record["reasoning_sha256"] = hashlib.sha256(reasoning.encode("utf-8")).hexdigest()
    record["output_artifact_encoding"] = "utf-8 bytes without newline translation"
    record["content_chars"] = len(content)
    record["reasoning_chars"] = len(reasoning)
    raw_events = record.pop("events", [])
    raw_lines = record.pop("raw_lines", [])
    (run_dir / f"{stem}.events.json").write_text(json.dumps(raw_events, ensure_ascii=False, indent=2), encoding="utf-8")
    (run_dir / f"{stem}.sse.txt").write_text("\n".join(raw_lines) + "\n", encoding="utf-8")
    return record


def percentile95(values: list[float]) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    return ordered[max(0, math.ceil(0.95 * len(ordered)) - 1)]


def fetch_live_identity(base: str, secret: str) -> dict[str, Any]:
    headers = {"Accept": "application/json"}
    if secret:
        headers["Authorization"] = "Bearer " + secret
    result: dict[str, Any] = {}
    for name, path in (("health", "/health"), ("models", "/v1/models"), ("props", "/props")):
        req = urllib.request.Request(base + path, headers=headers)
        try:
            with safe_open(req, timeout=10) as response:
                result[name] = json.loads(response.read().decode("utf-8-sig"))
        except Exception as e:
            result[name] = {"unavailable": type(e).__name__}
    return redact(result, secret)


def compare_identity(identity: dict[str, Any], live: dict[str, Any], supplied: bool) -> dict[str, Any]:
    if not supplied:
        return {"status": "unverified", "matches": {}, "unverifiable": ["schema", "source", "config", "model", "engine"]}
    health = live.get("health") or {}
    models = live.get("models") or {}
    props = live.get("props") or {}
    available = {"model": health.get("model"),
                 "engine": (props.get("build_info") or "").removeprefix("Strata ") or None}
    matches: dict[str, Any] = {}
    unverifiable: list[str] = []
    mismatches: list[str] = []
    for field in ("schema", "source", "config", "model", "engine"):
        expected = identity.get(field)
        actual = available.get(field)
        if actual is None:
            unverifiable.append(field)
            continue
        if isinstance(expected, dict):
            expected = expected.get("id", expected.get("name", expected.get("version")))
        if field == "model" and isinstance(models, dict):
            ids = [m.get("id") for m in models.get("data", []) if isinstance(m, dict)]
            equal = expected == actual and (not ids or expected in ids)
        else:
            equal = str(expected) == str(actual)
        matches[field] = equal
        if not equal:
            mismatches.append(field)
    status = "mismatch" if mismatches else "partial_match" if matches else "unverified"
    return {"status": status, "matches": matches, "mismatches": mismatches, "unverifiable": unverifiable,
            "live_model": health.get("model"), "live_engine_build_info": props.get("build_info")}


def _finite_positive(v: Any) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v) and v > 0


def summarize(records: list[dict[str, Any]], concurrency: int,
              batches: list[dict[str, Any]] | None = None,
              allow_capped: bool = False) -> dict[str, Any]:
    timing_states = {"completed", "truncated_by_length"} if allow_capped else {"completed"}
    good = [x for x in records if x.get("status") in timing_states]
    e2e = [x["client_e2e_s"] for x in good if _finite_positive(x.get("client_e2e_s"))]
    ttfg = [x["first_generated_s"] for x in good if _finite_positive(x.get("first_generated_s"))]
    ttfv = [x["first_visible_content_s"] for x in good if _finite_positive(x.get("first_visible_content_s"))]
    usage_counts = [(x.get("usage") or {}).get("completion_tokens") for x in good]
    valid_counts = [n for n in usage_counts if isinstance(n, int) and not isinstance(n, bool) and n > 0]
    rates = [(x.get("timings") or {}).get("predicted_per_second") for x in good]
    engine_rates = [n for n in rates if _finite_positive(n)]
    cache_per_request = [(x.get("timings") or {}).get("cache_n") for x in good]
    cache_valid = [n for n in cache_per_request if isinstance(n, int) and not isinstance(n, bool) and n >= 0]
    timed_cache = cache_valid
    batch_rows = batches or []
    batch_walls = [b.get("elapsed_s") for b in batch_rows if _finite_positive(b.get("elapsed_s"))]
    batch_tokens = [sum(((x.get("usage") or {}).get("completion_tokens")) for x in records
                        if x.get("batch_index") == b.get("batch_index") and x.get("status") in timing_states
                        and isinstance(((x.get("usage") or {}).get("completion_tokens")), int)
                        and not isinstance(((x.get("usage") or {}).get("completion_tokens")), bool)
                        and ((x.get("usage") or {}).get("completion_tokens")) > 0)
                    for b in batch_rows]
    batch_token_total = sum(batch_tokens) if batch_tokens else 0
    batch_elapsed_total = sum(batch_walls) if batch_walls else 0.0
    token_missing = sum(1 for x in good if not (isinstance(((x.get("usage") or {}).get("completion_tokens")), int)
                                                and not isinstance(((x.get("usage") or {}).get("completion_tokens")), bool)
                                                and ((x.get("usage") or {}).get("completion_tokens")) > 0))
    concurrent_rates = [n / b.get("elapsed_s") for n, b in zip(batch_tokens, batch_rows)
                        if n > 0 and _finite_positive(b.get("elapsed_s"))]
    prefill_rates = [(x.get("timings") or {}).get("prompt_per_second") for x in good
                     if (x.get("timings") or {}).get("cache_n") == 0]
    prefill_rates = [n for n in prefill_rates if _finite_positive(n)]
    return {"requests": len(records), "successful_requests": len(good), "concurrency": concurrency,
            "length_capped_requests_in_timing": sum(x.get("status") == "truncated_by_length" for x in good),
            "quality_pass_inferred_from_finish": False,
            "client_e2e_s_mean_success_only": statistics.mean(e2e) if e2e else None,
            "client_e2e_s_p95_success_only": percentile95(e2e),
            "first_generated_s_mean_includes_reasoning_success_only": statistics.mean(ttfg) if ttfg else None,
            "first_visible_content_s_mean_success_only": statistics.mean(ttfv) if ttfv else None,
            "completion_token_count_requests": len(valid_counts), "completion_token_count_missing_or_invalid": token_missing,
            "completion_tokens_total_matched_subset": sum(valid_counts) if valid_counts else None,
            "formal_batch_elapsed_s_total": batch_elapsed_total if batch_walls else None,
            "concurrent_batch_output_tok_s": (batch_token_total / batch_elapsed_total)
                                                if batch_token_total > 0 and batch_elapsed_total > 0 else None,
            "concurrent_batch_output_tok_s_note": "Known completion tokens from eligible requests divided by all formal batch wall time, including failed batches. A lower bound if counts are missing. Warmups excluded.",
            "requests_with_engine_timing_success_only": len(engine_rates),
            "requests_with_cache_reuse_success_only": sum(n > 0 for n in cache_valid),
            "engine_cache_reuse_tokens_total_success_only": sum(cache_valid) if cache_valid else None,
            "all_successful_requests_cache_clean": (all(n == 0 for n in timed_cache) if len(timed_cache) == len(good) and good else None),
            "mean_concurrent_batch_output_tok_s": statistics.mean(concurrent_rates) if concurrent_rates else None,
            "uncached_prompt_per_second_mean": statistics.mean(prefill_rates) if prefill_rates else None,
            "uncached_prompt_timing_requests": len(prefill_rates),
            "engine_predicted_per_second_mean_success_only": statistics.mean(engine_rates) if engine_rates else None,
            "engine_timing_note": "Uses server timings.predicted_per_second when finite and positive; Strata extension, not standard OpenAI metadata."}


def main(argv: list[str] | None = None) -> int:
    a = parse_args(argv)
    try:
        prompt_path, run_dir, identity, identity_supplied = validate_args(a)
        prompt = prompt_path.read_text(encoding="utf-8-sig")
        if not prompt.strip():
            raise ValueError("prompt file must contain non-whitespace text")
        base = validate_base_url(a.base_url)
    except Exception as e:
        print(f"validation failed: {e}")
        return 2
    if a.validate_only:
        print(json.dumps({"valid": True, "network": False, "run_directory": str(run_dir),
                          "identity_verification": "provided_unverified" if identity_supplied else "unverified"}, indent=2))
        return 0
    secret = os.environ.get("STRATA_HETERO_API_KEY", "")
    try:
        run_dir.mkdir(parents=True, exist_ok=False)
    except FileExistsError:
        print(f"run directory already exists: {run_dir}")
        return 2
    prompt_bytes = prompt_path.read_bytes()
    (run_dir / "prompt.txt").write_bytes(prompt_bytes)
    live = fetch_live_identity(base, secret)
    identity_check = compare_identity(identity, live, identity_supplied)
    metadata = {"run_id": run_dir.name, "created_utc": datetime.now(timezone.utc).isoformat(),
                "base_url": base, "endpoint": "/v1/chat/completions", "prompt_file": prompt_path.name,
                "prompt_sha256": hashlib.sha256(prompt_bytes).hexdigest(), "prompt_bytes": len(prompt_bytes),
                "repeats": a.repeats, "warmup": a.warmup, "max_tokens": a.max_tokens,
                "concurrency": a.concurrency, "seed": a.seed, "reasoning_effort": a.reasoning_effort,
                "length_cap_timing_allowed": a.allow_capped_performance,
                "model_requested": a.model or None,
                "identity": identity, "identity_verification": "provided_unverified" if identity_supplied else "unverified",
                "live_identity": live, "identity_comparison": identity_check,
                "acceptance_status": "not_accepted_pending_quality"}
    metadata["artifact_writer"] = {"client_os_name": os.name,
                                   "text_encoding": "utf-8 bytes without newline translation",
                                   "content_sha256_scope": "exact stored model-output bytes"}
    (run_dir / "metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    if identity_check["status"] == "mismatch":
        receipt = {"status": "identity_mismatch_preflight_failed", "benchmark_started": False,
                   "identity_comparison": identity_check, "live_identity": live}
        (run_dir / "preflight-receipt.json").write_text(json.dumps(receipt, indent=2, ensure_ascii=False), encoding="utf-8")
        print(json.dumps(receipt, indent=2, ensure_ascii=False))
        return 3
    warm_records: list[dict[str, Any]] = []
    formal_records: list[dict[str, Any]] = []
    warm_batches: list[dict[str, Any]] = []
    formal_batches: list[dict[str, Any]] = []

    def run_batch(label: str, batch_index: int) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        start_ns = time.monotonic_ns()
        start = time.perf_counter()
        with concurrent.futures.ThreadPoolExecutor(max_workers=a.concurrency) as pool:
            fs = [pool.submit(request_once, base, prompt, a, run_dir, label,
                              batch_index * a.concurrency + i, secret) for i in range(a.concurrency)]
            rows = [f.result() for f in fs]
        end = time.perf_counter()
        end_ns = time.monotonic_ns()
        for row in rows:
            row["batch_index"] = batch_index
        return rows, {"batch_index": batch_index, "label": label, "start_monotonic_ns": start_ns,
                      "end_monotonic_ns": end_ns, "elapsed_s": end - start}

    # Warmups are deliberately excluded from all formal summaries.
    for batch in range(a.warmup):
        rows, timing = run_batch("warmup", batch)
        warm_records.extend(rows)
        warm_batches.append(timing)
    for batch in range(a.repeats):
        rows, timing = run_batch("formal", batch)
        formal_records.extend(rows)
        formal_batches.append(timing)
    (run_dir / "warmups.json").write_text(json.dumps(warm_records, indent=2, ensure_ascii=False), encoding="utf-8")
    (run_dir / "requests.json").write_text(json.dumps(formal_records, indent=2, ensure_ascii=False), encoding="utf-8")
    (run_dir / "batches.json").write_text(json.dumps({"warmups": warm_batches, "formal": formal_batches}, indent=2), encoding="utf-8")
    summary = {"warmup_summary": summarize(warm_records, a.concurrency, warm_batches, a.allow_capped_performance),
               "formal_summary": summarize(formal_records, a.concurrency, formal_batches, a.allow_capped_performance),
               "formal_monotonic_start_ns": min((b["start_monotonic_ns"] for b in formal_batches), default=None),
               "formal_monotonic_end_ns": max((b["end_monotonic_ns"] for b in formal_batches), default=None),
               "formal_batch_elapsed_s_sum": sum(b["elapsed_s"] for b in formal_batches),
               "formal_quality_states": {s: sum(x.get("status") == s for x in formal_records)
                                         for s in sorted({x.get("status", "unknown") for x in formal_records})},
               "acceptance_status": "not_accepted_pending_quality",
               "acceptance_note": "Identity presence does not establish equivalence or quality; verify live config/model/engine and assess quality/cache conditions before accepting speedup."}
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"run_directory": str(run_dir), "acceptance_status": summary["acceptance_status"],
                      "formal_summary": summary["formal_summary"]}, indent=2))
    timing_states = {"completed", "truncated_by_length"} if a.allow_capped_performance else {"completed"}
    return 0 if all(x.get("status") in timing_states for x in formal_records) else 1


if __name__ == "__main__":
    raise SystemExit(main())
