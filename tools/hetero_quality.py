#!/usr/bin/env python3
"""Offline prompt preparation and bounded artifact-quality evaluation for Strata Hetero runs.

This tool never contacts a model endpoint, executes generated code, or changes process affinity.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import importlib.util
import json
import math
import random
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SEED = 42
CONTEXT = 32768
LONG_OUTPUT_BUDGET = 1024
LONG_TARGETS = (("long_1k", 1024), ("long_4k", 4096), ("long_16k", 16384), ("long_30k7", 30700))
KEY_NAMES = ("archive_key_a", "archive_key_b", "archive_key_c")
DEPTHS = (0.10, 0.50, 0.90)
MAX_DISCOVERED_FILES = 5000
DEFAULT_MAX_RESPONSE_BYTES = 1 << 20


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def strict_json(text: str) -> tuple[Any | None, str | None]:
    def reject_constant(value: str):
        raise ValueError(f"nonfinite JSON constant: {value}")
    try:
        return json.loads(text, parse_constant=reject_constant), None
    except ValueError as e:
        if "nonfinite JSON constant" in str(e):
            return None, "nonfinite_json_number"
        return None, "invalid_json"
    except json.JSONDecodeError:
        return None, "invalid_json"


def has_nonfinite_json_number(value: Any) -> bool:
    if isinstance(value, float):
        return not math.isfinite(value)
    if isinstance(value, dict):
        return any(has_nonfinite_json_number(v) for v in value.values())
    if isinstance(value, list):
        return any(has_nonfinite_json_number(v) for v in value)
    return False


def normalize_prompt_id(value: str) -> str:
    name = Path(value).name
    for suffix in (".text.txt", ".prompt.txt", ".txt"):
        if name.endswith(suffix):
            name = name[:-len(suffix)]
            break
    return name


def load_strata_tokenizer(tokenizer_module: Path, tokenizer_dir: Path):
    spec = importlib.util.spec_from_file_location("hetero_strata_tokenizer", tokenizer_module)
    if spec is None or spec.loader is None:
        raise ValueError(f"cannot load tokenizer module: {tokenizer_module}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    cfg = json.loads((tokenizer_dir / "tokenizer.json").read_text(encoding="utf-8"))
    vocab = json.loads((tokenizer_dir / "vocab.json").read_text(encoding="utf-8"))
    if not isinstance(vocab, dict) or not vocab:
        raise ValueError("tokenizer vocab.json must be a non-empty token-to-id object")
    tokens: list[str | None] = [None] * len(vocab)
    for token, token_id in vocab.items():
        if not isinstance(token_id, int) or not 0 <= token_id < len(tokens) or tokens[token_id] is not None:
            raise ValueError("tokenizer vocab ids are not unique and contiguous")
        tokens[token_id] = token
    if any(token is None for token in tokens):
        raise ValueError("tokenizer vocab ids contain gaps")
    merges = (tokenizer_dir / "merges.txt").read_text(encoding="utf-8").splitlines()
    token_types = json.loads((tokenizer_dir / "token_type.json").read_text(encoding="utf-8"))
    tokenizer = module.Tokenizer(tokens, merges, token_types, cfg.get("pre", "qwen35"), cfg.get("special_ids", {}))
    return tokenizer, cfg


def render_chat_message(body: str, template_path: Path) -> str:
    from jinja2 import Environment, StrictUndefined
    template_text = template_path.read_text(encoding="utf-8")
    env = Environment(undefined=StrictUndefined, autoescape=False, keep_trailing_newline=False)
    env.globals["raise_exception"] = lambda message: (_ for _ in ()).throw(ValueError(str(message)))
    env.filters["tojson"] = lambda value: json.dumps(value, ensure_ascii=False)
    template = env.from_string(template_text)
    return template.render(messages=[{"role": "user", "content": body}], tools=None,
                           add_generation_prompt=True, add_vision_id=False,
                           enable_thinking=False, reasoning_effort="none",
                           preserve_thinking=False)


def token_counts(tokenizer, body: str, template_path: Path) -> tuple[int, int, int]:
    body_count = len(tokenizer.encode(body))
    rendered = render_chat_message(body, template_path)
    chat_count = len(tokenizer.encode(rendered, parse_special=True))
    return body_count, chat_count, chat_count - body_count


def make_markers(corpus: str, seed: int = SEED) -> list[dict[str, str]]:
    rng = random.Random(seed)
    result = []
    for index, key_name in enumerate(KEY_NAMES):
        while True:
            key = f"ARCHIVE-{rng.getrandbits(40):010X}"
            value = f"{rng.randrange(100_000_000, 999_999_999)}"
            if key not in corpus and value not in corpus and all(key != row["key"] and value != row["value"] for row in result):
                break
        result.append({"name": key_name, "key": key, "value": value,
                       "marker": f"Checkpoint key {key} has value {value}."})
    return result


def _trim_at_word(text: str, chars: int) -> str:
    if chars >= len(text):
        return text
    if chars <= 0:
        return ""
    return text[:chars].rsplit(" ", 1)[0].rstrip()


def build_long_body(paragraphs: list[str], selected_count: int, last_chars: int,
                    markers: list[dict[str, str]], tokenizer, header: str, suffix: str) -> tuple[str, list[dict[str, Any]]]:
    selected = list(paragraphs[:selected_count])
    if not selected:
        raise ValueError("natural corpus yielded no paragraphs")
    selected[-1] = _trim_at_word(selected[-1], last_chars)
    selected = [p for p in selected if p]
    estimated = [len(tokenizer.encode(p)) for p in selected]
    separator_tokens = len(tokenizer.encode("\n\n"))
    natural_total = sum(estimated) + max(0, len(selected) - 1) * separator_tokens
    marker_sizes = [len(tokenizer.encode(m["marker"])) for m in markers]
    header_size = len(tokenizer.encode(header.strip()))
    suffix_size = len(tokenizer.encode(suffix.strip()))
    full_estimate = (header_size + suffix_size + natural_total + sum(marker_sizes)
                     + (len(selected) + len(markers) + 1) * separator_tokens)
    insertions: dict[int, list[dict[str, str]]] = {}
    last_index = -1
    prior_marker_overhead = 0
    for marker, depth, marker_size in zip(markers, DEPTHS, marker_sizes):
        cumulative = 0
        threshold = depth * full_estimate - header_size - separator_tokens - prior_marker_overhead
        threshold = max(0, min(natural_total, threshold))
        idx = len(selected) - 1
        for i, amount in enumerate(estimated):
            cumulative += amount + (separator_tokens if i else 0)
            if cumulative >= threshold:
                idx = i
                break
        idx = max(last_index + 1, idx)
        idx = min(idx, len(selected) - 1)
        insertions.setdefault(idx, []).append(marker)
        last_index = idx
        prior_marker_overhead += marker_size + separator_tokens
    parts = [header.strip()]
    positions: list[dict[str, Any]] = []
    for i, paragraph in enumerate(selected):
        for marker in insertions.get(i, []):
            parts.append(marker["marker"])
        parts.append(paragraph)
    # Ending query is included exactly once after the non-repeated natural-text prefix.
    parts.append(suffix.strip())
    body = "\n\n".join(parts)
    # Report actual positions from the final body, not the approximate paragraph selection points.
    for depth, marker in zip(DEPTHS, markers):
        offset = body.index(marker["marker"])
        position = len(tokenizer.encode(body[:offset]))
        positions.append({"name": marker["name"], "key": marker["key"], "value": marker["value"],
                          "target_relative_depth": depth, "token_position_body_prefix": position})
    return body, positions


def size_long_body(paragraphs: list[str], target_tokens: int, markers: list[dict[str, str]], tokenizer,
                   header: str, suffix: str) -> tuple[str, list[dict[str, Any]], int, int]:
    if target_tokens <= 0:
        raise ValueError("target token count must be positive")
    header_count = len(tokenizer.encode(header.strip()))
    suffix_count = len(tokenizer.encode(suffix.strip()))
    marker_count = sum(len(tokenizer.encode(m["marker"])) + 2 for m in markers)
    budget = max(1, target_tokens - header_count - suffix_count - marker_count)
    selected_count = 0
    estimated = 0
    while selected_count < len(paragraphs) and estimated < budget:
        paragraph = paragraphs[selected_count]
        estimated += len(tokenizer.encode(paragraph)) + len(tokenizer.encode("\n\n"))
        selected_count += 1
    if selected_count == 0:
        raise ValueError("not enough corpus text to size prompt")
    # Paragraph token estimates only choose a starting point. Measure the full rendered body and keep adding
    # distinct source paragraphs until the exact count reaches the requested neighborhood.
    while True:
        body, positions = build_long_body(paragraphs, selected_count, len(paragraphs[selected_count - 1]),
                                         markers, tokenizer, header, suffix)
        count = len(tokenizer.encode(body))
        if count >= target_tokens - 64 or selected_count >= len(paragraphs):
            break
        selected_count += 1
    # If the full last paragraph crosses the target, trim only that final source paragraph by binary search.
    if count > target_tokens:
        lo, hi = 0, len(paragraphs[selected_count - 1])
        best = (count, body, positions)
        while lo <= hi:
            mid = (lo + hi) // 2
            candidate, candidate_positions = build_long_body(paragraphs, selected_count, mid, markers,
                                                               tokenizer, header, suffix)
            candidate_count = len(tokenizer.encode(candidate))
            if abs(candidate_count - target_tokens) < abs(best[0] - target_tokens):
                best = (candidate_count, candidate, candidate_positions)
            if candidate_count == target_tokens:
                best = (candidate_count, candidate, candidate_positions)
                break
            if candidate_count < target_tokens:
                lo = mid + 1
            else:
                hi = mid - 1
        count, body, positions = best
    return body, positions, count, selected_count


def smoke_specs(markers: list[dict[str, str]]) -> list[dict[str, Any]]:
    retrieval = "\n".join(m["marker"] for m in markers)
    return [
        {"id": "smoke_arithmetic", "body": "What is 2 + 2? Reply with only the exact answer.",
         "quality": {"kind": "exact_text", "expected": "4"}},
        {"id": "smoke_unicode", "body": "Copy exactly this Unicode text and output nothing else: 你好，世界！🌏 e\u0301",
         "quality": {"kind": "exact_text", "expected": "你好，世界！🌏 e\u0301"}},
        {"id": "smoke_json_arithmetic", "body": "Compute 12 * 13. Return only JSON with numeric field result and string field expression, e.g. keys result and expression. Use expression 12*13.",
         "quality": {"kind": "json_arithmetic", "expected_result": 156, "expected_expression": "12*13"}},
        {"id": "smoke_python_function", "body": "Write only a Python function named add_two that takes parameter x and returns x + 2. Do not include prose or execute anything.",
         "quality": {"kind": "python_ast", "function": "add_two", "arguments": ["x"],
                     "return_expression": {"left_name": "x", "operator": "Add", "right_constant": 2}}},
        {"id": "smoke_three_key_retrieval", "body": "Read these three records once and return only JSON mapping each exact key to its exact string value:\n" + retrieval,
         "quality": {"kind": "json_retrieval", "expected": {m["key"]: m["value"] for m in markers}}},
    ]


def _file_provenance(path: Path) -> dict[str, Any]:
    raw = path.read_bytes()
    return {"path": str(path.resolve()), "bytes": len(raw), "sha256": sha256_bytes(raw)}


def prepare_prompt_set(output_dir: Path, corpus_path: Path, source_manifest_path: Path,
                       tokenizer_module: Path, tokenizer_dir: Path) -> dict[str, Any]:
    output_dir = output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(f"output directory already exists: {output_dir}")
    corpus_bytes = corpus_path.read_bytes()
    source_manifest_bytes = source_manifest_path.read_bytes()
    source_manifest = json.loads(source_manifest_bytes.decode("utf-8"))
    corpus_hash = sha256_bytes(corpus_bytes)
    if source_manifest.get("sha256") != corpus_hash:
        raise ValueError("historical natural corpus hash differs from its source manifest")
    corpus = corpus_bytes.decode("utf-8")
    paragraphs = [p.strip() for p in corpus.split("\n\n") if p.strip()]
    if len(paragraphs) < 100:
        raise ValueError("historical corpus is unexpectedly short or not paragraph separated")
    markers = make_markers(corpus, SEED)
    tokenizer, tokenizer_cfg = load_strata_tokenizer(tokenizer_module, tokenizer_dir)
    template_path = tokenizer_dir / "chat_template.jinja"
    if not template_path.is_file():
        raise FileNotFoundError(f"tokenizer chat template is missing: {template_path}")
    specs = smoke_specs(markers)
    header = ("Read this natural-text excerpt once. Three checkpoint sentences occur once in the prose. "
              "Remember each exact key and its string value.")
    suffix = ("Return exactly one JSON object with these three keys and their string values: "
              + ", ".join(f'"{m["key"]}"' for m in markers) + ". Do not add prose.")
    prompts_dir = output_dir / "prompts"
    prompts_dir.mkdir(parents=True, exist_ok=False)
    entries: list[dict[str, Any]] = []
    for spec in specs:
        body = spec["body"]
        body_count, chat_count, delta = token_counts(tokenizer, body, template_path)
        filename = f"{spec['id']}.prompt.txt"
        raw = body.encode("utf-8")
        (prompts_dir / filename).write_bytes(raw)
        entries.append({"id": spec["id"], "file": f"prompts/{filename}", "kind": "smoke",
                        "bytes": len(raw), "sha256": sha256_bytes(raw), "body_tokens_local": body_count,
                        "chat_prompt_tokens_local": chat_count, "chat_wrapper_delta_local": delta,
                        "max_output_tokens": 128, "quality": spec["quality"]})
    long_marker_positions: dict[str, Any] = {}
    for prompt_id, target in LONG_TARGETS:
        body, positions, measured_body, selected_count = size_long_body(paragraphs, target, markers, tokenizer,
                                                                         header, suffix)
        body_count, chat_count, delta = token_counts(tokenizer, body, template_path)
        if body_count != measured_body:
            raise AssertionError("prompt sizing count is inconsistent")
        if chat_count + LONG_OUTPUT_BUDGET > CONTEXT:
            raise ValueError(f"{prompt_id} leaves fewer than {LONG_OUTPUT_BUDGET} chat tokens for output")
        raw = body.encode("utf-8")
        filename = f"{prompt_id}.prompt.txt"
        (prompts_dir / filename).write_bytes(raw)
        for pos in positions:
            pos["actual_relative_depth_in_body"] = round(pos["token_position_body_prefix"] / body_count, 6)
        long_marker_positions[prompt_id] = positions
        entries.append({"id": prompt_id, "file": f"prompts/{filename}", "kind": "long_natural_text",
                        "target_body_tokens": target, "target_deviation_tokens": body_count - target,
                        "bytes": len(raw), "sha256": sha256_bytes(raw), "body_tokens_local": body_count,
                        "chat_prompt_tokens_local": chat_count, "chat_wrapper_delta_local": delta,
                        "max_output_tokens": LONG_OUTPUT_BUDGET, "configured_context": CONTEXT,
                        "remaining_context_tokens_after_output_budget": CONTEXT - chat_count - LONG_OUTPUT_BUDGET,
                        "selected_paragraphs_from_natural_prefix": selected_count,
                        "corpus_repetition": "none; each selected source paragraph appears once",
                        "needle_positions": positions,
                        "quality": {"kind": "json_retrieval", "expected": {m["key"]: m["value"] for m in markers}}})
    tokenizer_files = [tokenizer_dir / n for n in ("tokenizer.json", "vocab.json", "merges.txt", "token_type.json", "chat_template.jinja")]
    manifest = {"schema": "strata-hetero-quality-prompts-v1", "created_utc": datetime.now(timezone.utc).isoformat(),
                "tokenizer": {"module": str(tokenizer_module.resolve()),
                              "implementation_source": _file_provenance(tokenizer_module),
                              "assets": [_file_provenance(p) for p in tokenizer_files],
                              "pre": tokenizer_cfg.get("pre"), "vocab_size": tokenizer_cfg.get("vocab_size"),
                              "count_method": "tools/strata_tokenizer.py Tokenizer.encode; exact local count, no server count"},
                "chat_count_method": {"template": str(template_path.resolve()),
                                      "render_settings": {"messages": "single user message", "add_generation_prompt": True,
                                                          "enable_thinking": False, "reasoning_effort": "none"},
                                      "description": "Tokenizer count after rendering the pack's chat_template.jinja, including chat tokens; compare with later live API count."},
                "generation_request_defaults": {"temperature": 0.0, "top_p": 1.0, "seed": SEED,
                                                 "reasoning_effort": "none", "note": "intended later API-run settings; not exercised during preparation"},
                "context": CONTEXT, "max_output_tokens_long": LONG_OUTPUT_BUDGET,
                "marker_generation": {"seed": SEED, "algorithm": "random.Random(seed) 40-bit uppercase key plus 9-digit decimal value; reroll if present in source corpus",
                                      "records": markers},
                "corpus_provenance": {"corpus_id": source_manifest.get("corpus_id"),
                                      "historical_directory": str(corpus_path.parent.resolve()),
                                      "natural_corpus": _file_provenance(corpus_path),
                                      "source_manifest": _file_provenance(source_manifest_path),
                                      "normalization": source_manifest.get("normalization"),
                                      "source_titles_and_urls": [{"title": x.get("title"), "url": x.get("url"),
                                                                   "filename": x.get("filename"), "bytes": x.get("bytes"),
                                                                   "sha256": x.get("sha256")} for x in source_manifest.get("sources", [])],
                                      "source_file_downloads_performed": False},
                "prompts": entries,
                "limitations": ["Local tokenizer counts are not server-reported API counts; main should validate live counts later.",
                                "Canonical re-encoding of a response cannot recover actual emitted token IDs.",
                                "No model responses, quality results, or performance results are included in this preparation manifest."]}
    (output_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return manifest


def _has_bang_loop(text: str) -> bool:
    compact = "".join(text.split())
    # A request may begin correctly before degenerating. Detect its long bang
    # suffix before the generic cap/timeout classification masks the failure.
    return (len(compact) >= 8 and set(compact) == {"!"}) or compact.endswith("!" * 32)


def _eval_python_ast(text: str, expected: dict[str, Any]) -> dict[str, Any]:
    candidate = text.strip()
    fence = re.fullmatch(r"```(?:python)?\s*(.*?)\s*```", candidate, flags=re.I | re.S)
    if fence:
        candidate = fence.group(1)
    try:
        tree = ast.parse(candidate, mode="exec")
    except (SyntaxError, ValueError) as e:
        return {"status": "invalid_python_syntax", "detail": str(e)}
    funcs = [n for n in tree.body if isinstance(n, ast.FunctionDef)
             and n.name == expected.get("function")]
    if len(funcs) != 1:
        return {"status": "expected_function_missing_or_duplicated", "function": expected.get("function")}
    if len(tree.body) != 1:
        return {"status": "unexpected_module_statements", "execution": "not_run"}
    args = [a.arg for a in funcs[0].args.args]
    signature = funcs[0].args
    if (args != expected.get("arguments", []) or signature.posonlyargs or signature.kwonlyargs
            or signature.vararg is not None or signature.kwarg is not None or signature.defaults):
        return {"status": "function_signature_mismatch", "arguments": args}
    if funcs[0].decorator_list or len(funcs[0].body) != 1 or not isinstance(funcs[0].body[0], ast.Return):
        return {"status": "function_body_mismatch", "execution": "not_run"}
    expr = funcs[0].body[0].value
    want = expected.get("return_expression", {})
    valid_expr = (isinstance(expr, ast.BinOp) and type(expr.op).__name__ == want.get("operator")
                  and isinstance(expr.left, ast.Name) and expr.left.id == want.get("left_name")
                  and isinstance(expr.right, ast.Constant) and type(expr.right.value) in (int, float)
                  and expr.right.value == want.get("right_constant"))
    if not valid_expr:
        return {"status": "function_return_expression_mismatch", "execution": "not_run"}
    return {"status": "ast_valid", "function": funcs[0].name, "arguments": args,
            "execution": "not_run; generated code is parsed only"}


def evaluate_text(prompt: dict[str, Any], text: str, run_status: str = "completed", tokenizer=None) -> dict[str, Any]:
    common = {"prompt_id": prompt.get("id"), "run_status": run_status,
              "response_bytes_utf8": len(text.encode("utf-8")),
              "canonical_visible_text_reencoded_tokens": (len(tokenizer.encode(text))
                                                           if tokenizer is not None and run_status == "completed" else None),
              "actual_emitted_token_ids": None,
              "canonical_reencoding_note": "optional count re-encodes saved visible text only; no claim of agreement with actual emitted token IDs"}
    if _has_bang_loop(text):
        return {**common, "quality_status": "generation_degenerate_bang_loop",
                "generation_stability": "fail_single_character_bang_loop"}
    if run_status != "completed":
        return {**common, "quality_status": "not_assessed_incomplete_run", "generation_stability": "not_assessed"}
    expected = prompt.get("quality", {})
    kind = expected.get("kind")
    if kind == "exact_text":
        passed = text.strip() == expected.get("expected")
        return {**common, "quality_status": "pass" if passed else "exact_text_mismatch", "exact_match": passed}
    if kind == "json_arithmetic":
        parsed, error = strict_json(text.strip())
        if error:
            return {**common, "quality_status": error, "nonfinite_json_numeric_check": "fail" if error == "nonfinite_json_number" else "not_evaluated"}
        if has_nonfinite_json_number(parsed):
            return {**common, "quality_status": "nonfinite_json_number", "nonfinite_json_numeric_check": "fail",
                    "result_is_json_number": False}
        result = parsed.get("result") if isinstance(parsed, dict) else None
        expression = parsed.get("expression") if isinstance(parsed, dict) else None
        numeric = type(result) in (int, float) and (not isinstance(result, float) or math.isfinite(result))
        passed = numeric and result == expected.get("expected_result") and expression == expected.get("expected_expression")
        return {**common, "quality_status": "pass" if passed else "json_arithmetic_mismatch",
                "nonfinite_json_numeric_check": "pass" if numeric else "fail",
                "result_is_json_number": numeric, "result_matches_expected": bool(numeric and result == expected.get("expected_result")),
                "expression_matches_expected": expression == expected.get("expected_expression")}
    if kind == "json_retrieval":
        parsed, error = strict_json(text.strip())
        if error:
            return {**common, "quality_status": error, "retrieval_status": "not_evaluated",
                    "nonfinite_json_numeric_check": "fail" if error == "nonfinite_json_number" else "not_evaluated"}
        wanted = expected.get("expected", {})
        actual = parsed if isinstance(parsed, dict) else {}
        found = {k: actual.get(k) == v for k, v in wanted.items()}
        passed = isinstance(parsed, dict) and set(actual) == set(wanted) and len(found) == len(wanted) and all(found.values())
        extra = sorted(set(actual) - set(wanted))
        return {**common, "quality_status": "pass" if passed else "retrieval_mismatch",
                "retrieval_status": "all_exact" if passed else "some_or_all_missing_or_wrong",
                "exact_values_found": found, "extra_keys": extra,
                "nonfinite_json_numeric_check": "fail" if has_nonfinite_json_number(parsed) else "pass"}
    if kind == "python_ast":
        result = _eval_python_ast(text, expected)
        passed = result.get("status") == "ast_valid"
        return {**common, **result, "quality_status": "pass" if passed else result["status"]}
    return {**common, "quality_status": "unsupported_quality_kind", "kind": kind}


def _read_bounded(path: Path, max_bytes: int) -> str:
    if path.stat().st_size > max_bytes:
        raise ValueError(f"response file exceeds byte limit: {path.name}")
    return path.read_text(encoding="utf-8-sig")


def collect_responses(responses_dir: Path, prompts: dict[str, dict[str, Any]], max_bytes: int, tokenizer=None) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    files = []
    for path in responses_dir.rglob("*"):
        if path.is_file():
            files.append(path)
            if len(files) > MAX_DISCOVERED_FILES:
                break
    if len(files) > MAX_DISCOVERED_FILES:
        raise ValueError("too many files under responses directory")
    metadata_files = [p for p in files if p.name == "metadata.json"]
    seen_files: set[Path] = set()
    for metadata_path in metadata_files:
        try:
            meta = json.loads(_read_bounded(metadata_path, 1 << 20))
        except (OSError, ValueError, json.JSONDecodeError):
            continue
        prompt_id = normalize_prompt_id(str(meta.get("prompt_file", "")))
        if prompt_id not in prompts:
            continue
        requests_path = metadata_path.parent / "requests.json"
        if not requests_path.is_file():
            continue
        requests = json.loads(_read_bounded(requests_path, 8 << 20))
        if not isinstance(requests, list):
            continue
        for index, request in enumerate(requests):
            if not isinstance(request, dict):
                continue
            text_path = metadata_path.parent / f"formal-{index:04d}.text.txt"
            if not text_path.is_file() or text_path in seen_files:
                continue
            seen_files.add(text_path)
            try:
                text = _read_bounded(text_path, max_bytes)
            except (OSError, ValueError) as e:
                out.append({"prompt_id": prompt_id, "artifact": str(text_path), "quality_status": "response_file_unavailable",
                            "detail": str(e), "run_status": request.get("status")})
                continue
            out.append({"artifact": str(text_path), "response_text_sha256": sha256_bytes(text_path.read_bytes()),
                        **evaluate_text(prompts[prompt_id], text,
                                        str(request.get("status", "unknown")), tokenizer)})
    # Direct fixtures are useful before a benchmark run: <prompt-id>.txt or <prompt-id>.text.txt.
    for path in files:
        if path in seen_files:
            continue
        prompt_id = normalize_prompt_id(path.name)
        if prompt_id not in prompts or path.suffix.lower() != ".txt":
            continue
        try:
            text = _read_bounded(path, max_bytes)
        except (OSError, ValueError) as e:
            out.append({"prompt_id": prompt_id, "artifact": str(path), "quality_status": "response_file_unavailable", "detail": str(e)})
            continue
        out.append({"artifact": str(path), "response_text_sha256": sha256_bytes(path.read_bytes()),
                    **evaluate_text(prompts[prompt_id], text, tokenizer=tokenizer)})
    return out


def load_manifest(path: Path) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("schema") != "strata-hetero-quality-prompts-v1":
        raise ValueError("unsupported prompt manifest schema")
    by_id = {p["id"]: p for p in data.get("prompts", [])}
    if not by_id:
        raise ValueError("prompt manifest has no prompt records")
    return data, by_id


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    modes = ap.add_mutually_exclusive_group(required=True)
    modes.add_argument("--prepare", action="store_true", help="generate prompts/manifest from a local historical corpus")
    modes.add_argument("--responses-dir", help="evaluate existing bounded text artifacts; no API calls")
    ap.add_argument("--output-dir", help="exclusive preparation directory")
    ap.add_argument("--corpus")
    ap.add_argument("--source-manifest")
    ap.add_argument("--tokenizer-module", default="tools/strata_tokenizer.py")
    ap.add_argument("--tokenizer-dir")
    ap.add_argument("--manifest", help="prompt manifest path for response evaluation")
    ap.add_argument("--report-out", help="write a new report JSON, refusing to overwrite")
    ap.add_argument("--max-response-bytes", type=int, default=DEFAULT_MAX_RESPONSE_BYTES)
    args = ap.parse_args(argv)
    try:
        if args.prepare:
            if not all((args.output_dir, args.corpus, args.source_manifest, args.tokenizer_dir)):
                ap.error("--prepare requires --output-dir, --corpus, --source-manifest, and --tokenizer-dir")
            manifest = prepare_prompt_set(Path(args.output_dir), Path(args.corpus), Path(args.source_manifest),
                                          Path(args.tokenizer_module), Path(args.tokenizer_dir))
            print(json.dumps({"output_dir": str(Path(args.output_dir).resolve()),
                              "prompt_count": len(manifest["prompts"]),
                              "body_tokens": {p["id"]: p["body_tokens_local"] for p in manifest["prompts"]}},
                             ensure_ascii=False, indent=2))
            return 0
        if not args.manifest or args.max_response_bytes <= 0:
            ap.error("--responses-dir requires --manifest and positive --max-response-bytes")
        manifest, prompts = load_manifest(Path(args.manifest))
        response_tokenizer = None
        if args.tokenizer_dir:
            response_tokenizer, _ = load_strata_tokenizer(Path(args.tokenizer_module), Path(args.tokenizer_dir))
        responses = collect_responses(Path(args.responses_dir), prompts, args.max_response_bytes, response_tokenizer)
        report = {"schema": "strata-hetero-quality-report-v1", "created_utc": datetime.now(timezone.utc).isoformat(),
                  "prompt_manifest": str(Path(args.manifest).resolve()), "responses_dir": str(Path(args.responses_dir).resolve()),
                  "response_count": len(responses), "results": responses,
                  "canonical_reencoded_output_token_agreement": {"status": "actual_ids_unavailable",
                      "canonical_visible_text_counts": "per-result count only when tokenizer is supplied",
                      "actual_emitted_token_ids": "unavailable in saved response text; do not infer or claim agreement"}}
        raw = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
        if args.report_out:
            out_path = Path(args.report_out)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            with out_path.open("x", encoding="utf-8", newline="\n") as f:
                f.write(raw)
        else:
            print(raw, end="")
        return 0
    except Exception as e:
        print(json.dumps({"error_type": type(e).__name__, "error": str(e)}, ensure_ascii=False))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
