from __future__ import annotations

import base64
import copy
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
from typing import Any


REPO = Path(__file__).resolve().parents[4]
RUN = REPO / "bench/hetero/20261011-69-prefill-first-nonfinite"
RUN66 = REPO / "bench/hetero/20261011-66-v042-quality-pair"
RUN66_SIDE_ID = "hetero42"
RUN66_RUN_ID = "20261011-66-hetero42-quality"
RUN_ID = "20261011-69-probe69-quality"
SIDE_ID = "probe69"
SOURCE = Path(r"E:\Strata-Hetero-data\source\hetero-prefill-probe-69")
SOURCE_HEAD = "1eb85d2a5d27cb2f054a9c757c3b074d7d3d7c86"
BUILD_AUTH = RUN / "preparation" / "root-build-authorization.json"
SHARED_BRIDGE = Path(r"E:\Strata-Hetero-data\source\hetero-0-1-42\serve\server.py")
SHARED_BRIDGE_SHA = "9bc794cae42317dd3283c5f41d98a5c42b4053fd5c8aae8504669ade44a63ded"
MANIFEST_IDS = (
    "smoke_arithmetic", "smoke_unicode", "smoke_json_arithmetic", "smoke_python_function",
    "smoke_three_key_retrieval", "long_1k", "long_4k", "long_16k", "long_30k7",
)
PROMPT_IDS = MANIFEST_IDS[:6]
PROBE_STAGES = {"input_residual", "ple_residual", "hc_mixed", "gdn_output", "qsa_output", "moe_output", "residual_write"}
DIAGNOSTIC_ENV = {
    "STRATA_PREFILL_FINITE_PROBE": "1",
    "STRATA_DIAG_ZERO_LOGITS": "1",
}
RUN66_QUALITY = RUN66 / "rendered-scripts/hetero42/quality.ps1"


def sha_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def sha_file(path: Path) -> str:
    return sha_bytes(path.read_bytes())


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def write_new(path: Path, raw: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())


def _json_bytes(value: Any) -> bytes:
    return (json.dumps(value, indent=2, ensure_ascii=False) + "\n").encode("utf-8")


def split_complete_log_bytes(snapshot: bytes, cursor: int) -> tuple[list[bytes], int, bytes]:
    """Return LF-terminated records and advance only through the last complete line."""
    if type(cursor) is not int or cursor < 0 or cursor > len(snapshot):
        raise ValueError("diagnostic log cursor is invalid or the log was truncated")
    tail = snapshot[cursor:]
    last_lf = tail.rfind(b"\n")
    if last_lf < 0:
        return [], cursor, tail
    complete = tail[:last_lf + 1]
    lines = complete.split(b"\n")[:-1]
    return [line[:-1] if line.endswith(b"\r") else line for line in lines], cursor + len(complete), tail[len(complete):]


def _required_number(record: dict[str, Any], name: str, *, positive: bool = False) -> bool:
    value = record.get(name)
    if type(value) not in (int, float):
        return False
    return value > 0 if positive else value >= 0


def _required_integer(record: dict[str, Any], name: str, *, positive: bool = False) -> bool:
    value = record.get(name)
    if type(value) is not int:
        return False
    return value > 0 if positive else value >= 0


def classify_diagnostic_line(line: str) -> dict[str, Any] | None:
    """Classify only known probe lines; malformed diagnostic-looking lines fail closed."""
    zero_prefix = "strata zero-logits diag:"
    probe_prefix = "strata pf_nonfinite: "
    if line.startswith(zero_prefix):
        return {"source": "zero_logits", "classification": "fatal_zero_logits", "raw_line": line}
    if line.startswith("strata zero-logits"):
        return {"source": "zero_logits", "classification": "fatal_invalid", "raw_line": line,
                "error": "unrecognized zero-logits diagnostic format"}
    if line.startswith(probe_prefix):
        payload = line[len(probe_prefix):]
        try:
            record = json.loads(payload)
        except (json.JSONDecodeError, TypeError) as exc:
            return {"source": "prefill_finite_probe", "classification": "fatal_invalid",
                    "raw_line": line, "error": f"invalid JSON: {exc}"}
        if not isinstance(record, dict) or type(record.get("schema")) is not int or record["schema"] != 1:
            return {"source": "prefill_finite_probe", "classification": "fatal_invalid",
                    "raw_line": line, "parsed": record, "error": "unsupported probe record/schema"}
        event = record.get("event")
        if event == "begin":
            valid = (_required_integer(record, "run") and _required_integer(record, "device") and
                     _required_integer(record, "pos0") and _required_integer(record, "tokens", positive=True) and
                     record.get("diagnostic_only") is True)
            if valid:
                return {"source": "prefill_finite_probe", "classification": "normal_begin",
                        "raw_line": line, "parsed": record}
        elif event == "checkpoint":
            valid = all(_required_integer(record, key) for key in (
                "run", "device", "completed_chunk", "completed_pos0", "completed_rows", "scans"))
            if valid and "first_bad" in record and record["first_bad"] is None:
                return {"source": "prefill_finite_probe", "classification": "normal_checkpoint",
                        "raw_line": line, "parsed": record}
            bad = record.get("first_bad")
            if valid and isinstance(bad, dict):
                fields_ok = (isinstance(bad.get("stage"), str) and bad.get("stage") in PROBE_STAGES and
                             _required_integer(bad, "ordinal", positive=True) and
                             all(_required_integer(bad, key, positive=(key in ("rows", "width"))) for key in (
                                 "ordinal", "chunk", "pos0", "rows", "width", "index", "row",
                                 "column", "nan", "positive_inf", "negative_inf")) and
                             type(bad.get("layer")) is int and type(bad.get("half")) is int and
                             bad["nan"] + bad["positive_inf"] + bad["negative_inf"] > 0 and
                             bad["rows"] > 0 and bad["width"] > 0)
                if fields_ok:
                    return {"source": "prefill_finite_probe", "classification": "fatal_nonfinite",
                            "raw_line": line, "parsed": record}
        return {"source": "prefill_finite_probe", "classification": "fatal_invalid",
                "raw_line": line, "parsed": record, "error": "unknown event or schema fields/types"}
    if line.startswith("strata pf_nonfinite"):
        return {"source": "prefill_finite_probe", "classification": "fatal_invalid",
                "raw_line": line, "error": "probe prefix is malformed"}
    return None


def classify_complete_lines(lines: list[bytes]) -> list[dict[str, Any]]:
    records = []
    for raw in lines:
        try:
            text = raw.decode("utf-8", errors="strict")
        except UnicodeDecodeError as exc:
            records.append({"source": "diagnostic_reader", "classification": "fatal_invalid",
                            "raw_base64": base64.b64encode(raw).decode("ascii"),
                            "error": f"invalid UTF-8: {exc}"})
            continue
        record = classify_diagnostic_line(text)
        if record is not None:
            records.append(record)
    return records


def diagnostic_is_fatal(records: list[dict[str, Any]]) -> bool:
    return any(str(record.get("classification", "")).startswith("fatal_") for record in records)


def require_prompt_prefix(prompt_ids: list[str] | tuple[str, ...]) -> tuple[str, ...]:
    if tuple(prompt_ids) != MANIFEST_IDS:
        raise ValueError("pinned nine-prompt manifest order changed")
    return PROMPT_IDS


def exclusive_json(path: Path, value: Any) -> None:
    write_new(path, _json_bytes(value))


def validate_probe_build_controls(build: dict[str, Any]) -> None:
    """Require the diagnostic option in both the CMake cache and exact configure argv."""
    cache = build.get("cmake_cache_controls")
    args = build.get("configure_argv")
    if not isinstance(cache, dict) or cache.get("STRATA_PREFILL_FINITE_PROBE") != "ON":
        raise ValueError("successful Run69 build receipt cache does not bind STRATA_PREFILL_FINITE_PROBE=ON")
    if not isinstance(args, list) or "-DSTRATA_PREFILL_FINITE_PROBE=ON" not in args:
        raise ValueError("successful Run69 build receipt configure argv lacks the exact diagnostic option")


def run66_failure_history_path(run66_matrix: dict[str, Any]) -> Path:
    side = next((item for item in run66_matrix.get("sides", []) if item.get("id") == "upstream42"), None)
    if side is None or side.get("arm_path") != "arms/upstream42/quality-on":
        raise ValueError("Run66 upstream42 side is missing or has an unexpected history arm path")
    return RUN66 / side["arm_path"] / "quality/F-quality-failure.json"


def validate_run66_failure_history(history: dict[str, Any]) -> None:
    completed = history.get("completed_prompts")
    expected = list(MANIFEST_IDS[:5])
    actual = [item.get("prompt_id") for item in completed if isinstance(item, dict)] if isinstance(completed, list) else None
    if (history.get("status") != "failed_stopped" or actual != expected or
            "long_1k" not in history.get("error", "")):
        raise ValueError("Run66 upstream42 history no longer shows five completed prompts then the long_1k stop-failure")


def replace_once(text: str, old: str, new: str, name: str) -> str:
    count = text.count(old)
    if count != 1:
        raise ValueError(f"{name}: expected one anchored source segment, found {count}")
    return text.replace(old, new, 1)


def _powershell_diagnostic_functions() -> str:
    return r'''function Convert-DiagnosticLine([string]$Line) {
    function Test-DiagnosticInteger($Value) {
        if ($null -eq $Value -or $Value -is [bool]) { return $false }
        return ($Value -is [sbyte] -or $Value -is [byte] -or $Value -is [int16] -or $Value -is [uint16] -or
                $Value -is [int32] -or $Value -is [uint32] -or $Value -is [int64] -or $Value -is [uint64])
    }
    if ($Line.StartsWith('strata zero-logits diag:')) {
        return [ordered]@{source='zero_logits';classification='fatal_zero_logits';raw_line=$Line}
    }
    if ($Line.StartsWith('strata zero-logits')) {
        return [ordered]@{source='zero_logits';classification='fatal_invalid';raw_line=$Line;error='unrecognized zero-logits diagnostic format'}
    }
    if ($Line.StartsWith('strata pf_nonfinite: ')) {
        $payload=$Line.Substring('strata pf_nonfinite: '.Length)
        try { $record=$payload|ConvertFrom-Json -AsHashtable -ErrorAction Stop }
        catch { return [ordered]@{source='prefill_finite_probe';classification='fatal_invalid';raw_line=$Line;error=('invalid JSON: '+$_.Exception.Message)} }
        if ($record -isnot [System.Collections.IDictionary] -or -not (Test-DiagnosticInteger $record.schema) -or $record.schema -ne 1) {
            return [ordered]@{source='prefill_finite_probe';classification='fatal_invalid';raw_line=$Line;parsed=$record;error='unsupported probe schema'}
        }
        if ($record.event -eq 'begin') {
            if ((Test-DiagnosticInteger $record.run) -and (Test-DiagnosticInteger $record.device) -and
                (Test-DiagnosticInteger $record.pos0) -and (Test-DiagnosticInteger $record.tokens) -and $record.tokens -gt 0 -and
                $record.diagnostic_only -is [bool] -and $record.diagnostic_only -eq $true) {
                return [ordered]@{source='prefill_finite_probe';classification='normal_begin';raw_line=$Line;parsed=$record}
            }
        } elseif ($record.event -eq 'checkpoint') {
            $required=@('run','device','completed_chunk','completed_pos0','completed_rows','scans')
            $complete=$true;foreach($key in $required){if(-not $record.Contains($key) -or -not (Test-DiagnosticInteger $record[$key])){$complete=$false}}
            if(-not $record.Contains('first_bad')){$complete=$false}
            if ($complete -and $null -eq $record.first_bad) {
                return [ordered]@{source='prefill_finite_probe';classification='normal_checkpoint';raw_line=$Line;parsed=$record}
            }
            if ($complete -and $record.first_bad -is [System.Collections.IDictionary]) {
                $bad=$record.first_bad;$badRequired=@('ordinal','stage','chunk','pos0','layer','half','rows','width','index','row','column','nan','positive_inf','negative_inf')
                $badComplete=$true;foreach($key in $badRequired){if(-not $bad.Contains($key)){$badComplete=$false}};
                foreach($key in @('ordinal','chunk','pos0','layer','half','rows','width','index','row','column','nan','positive_inf','negative_inf')){if(-not (Test-DiagnosticInteger $bad[$key])){$badComplete=$false}}
                $validStages=@('input_residual','ple_residual','hc_mixed','gdn_output','qsa_output','moe_output','residual_write')
                if ($badComplete -and $bad.ordinal -gt 0 -and $bad.stage -is [string] -and $bad.stage -in $validStages -and
                    $bad.rows -gt 0 -and $bad.width -gt 0 -and ($bad.nan+$bad.positive_inf+$bad.negative_inf) -gt 0) {
                    return [ordered]@{source='prefill_finite_probe';classification='fatal_nonfinite';raw_line=$Line;parsed=$record}
                }
            }
        }
        return [ordered]@{source='prefill_finite_probe';classification='fatal_invalid';raw_line=$Line;parsed=$record;error='unknown event or missing schema fields'}
    }
    if ($Line.StartsWith('strata pf_nonfinite')) {
        return [ordered]@{source='prefill_finite_probe';classification='fatal_invalid';raw_line=$Line;error='malformed probe prefix'}
    }
    return $null
}

function Get-DiagnosticSlice([long]$StartOffset) {
    if ($StartOffset -lt 0) { return [pscustomobject]@{end_offset=$StartOffset;records=@();partial_count=0;partial_base64='';reader_error='negative diagnostic cursor'} }
    if (-not (Test-Path -LiteralPath $diagLog -PathType Leaf)) {
        return [pscustomobject]@{end_offset=$StartOffset;records=@();partial_count=0;partial_base64='';reader_error=$null}
    }
    $stream=$null
    $rawSnapshotBase64=''
    try {
        $stream=[IO.File]::Open($diagLog,[IO.FileMode]::Open,[IO.FileAccess]::Read,[IO.FileShare]::ReadWrite)
        $snapshot=[long]$stream.Length
        if ($snapshot -lt $StartOffset) { throw 'diagnostic log shrank behind its byte cursor' }
        $remaining=$snapshot-$StartOffset
        if ($remaining -gt 16777216) { throw 'diagnostic snapshot exceeds 16 MiB read bound' }
        $bytes=[byte[]]::new([int]$remaining)
        [void]$stream.Seek($StartOffset,[IO.SeekOrigin]::Begin)
        $read=0
        while ($read -lt $bytes.Length) {
            $n=$stream.Read($bytes,$read,$bytes.Length-$read)
            if ($n -le 0) { throw 'diagnostic snapshot ended before its captured file length' }
            $read+=$n
        }
        $rawSnapshotBase64=[Convert]::ToBase64String($bytes)
        $stream.Dispose();$stream=$null
        $lastLf=-1
        for($i=$bytes.Length-1;$i -ge 0;$i--){if($bytes[$i] -eq 10){$lastLf=$i;break}}
        $completeLength=$lastLf+1
        $partialLength=$bytes.Length-$completeLength
        $partial=[byte[]]::new($partialLength)
        if($partialLength -gt 0){[Array]::Copy($bytes,$completeLength,$partial,0,$partialLength)}
        $records=@()
        if($completeLength -gt 0){
            $text=[Text.UTF8Encoding]::new($false,$true).GetString($bytes,0,$completeLength)
            foreach($line in [regex]::Split($text,"`n")){
                if(-not $line){continue};$line=$line.TrimEnd("`r")
                $record=Convert-DiagnosticLine $line
                if($null -ne $record){$records+=,$record}
            }
        }
        return [pscustomobject]@{end_offset=($StartOffset+$completeLength);snapshot_length=$snapshot;records=$records;
            partial_count=$partialLength;partial_base64=[Convert]::ToBase64String($partial);reader_error=$null}
    } catch {
        return [pscustomobject]@{end_offset=$StartOffset;snapshot_length=$null;records=@([ordered]@{source='diagnostic_reader';classification='fatal_reader_error';error=$_.Exception.Message;raw_snapshot_base64=$rawSnapshotBase64});
            partial_count=0;partial_base64='';reader_error=$_.Exception.Message}
    } finally { if($stream){$stream.Dispose()} }
}

function Save-PromptDiagnostics([string]$PromptId,[long]$CursorStart,[long]$CursorEnd,[object[]]$Records,
                                [int]$PartialCount,[string]$PartialBase64,[string]$ReaderError) {
    $fatal=@($Records|Where-Object{$_.classification -like 'fatal_*'})
    $status=if($fatal.Count){'failclosed_fatal_diagnostic'}elseif($Records.Count){'normal_diagnostic_records'}else{'no_diagnostic_records'}
    $body=[ordered]@{schema_version=1;run_id='20261011-69-probe69-quality';prompt_id=$PromptId;status=$status;
        captured_utc=[DateTime]::UtcNow.ToString('o');cursor_start=$CursorStart;cursor_end=$CursorEnd;
        complete_record_count=$Records.Count;partial_line_bytes=$PartialCount;partial_line_base64=$PartialBase64;
        reader_error=$ReaderError;fatal_record_count=$fatal.Count;records=@($Records)}
    $path=Join-Path (Join-Path $runDir 'quality') ("F-{0}-diagnostics.json" -f $PromptId)
    Write-ExclusiveJson $path $body 16
    return $path
}
'''


def _rewrite_manifest_validation(source: str) -> str:
    start = "    $manifestCheck = @'\n"
    end = "\n'@\n    $idsJson = ConvertTo-Json -InputObject $expectedIds -Compress"
    if source.count(start) != 1 or source.count(end) != 1:
        raise ValueError("Run66 quality manifest validation block anchors changed")
    block = r'''    $manifestCheck = @'
import hashlib,json,pathlib,sys
root=pathlib.Path(sys.argv[1]); manifest_path=pathlib.Path(sys.argv[2]); expected_ids=json.loads(__import__("base64").b64decode(sys.argv[3]))
all_ids=["smoke_arithmetic","smoke_unicode","smoke_json_arithmetic","smoke_python_function","smoke_three_key_retrieval","long_1k","long_4k","long_16k","long_30k7"]
sys.path.insert(0,str(root/"tools"))
from hetero_quality import load_manifest
from hetero_compare_quality import verify_tokenizer_manifest
m,by_id=load_manifest(manifest_path)
prompts=m.get("prompts",[])
if [x.get("id") for x in prompts]!=all_ids: raise SystemExit("pinned nine-prompt manifest order changed")
if [x.get("id") for x in prompts[:len(expected_ids)]]!=expected_ids: raise SystemExit("diagnostic six-prompt prefix changed")
caps={x:(256 if x=="smoke_python_function" else 128) for x in all_ids}
checks=[]
for p in prompts:
    path=(manifest_path.parent/p["file"]).resolve(strict=True); data=path.read_bytes()
    if len(data)!=p["bytes"] or hashlib.sha256(data).hexdigest()!=p["sha256"]: raise SystemExit("prompt bytes/SHA mismatch: "+p["id"])
    if p.get("chat_prompt_tokens_local") is None or p.get("max_output_tokens",0)<1: raise SystemExit("prompt counts missing: "+p["id"])
    checks.append({"id":p["id"],"body_tokens":p["body_tokens_local"],"chat_tokens":p["chat_prompt_tokens_local"],"max_tokens":caps[p["id"]],"sha256":p["sha256"]})
tok=verify_tokenizer_manifest(m)
if tok.get("status")!="pass": raise SystemExit("tokenizer asset/source verification failed: "+json.dumps(tok))
selected=[p for p in checks if p["id"] in expected_ids]
print(json.dumps({"status":"pass","manifest_prompt_count":len(checks),"prompt_count":len(selected),"prompts":selected,"tokenizer":tok},separators=(",",":")))
'@
    $idsJson = ConvertTo-Json -InputObject $expectedIds -Compress'''
    before, _ = source.split(start, 1)
    _, after = source.split(end, 1)
    return before + block + after


def rewrite_quality_script(source: str, new_run_dir: str) -> str:
    old_run_id = RUN66_RUN_ID
    new_run_id = RUN_ID
    new_model = "qwen3.8-flash-next-unsloth-ud-q4_k_xl-hetero42"
    old_expected = "$expectedIds = @('smoke_arithmetic','smoke_unicode','smoke_json_arithmetic','smoke_python_function',\n                'smoke_three_key_retrieval','long_1k','long_4k','long_16k','long_30k7')"
    new_expected = "$expectedIds = @('smoke_arithmetic','smoke_unicode','smoke_json_arithmetic','smoke_python_function',\n                'smoke_three_key_retrieval','long_1k')\n$expectedManifestIds = @('smoke_arithmetic','smoke_unicode','smoke_json_arithmetic','smoke_python_function',\n                'smoke_three_key_retrieval','long_1k','long_4k','long_16k','long_30k7')"
    result = source.replace(old_run_id, new_run_id)
    old_dir = str(RUN66 / "arms/hetero42/quality-on").replace("/", "\\")
    if old_dir not in result:
        raise ValueError("Run66 quality run directory anchor missing")
    result = result.replace(old_dir, new_run_dir, 1)
    result = replace_once(result, old_expected, new_expected, "six-prompt prefix")
    result = _rewrite_manifest_validation(result)
    result = replace_once(
        result,
        "if ($manifestInfo.status -ne 'pass' -or $manifestInfo.prompt_count -ne 9) { throw 'manifest validation result was incomplete' }",
        "if ($manifestInfo.status -ne 'pass' -or $manifestInfo.manifest_prompt_count -ne 9 -or $manifestInfo.prompt_count -ne $expectedIds.Count) { throw 'manifest/prefix validation result was incomplete' }",
        "six-prompt manifest validation result",
    )
    result = replace_once(
        result,
        "[ordered]@{status='prepared_quality_validated';run_started=$false;prompt_count=9;formal_per_prompt=3;warmups_per_prompt=1;\n        caps='128 for eight non-code tasks, 256 for smoke_python_function';manifest_sha256=$contract.manifest_sha256;",
        "[ordered]@{status='prepared_quality_validated';run_started=$false;prompt_count=6;manifest_prompt_count=9;formal_per_prompt=3;warmups_per_prompt=1;\n        caps='128 for five prompts, 256 for smoke_python_function';manifest_sha256=$contract.manifest_sha256;",
        "six-prompt default validation result",
    )
    result = replace_once(
        result,
        "$preflight = [ordered]@{status='live_quality_preflight_pass';chat_requests_sent=0;prompt_count=$contract.manifest.prompts.Count;",
        "$preflight = [ordered]@{status='live_quality_preflight_pass';chat_requests_sent=0;prompt_count=$expectedIds.Count;manifest_prompt_count=$contract.manifest.prompts.Count;",
        "six-prompt ready-preflight result",
    )
    old_env_guard = "if ($config.env.STRATA_PREFILL_CPU_SHARE -ne '0' -or $config.env.STRATA_STAGE_PIN -ne '0') { throw 'v0.1.42 CPU-share/stage-pin controls must remain explicitly disabled' }"
    new_env_guard = "if ($config.env.STRATA_PREFILL_CPU_SHARE -ne '0' -or $config.env.STRATA_STAGE_PIN -ne '0' -or $config.env.STRATA_ROUTE_TAIL_SKIP -ne '0') { throw 'Run66 CPU-share/stage-pin/route-tail controls must remain explicitly disabled' }\n    if ($config.env.STRATA_PREFILL_FINITE_PROBE -ne '1' -or $config.env.STRATA_DIAG_ZERO_LOGITS -ne '1' -or $identity.config.diagnostic_build_option -ne 'STRATA_PREFILL_FINITE_PROBE=ON' -or $provenance.diagnostic_environment_delta.STRATA_PREFILL_FINITE_PROBE -ne '1' -or $provenance.diagnostic_environment_delta.STRATA_DIAG_ZERO_LOGITS -ne '1' -or $provenance.build_receipt.compile_option -ne 'STRATA_PREFILL_FINITE_PROBE=ON') { throw 'Run69 compile/runtime diagnostic contract mismatch' }"
    result = replace_once(result, old_env_guard, new_env_guard, "Run69 explicit diagnostic contract")
    result = replace_once(
        result,
        "$promptResults = @(); $diagOffset = (Get-Item -LiteralPath $diagLog).Length",
        "$promptResults = @(); $diagOffset = (Get-Item -LiteralPath $diagLog).Length\n$currentPromptId=$null;$currentDiagnosticsReceipt=$null\n$selectedPrompts = @()\nforeach($expectedId in $expectedIds){$selected=@($contract.manifest.prompts|Where-Object{$_.id -eq $expectedId});if($selected.Count -ne 1){throw \"${expectedId}: selected prompt is not unique\"};$selectedPrompts += $selected[0]}",
        "quality execution prompt scope",
    )
    result = replace_once(result, "foreach ($prompt in $contract.manifest.prompts) {", "foreach ($prompt in $selectedPrompts) {", "six-prompt loop")
    result = result.replace(_powershell_diagnostic_functions_old(source), _powershell_diagnostic_functions())
    if "function Get-DiagnosticSlice([long]$StartOffset)" not in result:
        raise ValueError("Run66 diagnostic reader anchor did not survive replacement")
    result = _rewrite_prompt_diagnostic_flow(result)
    result = _rewrite_six_prompt_final(result)
    result = result.replace("diagnostic_observed=$false", "diagnostic_observed=(@($promptDiagnosticRecords|Where-Object{$_.classification -eq 'fatal_zero_logits'}).Count -gt 0);prefill_diagnostic_receipt=$diagnosticsReceiptPath;prefill_diagnostic_status=$diagnosticsStatus")
    result = replace_once(result, "$id = $prompt.id; $maxTokens = if ($id -eq 'smoke_python_function') { 256 } else { 128 }",
                          "$id = $prompt.id; $currentPromptId=$id;$currentDiagnosticsReceipt=$null; $maxTokens = if ($id -eq 'smoke_python_function') { 256 } else { 128 }",
                          "active diagnostic prompt identity")
    result = result.replace("partial_prompts=$promptResults.Count}", "partial_prompts=$promptResults.Count;failed_prompt=$currentPromptId;diagnostics_file=$currentDiagnosticsReceipt}")
    result = result.replace("error=$_.Exception.Message;completed_prompts=$promptResults})", "error=$_.Exception.Message;failed_prompt=$currentPromptId;diagnostics_file=$currentDiagnosticsReceipt;completed_prompts=$promptResults})")
    result = result.replace("$expectedModel = 'qwen3.8-flash-next-unsloth-ud-q4_k_xl-hetero42'", f"$expectedModel = '{new_model}'")
    return result


def _powershell_diagnostic_functions_old(source: str) -> str:
    start = source.find("function Get-DiagnosticSlice([long]$StartOffset) {")
    end = source.find("\nfunction Get-ReadyBindings(", start)
    if start < 0 or end < 0:
        raise ValueError("Run66 diagnostic reader source function anchors changed")
    return source[start:end]


def _rewrite_prompt_diagnostic_flow(source: str) -> str:
    # This patch is intentionally split around unique control-flow anchors so source drift fails closed.
    poll_old = "$diag = Get-DiagnosticSlice $diagOffset; $diagOffset = $diag.end_offset\n                if (@($diag.lines).Count -gt 0) { Write-Progress @{phase=$id;event='zero_logits_diagnostic';lines=$diag.lines} }"
    poll_new = "$diag = Get-DiagnosticSlice $diagOffset; $diagOffset = $diag.end_offset\n                foreach($record in @($diag.records)){$record['origin']='during_request';$promptDiagnosticRecords+=,$record}\n                if ($diag.reader_error) { $promptDiagnosticReaderError = $diag.reader_error }\n                if (@($diag.records).Count -gt 0 -or $diag.partial_count -gt 0 -or $diag.reader_error) { Write-Progress @{phase=$id;event='prefill_diagnostic_batch';records=@($diag.records);partial_line_bytes=$diag.partial_count;reader_error=$diag.reader_error} }"
    source = replace_once(source, poll_old, poll_new, "diagnostic polling batch")
    before_old = "$beforeDiag = Get-DiagnosticSlice $diagOffset; $diagOffset = $beforeDiag.end_offset\n        if (@($beforeDiag.lines).Count -gt 0) { throw \"${id}: zero-logits diagnostic arrived before prompt; no request sent\" }"
    before_new = "$promptDiagCursorStart=$diagOffset;$promptDiagnosticRecords=@();$promptDiagnosticReaderError=$null\n        $beforeDiag=Get-DiagnosticSlice $diagOffset;$diagOffset=$beforeDiag.end_offset\n        foreach($record in @($beforeDiag.records)){$record['origin']='before_request';$promptDiagnosticRecords+=,$record}\n        if($beforeDiag.reader_error){$promptDiagnosticReaderError=$beforeDiag.reader_error}\n        if($beforeDiag.partial_count -gt 0){$promptDiagnosticRecords+=[ordered]@{source='diagnostic_reader';classification='fatal_partial_line';origin='before_request';partial_base64=$beforeDiag.partial_base64;partial_bytes=$beforeDiag.partial_count}}\n        $beforeFatal=(@($promptDiagnosticRecords|Where-Object{$_.classification -like 'fatal_*'}).Count -gt 0) -or [bool]$beforeDiag.reader_error -or ($beforeDiag.partial_count -gt 0)\n        if($beforeFatal){\n            $preDiagPath=Save-PromptDiagnostics $id $promptDiagCursorStart $diagOffset $promptDiagnosticRecords $beforeDiag.partial_count $beforeDiag.partial_base64 $promptDiagnosticReaderError\n            $currentDiagnosticsReceipt=$preDiagPath\n            throw \"${id}: fatal or malformed diagnostic boundary before request; saved $preDiagPath\"\n        }\n        if($promptDiagnosticRecords.Count -gt 0){Write-Progress @{phase=$id;event='prefill_diagnostic_before_request';classification='normal';records=@($promptDiagnosticRecords)}}"
    source = replace_once(source, before_old, before_new, "pre-request diagnostic boundary")
    old_targets = "        $stderrPath = Join-Path (Join-Path $runDir 'logs') \"quality-F-$id.stderr.log\"\n        foreach ($target in @($requestDir,$reportPath,$stdoutPath,$stderrPath))"
    new_targets = "        $stderrPath = Join-Path (Join-Path $runDir 'logs') \"quality-F-$id.stderr.log\"\n        $diagnosticsPath = Join-Path (Join-Path $runDir 'quality') \"F-$id-diagnostics.json\"\n        foreach ($target in @($requestDir,$reportPath,$stdoutPath,$stderrPath,$diagnosticsPath))"
    source = replace_once(source, old_targets, new_targets, "fresh diagnostic receipt output leaf guard")
    after_runner_anchor = "        if (-not (Test-Path -LiteralPath $requestDir -PathType Container)) { throw \"${id}: request output directory missing (runner exit $runnerExit)\" }"
    add_save = "        $finalDiag=Get-DiagnosticSlice $diagOffset;$diagOffset=$finalDiag.end_offset;$promptDiagnosticRecords+=@($finalDiag.records)\n        if($finalDiag.reader_error){$promptDiagnosticReaderError=$finalDiag.reader_error}\n        if($finalDiag.partial_count -gt 0){$promptDiagnosticRecords+=[ordered]@{source='diagnostic_reader';classification='fatal_partial_line';partial_base64=$finalDiag.partial_base64;partial_bytes=$finalDiag.partial_count}}\n        $diagnosticsReceiptPath=Save-PromptDiagnostics $id $promptDiagCursorStart $diagOffset $promptDiagnosticRecords $finalDiag.partial_count $finalDiag.partial_base64 $promptDiagnosticReaderError\n        $currentDiagnosticsReceipt=$diagnosticsReceiptPath\n        $diagnosticsStatus=if(@($promptDiagnosticRecords|Where-Object{$_.classification -like 'fatal_*'}).Count){'fatal_diagnostic_after_prompt'}elseif($promptDiagnosticRecords.Count){'normal_diagnostic_records'}else{'no_diagnostic_records'}\n        Write-Progress @{phase=$id;event='prompt_diagnostics_saved';status=$diagnosticsStatus;path=$diagnosticsReceiptPath;record_count=$promptDiagnosticRecords.Count}\n" + after_runner_anchor
    source = replace_once(source, after_runner_anchor, add_save, "diagnostics receipt before quality evaluator")
    old_after = "        $afterDiag = Get-DiagnosticSlice $diagOffset; $diagOffset = $afterDiag.end_offset\n        if (@($afterDiag.lines).Count -gt 0) { throw \"${id}: zero-logits diagnostic appeared during request; stop matrix\" }\n        $null = Get-LatestResourceGate \"after $id\"; $null = Get-ReadyBindings \"after $id\" 0"
    new_after = "        $null = Get-LatestResourceGate \"after $id\"; $null = Get-ReadyBindings \"after $id\" 0"
    source = replace_once(source, old_after, new_after, "remove any-diagnostic-fatal rule")
    save_result_anchor = "        Write-ExclusiveJson (Join-Path (Join-Path $runDir 'quality') \"F-$id-result.json\") $row"
    save_result_new = save_result_anchor + "\n        if($diagnosticsStatus -eq 'fatal_diagnostic_after_prompt'){throw \"${id}: fatal zero-logits/prefill diagnostic; stop before the next prompt\"}"
    source = replace_once(source, save_result_anchor, save_result_new, "stop after current prompt diagnostics are saved")
    return source


def _rewrite_six_prompt_final(source: str) -> str:
    start = "    $final = [ordered]@{schema_version=1;run_id='20261011-69-probe69-quality';status='nine_prompt_matrix_complete';"
    end = "        sampler_stop_requested=$false;sampler_remains_owned_for_root_cleanup=$true} | ConvertTo-Json -Compress"
    begin = source.find(start)
    finish = source.find(end, begin)
    if begin < 0 or finish < 0:
        raise ValueError("Run66 nine-prompt final metadata block anchors changed")
    finish += len(end)
    block = """    $remaining=@('long_4k','long_16k','long_30k7')
    $final = [ordered]@{schema_version=1;run_id='20261011-69-probe69-quality';status='six_prompt_probe_diagnostic_complete';
        completed_utc=[DateTime]::UtcNow.ToString('o');prompt_count=6;warmups_per_prompt=1;formal_per_prompt=3;
        formal_requests=18;warmup_requests=6;caps='128 for five prompts; 256 for smoke_python_function';
        engine_sha256=$identity.engine.sha256;server_file_sha256=$provenance.server_file_sha256;
        long_1k_completed=(@($promptResults|Where-Object{$_.prompt_id -eq 'long_1k'}).Count -eq 1);
        remaining_long_prompts_not_run=$remaining;results=$promptResults;diagnostic_only=$true}
    Write-ExclusiveJson (Join-Path (Join-Path $runDir 'quality') 'F-six-prompt-stage.json') $final
    Write-Progress @{phase='complete';status='six_prompt_probe_diagnostic_complete';sampler_stop_requested=$false;sampler_remains_owned_for_root_review=$true}
    [ordered]@{run=$runDir;status='six_prompt_probe_diagnostic_complete';prompt_count=6;formal_requests=18;warmup_requests=6;
        long_1k_completed=$final.long_1k_completed;remaining_long_prompts_not_run=$remaining;
        sampler_stop_requested=$false;sampler_remains_owned_for_root_review=$true} | ConvertTo-Json -Compress"""
    return source[:begin] + block + source[finish:]


def render_quality_script(source: str, new_run_dir: str) -> str:
    return rewrite_quality_script(source, new_run_dir)


def _source_tree_and_build_receipt() -> tuple[dict[str, Any], dict[str, Any], Path]:
    auth = load_json(BUILD_AUTH)
    if auth.get("status") != "root_reviewed_authorized_build_only" or auth.get("source_head") != SOURCE_HEAD:
        raise ValueError("Run69 build authorization/source pin is absent or changed")
    build_dir = Path(auth.get("build_path", ""))
    expected_build_dir = Path(r"E:\Strata-Hetero-data\build\cuda-prefill-probe69-01")
    if build_dir.resolve() != expected_build_dir:
        raise ValueError("Run69 build directory differs from the reviewed build authorization")
    receipt_path = build_dir / "build-receipt.json"
    if not receipt_path.is_file():
        raise RuntimeError("Run69 build receipt is not present yet; this preparation program must wait")
    build = load_json(receipt_path)
    if build.get("status") != "success" or build.get("gpu_executed") is not False or build.get("model_or_engine_cli_executed") is not False:
        raise ValueError("Run69 build receipt is not a successful build-only receipt")
    if build.get("tests_run") is not False or build.get("test_binary_executed") is not False or build.get("tests_built") is not True:
        raise ValueError("Run69 build receipt test build/run boundary differs")
    for field in ("source_before", "source_after"):
        pin = build.get(field) or {}
        if pin.get("head") != SOURCE_HEAD or pin.get("clean") is not True:
            raise ValueError(f"Run69 source {field} is not clean at the pinned E69 HEAD")
    before, after = build["source_before"], build["source_after"]
    if before.get("tree") != after.get("tree"):
        raise ValueError("Run69 source tree changed during the successful build")
    binary = build.get("binary") or {}
    expected_binary = str(expected_build_dir / "strata.exe")
    if (binary.get("path") != expected_binary or not isinstance(binary.get("sha256"), str) or
            re.fullmatch(r"[0-9a-f]{64}", binary["sha256"]) is None or
            type(binary.get("size_bytes")) is not int or binary["size_bytes"] <= 0):
        raise ValueError("Run69 binary identity is missing or does not match the reviewed build directory")
    test_binary = build.get("test_binary") or {}
    if (not isinstance(test_binary.get("sha256"), str) or re.fullmatch(r"[0-9a-f]{64}", test_binary["sha256"]) is None or
            type(test_binary.get("size_bytes")) is not int or test_binary["size_bytes"] <= 0):
        raise ValueError("Run69 synthetic test binary identity is missing from the successful build receipt")
    validate_probe_build_controls(build)
    head = subprocess.run(["git", "-C", str(SOURCE), "rev-parse", "HEAD"], capture_output=True, text=True, check=False)
    status = subprocess.run(["git", "-C", str(SOURCE), "status", "--porcelain=v1", "--untracked-files=all"],
                            capture_output=True, text=True, check=False)
    if head.returncode or head.stdout.strip() != SOURCE_HEAD or status.returncode or status.stdout.strip():
        raise ValueError("Run69 source worktree no longer matches the clean reviewed E69 HEAD")
    return auth, build, receipt_path


def _replace_run_bindings(script: str, old_root: str, new_root: str, *, allow_zero_run_id: bool = False) -> str:
    old_normal = old_root.replace("/", "\\")
    new_normal = new_root.replace("/", "\\")
    if script.count(old_normal) != 1:
        raise ValueError("rendered lifecycle script must contain exactly one Run66 absolute run-directory binding")
    result = script.replace(old_normal, new_normal, 1)
    if RUN66_RUN_ID not in script and not allow_zero_run_id:
        raise ValueError("rendered lifecycle script is missing Run66 RunId binding")
    return result.replace(RUN66_RUN_ID, RUN_ID)


def render_phase_scripts(phase_dir: Path, out_dir: Path, model_name: str) -> tuple[dict[str, str], dict[str, Any]]:
    source_dir = RUN66 / "rendered-scripts" / RUN66_SIDE_ID
    old_dir = str(RUN66 / "arms/hetero42/quality-on")
    new_dir = str(phase_dir.resolve())
    result: dict[str, str] = {}
    bindings: dict[str, Any] = {}
    for source_name, target_name in (("launch.ps1", "launch.ps1"),
                                     ("startup-watch.ps1", "startup-watch.ps1"),
                                     ("quality.ps1", "quality.ps1"),
                                     ("cleanup.ps1", "cleanup.ps1")):
        source_path = source_dir / source_name
        source = source_path.read_text(encoding="utf-8-sig")
        rendered = render_quality_script(source, new_dir) if source_name == "quality.ps1" else _replace_run_bindings(
            source, old_dir, new_dir, allow_zero_run_id=(source_name == "cleanup.ps1"))
        old_model = "qwen3.8-flash-next-unsloth-ud-q4_k_xl-hetero42"
        if model_name != old_model:
            rendered = rendered.replace(old_model, model_name)
        if source_name == "launch.ps1":
            if "RUN47_CONTROLLER_PROOF" in rendered or "--owned-controller-proof" in rendered:
                raise ValueError("Run66 effective direct-PS launch unexpectedly contains pool proof argument")
            if "--required-vram-mib 46000 --minimum-ram-gib 12 --check-wsl-current" not in rendered:
                raise ValueError("Run69 direct launch lost strict original admission arguments")
        if source_name == "cleanup.ps1":
            if rendered.count("RUN41-SAMPLER-FIRST-FINALIZATION-BEGIN") != 1 or "finally {" not in rendered:
                raise ValueError("Run69 cleanup did not retain sampler-first exact-owner finally block")
        target = out_dir / target_name
        result[source_name.removesuffix(".ps1").replace("startup-watch", "watch")] = rendered
        bindings[source_name.removesuffix(".ps1").replace("startup-watch", "watch")] = {
            "source_run66_path": str(source_path.resolve()),
            "source_run66_sha256": sha_file(source_path),
            "rendered_path": str(target.resolve()),
            "rendered_sha256": sha_bytes(rendered.encode("utf-8")),
            "rendered_bytes": len(rendered.encode("utf-8")),
            "status": "prepared_not_executed_parser_validation_required",
        }
    return result, bindings


def _render_run_phase(source: str) -> str:
    old = "[ValidateSet('upstream42','hetero42')][string]$Side='upstream42'"
    new = "[ValidateSet('probe69')][string]$Side='probe69'"
    result = replace_once(source, old, new, "RunPhase single-side allowlist")
    if "-owned-controller-proof" in result or "RUN47_CONTROLLER_PROOF" in result:
        raise ValueError("RunPhase wrapper unexpectedly contains coordinator-proof exemption")
    return result


def _fresh_matrix_side(base_side: dict[str, Any], phase_dir: Path, build: dict[str, Any], config_sha: str,
                       identity_sha: str, provenance_sha: str, prep_sha: str,
                       rendered: dict[str, dict[str, Any]]) -> dict[str, Any]:
    side = copy.deepcopy(base_side)
    side.update({
        "id": SIDE_ID,
        "run_id": RUN_ID,
        "arm_path": "arms/probe69/quality-on",
        "config_path": "arms/probe69/quality-on/config.json",
        "identity_path": "arms/probe69/quality-on/identity.json",
        "provenance_path": "arms/probe69/quality-on/provenance.json",
        "preparation_path": "arms/probe69/quality-on/preparation.json",
        "source_commit": SOURCE_HEAD,
        "binary_sha256": build["binary"]["sha256"],
        "binary_bytes": build["binary"]["size_bytes"],
        "config_sha256": config_sha,
        "identity_sha256": identity_sha,
        "provenance_sha256": provenance_sha,
        "preparation_sha256": prep_sha,
        "status": "prepared_not_launched_root_build_receipt_bound_fresh_launch_recheck_required",
        "binary_identity_status": "root_build_receipt_bound_fresh_launch_rehash_required",
        "rendered_scripts": rendered,
    })
    return side


def _build_one_side_contract(base_side: dict[str, Any], build: dict[str, Any], build_receipt: Path,
                             phase_dir: Path, script_bindings: dict[str, Any], wrapper_sha: str,
                             wrapper_bytes: int, preparation_source_sha: str) -> dict[str, Any]:
    base_phase = RUN66 / base_side["arm_path"]
    base_config = load_json(base_phase / "config.json")
    config = copy.deepcopy(base_config)
    if set(config.get("env", {})).intersection(DIAGNOSTIC_ENV):
        raise ValueError("Run66 base config already contains Run69 diagnostic environment keys")
    if config["env"].get("STRATA_ROUTE_TAIL_SKIP") != "0":
        raise ValueError("Run66 route-tail-skip control is not zero")
    binary = build["binary"]
    build_receipt_sha = sha_file(build_receipt)
    config["exe"] = binary["path"]
    config["log"] = str((phase_dir / "logs" / "engine.log").resolve())
    config["env"].update(DIAGNOSTIC_ENV)
    normalized_base = copy.deepcopy(base_config)
    normalized_new = copy.deepcopy(config)
    normalized_base["exe"] = normalized_new["exe"] = "@ENGINE_PATH@"
    normalized_base["log"] = normalized_new["log"] = "@RUN_LOCAL_LOG@"
    for name in DIAGNOSTIC_ENV:
        normalized_new["env"].pop(name, None)
    if normalized_base != normalized_new:
        raise ValueError("Run69 config differs from Run66 beyond binary/log paths and the two diagnostic env additions")
    config_raw = _json_bytes(config)
    config_sha = sha_bytes(config_raw)

    identity = copy.deepcopy(load_json(base_phase / "identity.json"))
    source_after = build["source_after"]
    identity["source"].update({"sha": SOURCE_HEAD, "tree": source_after["tree"], "clean": True,
                               "binary_hash_finalized": True})
    identity["config"].update({
        "sha256": config_sha,
        "scope": "v0.1.42-E69-prefill-first-nonfinite-diagnostic",
        "diagnostic_environment_delta": copy.deepcopy(DIAGNOSTIC_ENV),
        "diagnostic_build_option": "STRATA_PREFILL_FINITE_PROBE=ON",
        "only_runtime_difference": "E69 source/binary plus opt-in prefill finite-probe and zero-logits diagnostics; Run66 controls retained",
    })
    identity["engine"].update({"version": "0.1.42", "path": binary["path"],
                                "sha256": binary["sha256"], "size_bytes": binary["size_bytes"],
                                "source_sha": SOURCE_HEAD, "build_receipt": str(build_receipt.resolve()),
                                "build_receipt_sha256": build_receipt_sha,
                                "cmake_probe_option": "STRATA_PREFILL_FINITE_PROBE=ON"})
    identity["server_python"]["scope"] = "pinned v1.42 token-ID bridge for the single E69 diagnostic engine"
    identity["acceptance_status"] = "prepared_not_launched"
    identity_raw = _json_bytes(identity)
    identity_sha = sha_bytes(identity_raw)

    provenance = copy.deepcopy(load_json(base_phase / "provenance.json"))
    provenance.update({
        "run_id": RUN_ID, "arm": SIDE_ID, "status": "prepared_not_launched",
        "prepared_utc": dt.datetime.now(dt.timezone.utc).isoformat().replace("+00:00", "Z"),
        "engine_cpp_sha": SOURCE_HEAD, "engine_version": "0.1.42",
        "engine_path": binary["path"], "engine_sha256": binary["sha256"],
        "engine_size_bytes": binary["size_bytes"], "config_sha256": config_sha,
        "identity_sha256": identity_sha, "acceptance_status": "prepared_not_launched",
        "diagnostic_only": True, "diagnostic_environment_delta": copy.deepcopy(DIAGNOSTIC_ENV),
    })
    controls = provenance["shared_controls"]
    controls["environment_frozen"] = copy.deepcopy(config["env"])
    controls["diagnostic_environment_delta"] = copy.deepcopy(DIAGNOSTIC_ENV)
    controls["diagnostic_build_option"] = "STRATA_PREFILL_FINITE_PROBE=ON"
    controls["new_v0_1_42_controls"] = {
        "STRATA_PREFILL_CPU_SHARE": "0", "STRATA_STAGE_PIN": "0", "STRATA_ROUTE_TAIL_SKIP": "0",
    }
    comparison = provenance["comparison_contract"]
    for key in ("normalized_controls_sha256", "normalized_config_sha256", "server_bridge_identical_for_both_runs",
                "native_dense_paths_same", "native_dense_contains_original_F_shard_2_in_both_arms",
                "normalization"):
        comparison.pop(key, None)
    comparison.update({
        "scope": "single-side prefill diagnostic; no pair comparison or quality-cause claim",
        "only_runtime_setting_difference": "E69 source/binary plus two explicitly enabled diagnostics; all other Run66 side controls are retained",
        "run66_controls_retained_except_named_diagnostics": True,
        "diagnostic_environment_delta": copy.deepcopy(DIAGNOSTIC_ENV),
        "diagnostic_build_option": "STRATA_PREFILL_FINITE_PROBE=ON",
        "prompt_scope": list(PROMPT_IDS), "warmups_per_prompt": 1,
        "formal_requests_per_prompt": 3, "stop_first_failed_prompt": True,
    })
    provenance["build_receipt"].update({
        "build_receipt": str(build_receipt.resolve()), "build_path": build["build_directory"],
        "source_commit": SOURCE_HEAD, "source_tree": source_after["tree"],
        "compile_option": "STRATA_PREFILL_FINITE_PROBE=ON",
        "build_receipt_sha256": build_receipt_sha,
        "binary_sha256": binary["sha256"], "binary_bytes": binary["size_bytes"],
        "test_binary_sha256": build["test_binary"]["sha256"],
        "test_binary_bytes": build["test_binary"]["size_bytes"],
        "tests_built": True, "tests_run": False, "test_binary_executed": False,
        "gpu_executed": False, "model_or_engine_cli_executed": False,
    })
    provenance["build_receipt"].pop("pair_summary", None)
    source_review_path = RUN / "preparation/root-source-review.json"
    provenance["source_review"] = {
        "path": str(source_review_path.resolve()),
        "sha256": sha_file(source_review_path),
        "status": "root_source_review_pass_build_pending",
        "source_head": SOURCE_HEAD,
        "source_tree": source_after["tree"],
    }
    provenance["diagnostic_history_reference"] = {
        "run_id": "20261011-66-v042-quality-pair",
        "matrix_path": str((RUN66 / "matrix.json").resolve()),
        "history_arm_id": "upstream42",
        "history_path": str(run66_failure_history_path(load_json(RUN66 / "matrix.json")).resolve()),
        "history_status": "five prior short prompts passed; long_1k was first stop-failure",
        "copied_request_outputs": False,
    }
    provenance_raw = _json_bytes(provenance)
    provenance_sha = sha_bytes(provenance_raw)

    fresh = copy.deepcopy(load_json(base_phase / "fresh-identities.json"))
    fresh["preparation_reference_only"] = {
        "source_run_id": "20261011-66-hetero42-quality",
        "source_path": str((base_phase / "fresh-identities.json").resolve()),
        "current_launch_stat_recheck_required": True,
        "payloads_copied_or_read": False,
        "this_preparation_performed_asset_observation": False,
    }
    fresh_raw = _json_bytes(fresh)
    fresh_sha = sha_bytes(fresh_raw)

    prep = copy.deepcopy(load_json(base_phase / "preparation.json"))
    prep.update({
        "run_id": RUN_ID, "arm": SIDE_ID, "status": "prepared_not_launched",
        "source_run66_reference": str((RUN66 / "matrix.json").resolve()),
        "source_commit": SOURCE_HEAD, "source_tree": source_after["tree"],
        "binary_path": binary["path"], "binary_sha256": binary["sha256"],
        "binary_bytes": binary["size_bytes"],
        "build_receipt_sha256": build_receipt_sha,
        "binary_identity_status": "root-build-receipt-bound-fresh-launch-recheck-required",
        "config_sha256": config_sha, "identity_sha256": identity_sha,
        "provenance_sha256": provenance_sha, "fresh_identities_sha256": fresh_sha,
        "diagnostic_environment_delta": copy.deepcopy(DIAGNOSTIC_ENV),
        "diagnostic_compile_option": "STRATA_PREFILL_FINITE_PROBE=ON",
        "prompt_ids": list(PROMPT_IDS), "warmups_per_prompt": 1,
        "formal_per_prompt": 3, "performance_phases": 0,
        "execution_authorized": False,
    })
    prep_raw = _json_bytes(prep)
    side = _fresh_matrix_side(base_side, phase_dir, build, config_sha, identity_sha, provenance_sha,
                              sha_bytes(prep_raw), script_bindings)
    side["fresh_identities_sha256"] = fresh_sha
    side["model_name"] = config["model_name"]
    return {
        "config": config, "config_raw": config_raw,
        "identity": identity, "identity_raw": identity_raw,
        "provenance": provenance, "provenance_raw": provenance_raw,
        "fresh_identities": fresh, "fresh_raw": fresh_raw,
        "preparation": prep, "preparation_raw": prep_raw,
        "side": side, "normalized_config_equal_after_diagnostic_delta": True,
        "build_receipt_sha256": build_receipt_sha,
        "script_bindings": script_bindings,
    }


def _build_control_contract(base_side: dict[str, Any], phase: dict[str, Any], build: dict[str, Any],
                           source_head: str, wrapper_sha: str,
                           rendered_scripts: dict[str, dict[str, Any]]) -> tuple[dict[str, Any], dict[str, Any]]:
    """Build single-side normalization and template provenance without pair-comparison claims."""
    base_config = load_json(RUN66 / base_side["config_path"])
    candidate = copy.deepcopy(phase["config"])

    def normalize(config: dict[str, Any], *, remove_diagnostics: bool) -> dict[str, Any]:
        value = copy.deepcopy(config)
        value["exe"] = "@ENGINE_PATH@"
        value["model_name"] = "@MODEL_NAME@"
        value["log"] = "@RUN_LOCAL_LOG@"
        if remove_diagnostics:
            for key in DIAGNOSTIC_ENV:
                value["env"].pop(key, None)
        return value

    normalized_base = normalize(base_config, remove_diagnostics=False)
    normalized_candidate = normalize(candidate, remove_diagnostics=True)
    if normalized_base != normalized_candidate:
        raise ValueError("Run69 controls differ from Run66 hetero42 beyond engine/log paths and the two diagnostic env additions")
    canonical_sha = sha_bytes(_json_bytes(normalized_candidate))

    controls = load_json(RUN66 / "controls-normalization.json")
    controls.update({
        "schema_version": 1,
        "status": "single_side_exact_normalized_controls_match_with_named_diagnostic_delta",
        "normalization": {
            "replace_executable_path_with": "@ENGINE_PATH@",
            "replace_model_name_with": "@MODEL_NAME@",
            "replace_run_local_log_path_with": "@RUN_LOCAL_LOG@",
            "remove_from_candidate_only": list(DIAGNOSTIC_ENV),
            "do_not_normalize_or_reorder_args": True,
            "do_not_normalize_other_environment": True,
        },
        "comparison_only_differences": [
            "E69 source/binary identity",
            "two opt-in diagnostic environment variables",
            "run-local log and owned evidence paths",
        ],
        "canonical_normalized_config_sha256": canonical_sha,
        "normalized_config_equality": "pass_exact_json_equality_after_named_diagnostic_and_path_normalization",
        "normalization_status": "exact_normalized_single_side_controls",
        "config_hashes": {SIDE_ID: phase["side"]["config_sha256"]},
        "identity_hashes": {SIDE_ID: phase["side"]["identity_sha256"]},
        "provenance_hashes": {SIDE_ID: phase["side"]["provenance_sha256"]},
        "normalization_verification": {
            "status": "pass_exact_json_equality",
            "normalized_fields": ["exe", "model_name", "log"],
            "removed_candidate_environment": list(DIAGNOSTIC_ENV),
            "all_other_config_fields_equal": True,
            "canonical_normalized_config_sha256": canonical_sha,
            "route_tail_skip": "0",
        },
        "binary_bindings": {
            SIDE_ID: {"sha256": build["binary"]["sha256"], "bytes": build["binary"]["size_bytes"],
                      "source_commit": source_head}
        },
        "source_commits": {SIDE_ID: source_head},
        "quality_scope": {
            "manifest_prompt_ids": list(MANIFEST_IDS), "executed_prompt_ids": list(PROMPT_IDS),
            "warmups_per_prompt": 1, "formal_repeats_per_prompt": 3,
            "strict_quality_stop_first": True, "performance_phase": False,
        },
        "quality": {
            "captured_actual_engine_token_ids": True,
            "warmups_per_prompt": 1, "formal_repeats_per_prompt": 3,
            "prompts": len(PROMPT_IDS), "manifest_prompts": len(MANIFEST_IDS),
            "performance_phase": False, "strict_stop_first": True,
        },
        "rendered_launch_hashes": {SIDE_ID: rendered_scripts["launch"]["rendered_sha256"]},
        "identity_provenance_schema_sources": {
            SIDE_ID: {
                "identity_template": str((RUN66 / base_side["identity_path"]).resolve()),
                "identity_template_sha256": sha_file(RUN66 / base_side["identity_path"]),
                "provenance_template": str((RUN66 / base_side["provenance_path"]).resolve()),
                "provenance_template_sha256": sha_file(RUN66 / base_side["provenance_path"]),
                "identity_sha256": phase["side"]["identity_sha256"],
                "provenance_sha256": phase["side"]["provenance_sha256"],
                "full_original_schema_copied": True,
            }
        },
        "execution_mode": "direct_powershell_controllers",
        "controller_wrapper": {
            "path": str((RUN / "run-phase.ps1").resolve()), "sha256": wrapper_sha,
            "mutated_by_this_preparation": True,
        },
        "authorization_status": "none; fresh Run69 root authorization required",
        "execution_enabled": False,
    })
    controls.pop("no_start_validation", None)
    controls.pop("v42_contract_coverage", None)
    controls["normalized_config"] = normalized_candidate

    source_bindings = load_json(RUN66 / "template-bindings.json")
    bindings = {
        "schema_version": 1,
        "status": "Run69 direct PowerShell scripts rendered from pinned Run66 templates; not parser-validated or executed",
        "source_run66_template_bindings_path": str((RUN66 / "template-bindings.json").resolve()),
        "source_run66_template_bindings_sha256": sha_file(RUN66 / "template-bindings.json"),
        "source_run66_templates": source_bindings.get("templates", {}),
        "run_phase_source_path": str((RUN66 / "run-phase.ps1").resolve()),
        "run_phase_source_sha256": sha_file(RUN66 / "run-phase.ps1"),
        "run_phase_rendered_path": str((RUN / "run-phase.ps1").resolve()),
        "run_phase_rendered_sha256": wrapper_sha,
        "run_id": RUN_ID, "side_id": SIDE_ID, "source_commit": source_head,
        "rendered_scripts": rendered_scripts,
        "parser_validation_performed": False, "scripts_invoked": False,
    }
    return controls, bindings


def _make_matrix(base: dict[str, Any], phase: dict[str, Any], build: dict[str, Any],
                 build_receipt: Path, wrapper_sha: str, wrapper_bytes: int,
                 controls_sha: str, bindings_sha: str, preparation_source_sha: str) -> dict[str, Any]:
    matrix = copy.deepcopy(base)
    matrix.pop("upstream", None)
    matrix["run_id"] = "20261011-69-prefill-first-nonfinite"
    matrix["status"] = "prepared_probe69_review_pending_no_execution"
    matrix["prepared_at_utc"] = dt.datetime.now(dt.timezone.utc).isoformat().replace("+00:00", "Z")
    matrix["hetero"] = {
        "release": "v0.1.42", "source_path": str(SOURCE), "source_commit": SOURCE_HEAD,
        "source_tree": build["source_after"]["tree"], "binary_path": build["binary"]["path"],
        "binary_sha256": build["binary"]["sha256"], "binary_bytes": build["binary"]["size_bytes"],
        "binary_identity_status": "root build receipt bound; fresh launch-time identity recheck required",
        "build_receipt": str(build_receipt.resolve()), "build_receipt_sha256": sha_file(build_receipt),
        "cmake_probe_option": "STRATA_PREFILL_FINITE_PROBE=ON",
    }
    matrix["quality_pair_order"] = [SIDE_ID]
    matrix["sides"] = [phase["side"]]
    matrix["quality_recipe"] = copy.deepcopy(base["quality_recipe"])
    matrix["quality_recipe"].update({
        "prompt_ids": list(PROMPT_IDS), "manifest_prompt_ids": list(MANIFEST_IDS),
        "warmups_per_prompt": 1, "formals_per_prompt": 3,
        "strict_quality_stop_first": True, "performance_phase": False,
        "scope": "five Run66 short prompts followed by long_1k; no later long prompts",
    })
    matrix["shared_server_bridge"] = copy.deepcopy(base["shared_server_bridge"])
    matrix["shared_server_bridge"]["used_by_both_sides"] = False
    matrix["root_authorization"] = None
    matrix["authorization_file_status"] = "not created; fresh Run69 authorization required after root review"
    matrix["execution_enabled"] = False
    matrix["execution_mode"] = "direct_powershell_controllers"
    matrix["controller_wrapper"] = {
        "path": str((RUN / "run-phase.ps1").resolve()), "sha256": wrapper_sha,
        "bytes": wrapper_bytes, "owner": "Run69 preparation", "mutated_by_pool_coordinator": False,
    }
    matrix["template_bindings"] = "template-bindings.json"
    matrix["template_bindings_sha256"] = bindings_sha
    matrix["controls_normalization_sha256"] = controls_sha
    matrix["only_control_differences"] = [
        "E69 source/binary identity", "STRATA_PREFILL_FINITE_PROBE=ON compile option",
        "STRATA_PREFILL_FINITE_PROBE=1 and STRATA_DIAG_ZERO_LOGITS=1 runtime diagnostics",
        "run-local executable/log/evidence paths",
    ]
    matrix["probe69_diagnostic_contract"] = {
        "compile_option": "STRATA_PREFILL_FINITE_PROBE=ON",
        "runtime_environment": copy.deepcopy(DIAGNOSTIC_ENV),
        "invalid_or_unknown_probe_records_fail_closed": True,
        "normal_begin_or_checkpoint_with_null_first_bad_is_not_fatal": True,
        "non_null_first_bad_or_zero_logits_is_fatal_after_current_prompt_is_preserved": True,
        "partial_log_line_cursor_advances_only_after_lf": True,
    }
    matrix["source_run66_history_reference"] = {
        "matrix_path": str((RUN66 / "matrix.json").resolve()),
        "history_arm_id": "upstream42",
        "history_path": str(run66_failure_history_path(base).resolve()),
        "root_authorization_03_path": str((RUN66 / "root-authorization-03.json").resolve()),
        "copied_runtime_or_request_outputs": False,
    }
    matrix["preparation_source"] = {"path": str(Path(__file__).resolve()), "sha256": preparation_source_sha}
    matrix["fresh_input_identity_snapshot"] = {
        "source": str((RUN66 / "fresh-input-identities-01.json").resolve()),
        "status": "prior-reference-only; fresh launch-time identity recheck required",
    }
    matrix["no_model_api_device_or_build_action"] = True
    return matrix


def _write_preparation() -> dict[str, Any]:
    phase_dir = RUN / "arms/probe69/quality-on"
    rendered_dir = RUN / "rendered-scripts/probe69"
    output_files = [
        RUN / "matrix.json", RUN / "run-phase.ps1", RUN / "template-bindings.json",
        RUN / "controls-normalization.json", RUN / "preparation.json", RUN / "preparation-receipt.json",
        phase_dir / "config.json", phase_dir / "identity.json", phase_dir / "provenance.json",
        phase_dir / "fresh-identities.json", phase_dir / "preparation.json",
        *(rendered_dir / name for name in ("launch.ps1", "startup-watch.ps1", "quality.ps1", "cleanup.ps1")),
    ]
    existing = [str(path) for path in output_files if path.exists()]
    if phase_dir.exists() or rendered_dir.exists() or existing:
        raise FileExistsError("Run69 output exists; refusing overwrite: " + ", ".join(existing or [str(phase_dir if phase_dir.exists() else rendered_dir)]))

    _build_auth, build, build_receipt = _source_tree_and_build_receipt()
    run66_matrix = load_json(RUN66 / "matrix.json")
    if run66_matrix.get("execution_mode") != "direct_powershell_controllers" or run66_matrix.get("root_authorization") is not None:
        raise ValueError("Run66 must remain the direct-PS, matrix-unauthorized preparation source")
    base_side = next((side for side in run66_matrix["sides"] if side.get("id") == RUN66_SIDE_ID), None)
    if base_side is None:
        raise ValueError("Run66 hetero42 quality side is missing")
    base_phase = RUN66 / base_side["arm_path"]
    # Contracts come from prepared hetero42; only the recorded failure history is upstream42.
    history_path = run66_failure_history_path(run66_matrix)
    history = load_json(history_path)
    validate_run66_failure_history(history)
    if tuple(run66_matrix["quality_recipe"]["prompt_ids"]) != MANIFEST_IDS:
        raise ValueError("Run66 quality manifest contract is not the pinned nine-prompt list")

    wrapper_source = (RUN66 / "run-phase.ps1").read_text(encoding="utf-8-sig")
    wrapper = _render_run_phase(wrapper_source)
    wrapper_raw = wrapper.encode("utf-8")
    wrapper_sha = sha_bytes(wrapper_raw)
    phase_dir_string = str(phase_dir.resolve())
    script_texts, script_bindings = render_phase_scripts(phase_dir, rendered_dir, base_side["model_name"])
    phase = _build_one_side_contract(base_side, build, build_receipt, phase_dir,
                                     script_bindings, wrapper_sha, len(wrapper_raw),
                                     sha_file(Path(__file__)))
    controls, bindings = _build_control_contract(base_side, phase, build, SOURCE_HEAD, wrapper_sha,
                                                  phase["side"]["rendered_scripts"])
    controls_raw, bindings_raw = _json_bytes(controls), _json_bytes(bindings)
    controls_sha, bindings_sha = sha_bytes(controls_raw), sha_bytes(bindings_raw)
    matrix = _make_matrix(run66_matrix, phase, build, build_receipt, wrapper_sha, len(wrapper_raw),
                          controls_sha, bindings_sha, sha_file(Path(__file__)))
    matrix_raw = _json_bytes(matrix)
    prep_summary = {
        "schema_version": 1, "status": "prepared_probe69_pending_root_review_not_launched",
        "run_id": matrix["run_id"], "side_id": SIDE_ID,
        "source_head": SOURCE_HEAD, "source_tree": build["source_after"]["tree"],
        "build_receipt": str(build_receipt.resolve()),
        "build_receipt_sha256": sha_file(build_receipt),
        "binary_path": build["binary"]["path"], "binary_sha256": build["binary"]["sha256"],
        "binary_bytes": build["binary"]["size_bytes"],
        "matrix_sha256": sha_bytes(matrix_raw), "template_bindings_sha256": bindings_sha,
        "controls_normalization_sha256": controls_sha,
        "quality_prompt_ids": list(PROMPT_IDS), "manifest_prompt_ids": list(MANIFEST_IDS),
        "warmups_per_prompt": 1, "formal_per_prompt": 3, "request_output_leaves_created": 0,
        "model_or_runtime_assets_copied": False, "authorization_present": False,
        "scripts_invoked": False, "no_start_cli_validation_performed": False,
    }
    prep_summary_raw = _json_bytes(prep_summary)
    prep_receipt = {
        "schema_version": 1, "status": "prepared_not_executed_root_review_required",
        "run_id": matrix["run_id"], "matrix_sha256": sha_bytes(matrix_raw),
        "preparation_sha256": sha_bytes(prep_summary_raw), "template_bindings_sha256": bindings_sha,
        "controls_normalization_sha256": controls_sha, "rendered_script_count": 4,
        "request_output_leaves_created": 0, "authorization_present": False,
        "script_or_lifecycle_execution": False, "no_model_payload_read_or_hashed": True,
    }
    output_map: dict[Path, bytes] = {
        phase_dir / "config.json": phase["config_raw"],
        phase_dir / "identity.json": phase["identity_raw"],
        phase_dir / "provenance.json": phase["provenance_raw"],
        phase_dir / "fresh-identities.json": phase["fresh_raw"],
        phase_dir / "preparation.json": phase["preparation_raw"],
        RUN / "matrix.json": matrix_raw,
        RUN / "run-phase.ps1": wrapper_raw,
        RUN / "template-bindings.json": bindings_raw,
        RUN / "controls-normalization.json": controls_raw,
        RUN / "preparation.json": prep_summary_raw,
        RUN / "preparation-receipt.json": _json_bytes(prep_receipt),
    }
    for key, script in script_texts.items():
        file_name = "startup-watch.ps1" if key == "watch" else key + ".ps1"
        output_map[rendered_dir / file_name] = script.encode("utf-8")
    preexisting = [str(path) for path in output_map if path.exists()]
    if preexisting:
        raise FileExistsError("Run69 exclusive output exists; refusing overwrite: " + ", ".join(preexisting))

    phase_dir.mkdir(parents=True, exist_ok=False)
    for subdir in ("logs", "resource", "requests", "quality", "cleanup"):
        (phase_dir / subdir).mkdir()
    rendered_dir.mkdir(parents=True, exist_ok=False)
    for path, raw in output_map.items():
        write_new(path, raw)
    request_leaves = [item for item in (phase_dir / "requests").iterdir() if item.is_dir()]
    if request_leaves:
        raise RuntimeError("Run69 preparation unexpectedly created per-prompt request leaves")
    return {
        "status": "prepared", "run_id": matrix["run_id"],
        "matrix_path": str((RUN / "matrix.json").resolve()),
        "matrix_sha256": sha_file(RUN / "matrix.json"),
        "request_output_leaves_created": len(request_leaves),
        "authorization_created": False, "scripts_invoked": False,
    }


def main() -> int:
    result = _write_preparation()
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
