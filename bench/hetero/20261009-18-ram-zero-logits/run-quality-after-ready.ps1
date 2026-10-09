param(
    [Parameter(Mandatory = $true)][int]$LauncherPid,
    [Parameter(Mandatory = $true)][int]$SamplerPid,
    [Parameter(Mandatory = $true)][switch]$RootReadyConfirmed
)

$ErrorActionPreference = 'Stop'
if (-not $RootReadyConfirmed) { throw 'Wait for root to confirm this run is ready before invoking the quality sequence.' }

$repo = 'C:\Users\DC\Documents\ChatGPT\Strata-Hetero'
$run = 'C:\Users\DC\Documents\ChatGPT\Strata-Hetero\bench\hetero\20261009-18-ram-zero-logits'
$manifest = 'C:\Users\DC\Documents\ChatGPT\Strata-Hetero\bench\hetero\20261009-06-quality-prompts\manifest.json'
$baselineH4 = 'C:\Users\DC\Documents\ChatGPT\Strata-Hetero\bench\hetero\20261009-04-hetero04-direct-quality'
$baselineUpstream03 = 'C:\Users\DC\Documents\ChatGPT\Strata-Hetero\bench\hetero\20261009-03-upstream04-quality'
$python = 'E:\Strata-Hetero-data\venv\Scripts\python.exe'
$port = 8081
$expectedModel = 'qwen3.8-flash-next-unsloth-ud-q4_k_xl-hetero'
$prompts = @(
    [pscustomobject]@{id='smoke_arithmetic'; local_chat_tokens=27; max_tokens=128},
    [pscustomobject]@{id='smoke_unicode'; local_chat_tokens=32; max_tokens=128},
    [pscustomobject]@{id='smoke_json_arithmetic'; local_chat_tokens=50; max_tokens=128},
    [pscustomobject]@{id='smoke_python_function'; local_chat_tokens=39; max_tokens=256},
    [pscustomobject]@{id='smoke_three_key_retrieval'; local_chat_tokens=114; max_tokens=128}
)

$diagLog = Join-Path $run 'logs\engine.log'
function Get-ZeroLogitsDiagnosticSlice([long]$StartOffset) {
    if (-not (Test-Path -LiteralPath $diagLog)) {
        return [pscustomobject]@{start_offset=0; end_offset=0; lines=@()}
    }
    $stream = [IO.File]::Open($diagLog, [IO.FileMode]::Open, [IO.FileAccess]::Read, [IO.FileShare]::ReadWrite)
    try {
        $end = [long]$stream.Length
        $start = [Math]::Min($StartOffset, $end)
        [void]$stream.Seek($start, [IO.SeekOrigin]::Begin)
        $reader = [IO.StreamReader]::new($stream, [Text.Encoding]::UTF8, $true, 4096, $true)
        $text = $reader.ReadToEnd()
        $reader.Dispose()
        $lines = @([regex]::Split($text, "\r?\n") | Where-Object { $_.Contains('strata zero-logits diag:') })
        return [pscustomobject]@{start_offset=$start; end_offset=$end; lines=$lines}
    } finally {
        $stream.Dispose()
    }
}

function Write-ZeroLogitsDiagnosticEvent([string]$Name, [string]$PromptId, [int]$MaxTokens,
                                         [long]$StartOffset, $Slice, [bool]$RequestCompleted,
                                         [int]$RunnerExit) {
    $path = Join-Path (Join-Path $run 'quality') $Name
    if (Test-Path -LiteralPath $path) { throw "Diagnostic event already exists; preserving it: $path" }
    $event = [ordered]@{
        schema_version = 1
        status = 'diagnostic_observed'
        observed_utc = [DateTime]::UtcNow.ToString('o')
        prompt_id = $PromptId
        max_tokens = $MaxTokens
        engine_log = $diagLog
        log_start_offset = $StartOffset
        log_end_offset = $Slice.end_offset
        diagnostic_lines = @($Slice.lines)
        current_request_completed = $RequestCompleted
        runner_exit_code = $RunnerExit
        action = 'Preserve the current request/SSE and finish offline classification; send no subsequent prompt.'
        client_terminated_for_diagnostic = $false
    }
    $json = $event | ConvertTo-Json -Depth 6
    $bytes = [Text.UTF8Encoding]::new($false).GetBytes($json + "`n")
    $out = [IO.File]::Open($path, [IO.FileMode]::CreateNew, [IO.FileAccess]::Write, [IO.FileShare]::Read)
    try { $out.Write($bytes, 0, $bytes.Length); $out.Flush($true) } finally { $out.Dispose() }
    return $path
}
function Assert-ResourceGate([string]$When) {
    $samplePath = Join-Path $run 'resource\samples.jsonl'
    $line = Get-Content -LiteralPath $samplePath -Tail 1
    if (-not $line) { throw "${When}: resource sampler has no sample yet." }
    $sampleDoc = [System.Text.Json.JsonDocument]::Parse([string]$line)
    $sampleAt = [DateTimeOffset]::Parse($sampleDoc.RootElement.GetProperty('observed_at_utc').GetString())
    $sample = $line | ConvertFrom-Json
    if ($sample.owner_pid -ne $LauncherPid) { throw "${When}: resource sample owner PID differs from the confirmed H5 launcher." }
    $sampleAge = ([DateTimeOffset]::UtcNow - $sampleAt).TotalSeconds
    if ($sampleAge -lt -2 -or $sampleAge -gt 5) { throw "${When}: latest resource sample is stale or from the future." }
    if ($sample.ram_gate.status -ne 'pass' -or $null -eq $sample.ram_gate.available_gib -or
        [double]$sample.ram_gate.available_gib -lt 12) {
        throw "${When}: physical RAM gate failed or is unknown; preserve current outputs and stop." 
    }
    if ($sample.commit_gate.status -ne 'pass' -or $null -eq $sample.commit_gate.available_gib -or
        [double]$sample.commit_gate.available_gib -lt 4) {
        throw "${When}: system commit gate failed or is unknown; preserve current outputs and stop." 
    }
    if (@($sample.alerts).Count -gt 0 -or @($sample.errors).Count -gt 0) {
        throw "${When}: resource sampler reported alerts/errors; preserve current outputs and stop." 
    }
    return $sample
}

function Assert-ReadyAndIdle([string]$When) {
    $server = Get-CimInstance Win32_Process | Where-Object {
        $_.ParentProcessId -eq $LauncherPid -and $_.Name -eq 'python.exe' -and
        $_.CommandLine.Contains((Join-Path $run 'config.json'))
    } | Select-Object -First 1
    if (-not $server) { throw "${When}: H5 server child with the exact run config is absent." }
    $listeners = @(Get-NetTCPConnection -LocalPort $port -State Listen -ErrorAction SilentlyContinue)
    if ($listeners.Count -ne 1 -or $listeners[0].OwningProcess -ne $server.ProcessId -or
        $listeners[0].LocalAddress -ne '127.0.0.1') {
        throw "${When}: loopback listener is missing or is not owned by the H5 server child." 
    }
    $status = Invoke-RestMethod -Uri "http://127.0.0.1:$port/v1/status" -Method Get -TimeoutSec 5
    if ($status.model -ne $expectedModel -or $status.engine -ne '0.1.40.4' -or $status.loaded -ne $true) {
        throw "${When}: live model/engine identity or loaded state is invalid." 
    }
    if ($status.activity.in_flight -ne 0) { throw "${When}: another request is in flight; no new prompt will be submitted." }
    $engineLog = Get-Content -LiteralPath (Join-Path $run 'logs\engine.log') -Raw
    foreach ($marker in @('R4 hit path ON', 'resident RAM mode: 20.00 GiB of experts in RAM',
                          'the file tier reads through the file cache (STRATA_UNBUFFERED_LOAD=0)',
                          'PLE table locked in RAM', 'session is up (engine 0.1.40.4)')) {
        if (-not $engineLog.Contains($marker)) { throw "${When}: required strict-RAM20/buffered startup evidence is missing: $marker" }
    }
    return [pscustomobject]@{server_pid=$server.ProcessId;status=$status}
}

# This file only runs after root has launched H5 and supplied both owner PIDs plus the ready confirmation.
$processPath = Join-Path $run 'process.json'
$samplerProcessPath = Join-Path $run 'sampler-process.json'
$provenancePath = Join-Path $run 'provenance.json'
$identityPath = Join-Path $run 'identity.json'
$configPath = Join-Path $run 'config.json'
$process = Get-Content -LiteralPath $processPath -Raw | ConvertFrom-Json
$samplerProcess = Get-Content -LiteralPath $samplerProcessPath -Raw | ConvertFrom-Json
$provenance = Get-Content -LiteralPath $provenancePath -Raw | ConvertFrom-Json
$identity = Get-Content -LiteralPath $identityPath -Raw | ConvertFrom-Json
$config = Get-Content -LiteralPath $configPath -Raw | ConvertFrom-Json
if ($process.launcher_pid -ne $LauncherPid -or $samplerProcess.sampler_launcher_pid -ne $SamplerPid) {
    throw 'Supplied PIDs do not match the run-owned process receipts.'
}
if (-not $provenance.launch_admission -or $provenance.launch_admission.status -ne 'pass') {
    throw 'H5 provenance does not bind a passing fresh launch admission.'
}
if ((Get-FileHash -Algorithm SHA256 -LiteralPath (Join-Path $run $provenance.launch_admission.receipt)).Hash.ToLowerInvariant() -ne
    $provenance.launch_admission.receipt_sha256.ToLowerInvariant()) { throw 'H5 launch admission receipt hash mismatch.' }
if ((Get-FileHash -Algorithm SHA256 -LiteralPath (Join-Path $run $provenance.launch_admission.launch_process_receipt)).Hash.ToLowerInvariant() -ne
    $provenance.launch_admission.process_receipt_sha256.ToLowerInvariant()) { throw 'H5 launch process receipt hash mismatch.' }
$launcher = Get-CimInstance Win32_Process -Filter "ProcessId = $LauncherPid"
$sampler = Get-CimInstance Win32_Process -Filter "ProcessId = $SamplerPid"
$server = Get-CimInstance Win32_Process | Where-Object {
    $_.ParentProcessId -eq $LauncherPid -and $_.Name -eq 'python.exe' -and $_.CommandLine.Contains($configPath)
} | Select-Object -First 1
$engine = if ($server) {
    Get-CimInstance Win32_Process | Where-Object {
        $_.ParentProcessId -eq $server.ProcessId -and $_.ExecutablePath -eq $identity.engine.path
    } | Select-Object -First 1
}
$processJson = [System.Text.Json.JsonDocument]::Parse([System.IO.File]::ReadAllText($processPath))
$launchUtc = [DateTimeOffset]::Parse($processJson.RootElement.GetProperty('start_utc').GetString())
$launcherDeltaMs = [Math]::Abs((([DateTimeOffset]$launcher.CreationDate).ToUniversalTime() - $launchUtc.ToUniversalTime()).TotalMilliseconds)
$expectedLauncherCommand = '"' + $process.python + '" ' + ($process.argv -join ' ')
$expectedEngineCommand = $identity.engine.path + ' --serve ' + ($config.args -join ' ')
if (-not $launcher -or $launcher.ExecutablePath -ne $process.python -or $launcherDeltaMs -ge 1 -or
    $launcher.CommandLine -ne $expectedLauncherCommand) { throw 'H5 launcher PID/create-time/executable/command does not match process.json.' }
if (-not $sampler -or -not $sampler.CommandLine.Contains('hetero_resources.py') -or
    -not $sampler.CommandLine.Contains("--pid $LauncherPid")) { throw 'H5 resource sampler does not match sampler-process.json.' }
if (-not $engine -or $engine.ParentProcessId -ne $server.ProcessId -or $engine.CommandLine -ne $expectedEngineCommand) {
    throw 'H5 engine child parent/command does not match the saved run config.'
}
if ($config.args -notcontains '--ple-io' -or $config.args[([array]::IndexOf($config.args, '--ple-io') + 1)] -ne 'ram' -or
    $config.args[([array]::IndexOf($config.args, '--resident-budget-gib') + 1)] -ne '20' -or
    $config.env.STRATA_UNBUFFERED_LOAD -ne '0') { throw 'H5 config does not identify PLE RAM + resident budget 20 + buffered file reads.' }
if ((Get-FileHash -Algorithm SHA256 -LiteralPath $config.exe).Hash.ToLowerInvariant() -ne
    $identity.engine.sha256.ToLowerInvariant()) { throw 'H5 engine binary does not match identity.json.' }
if (-not (Test-Path -LiteralPath $baselineH4 -PathType Container) -or
    -not (Test-Path -LiteralPath $baselineUpstream03 -PathType Container) -or
    -not (Test-Path -LiteralPath $manifest -PathType Leaf)) { throw 'A required absolute baseline or manifest path is missing.' }
if ((Get-FileHash -Algorithm SHA256 -LiteralPath $manifest).Hash.ToLowerInvariant() -ne
    'ac749f67d3205760785b7f8d0cf67b813f9759944cfd033ee76003b2d8fdfbbd') {
    throw 'Quality manifest SHA-256 differs from the prepared contract.'
}

# Readiness is asserted at entry; repeat the resource and idle checks around each prompt.
$lastDiagnosticLogOffset = 0L
$null = Assert-ReadyAndIdle 'before first prompt'
$null = Assert-ResourceGate 'before first prompt'
foreach ($prompt in $prompts) {
    $prePromptDiagnostic = Get-ZeroLogitsDiagnosticSlice $lastDiagnosticLogOffset
    $lastDiagnosticLogOffset = $prePromptDiagnostic.end_offset
    if (@($prePromptDiagnostic.lines).Count -gt 0) {
        $null = Write-ZeroLogitsDiagnosticEvent "zero-logits-before-$($prompt.id).json" $prompt.id $prompt.max_tokens `
            $prePromptDiagnostic.start_offset $prePromptDiagnostic $false -1
        throw "$($prompt.id): zero-logits diagnostic was recorded between requests; no new request was sent."
    }
    $null = Assert-ReadyAndIdle "before $($prompt.id)"
    $null = Assert-ResourceGate "before $($prompt.id)"
    $promptFile = Join-Path $repo "bench\hetero\20261009-06-quality-prompts\prompts\$($prompt.id).prompt.txt"
    $requestDir = Join-Path (Join-Path $run 'requests') $prompt.id
    $reportPath = Join-Path (Join-Path $run 'quality') "ram-zero-logits-$($prompt.id)-quality.json"
    if (Test-Path -LiteralPath $requestDir) { throw "Prompt request directory already exists; preserving it: $requestDir" }
    if (Test-Path -LiteralPath $reportPath) { throw "Quality report already exists; preserving it: $reportPath" }
    $runnerStdout = Join-Path (Join-Path $run 'logs') "ram-zero-logits-$($prompt.id).stdout.log"
    $runnerStderr = Join-Path (Join-Path $run 'logs') "ram-zero-logits-$($prompt.id).stderr.log"
    if ((Test-Path -LiteralPath $runnerStdout) -or (Test-Path -LiteralPath $runnerStderr)) {
        throw "$($prompt.id): runner log already exists; refusing overwrite."
    }
    $runnerArguments = @((Join-Path $repo 'tools\hetero_bench.py'), '--prompt-file', $promptFile,
        '--output', (Join-Path $run 'requests'), '--run-id', $prompt.id, '--repeats', '3', '--warmup', '1',
        '--max-tokens', [string]$prompt.max_tokens, '--concurrency', '1', '--seed', '42',
        '--reasoning-effort', 'none', '--identity-json', $identityPath, '--base-url', "http://127.0.0.1:$port")
    $diagnosticRequestStartOffset = $lastDiagnosticLogOffset
    $runner = Start-Process -FilePath $python -ArgumentList $runnerArguments -WorkingDirectory $repo `
        -WindowStyle Hidden -PassThru -RedirectStandardOutput $runnerStdout -RedirectStandardError $runnerStderr
    try {
        while (-not $runner.HasExited) {
            $null = Assert-ResourceGate "during $($prompt.id)"
            $live = Invoke-RestMethod -Uri "http://127.0.0.1:$port/v1/status" -Method Get -TimeoutSec 5
            if ($live.model -ne $expectedModel -or $live.engine -ne '0.1.40.4' -or $live.loaded -ne $true -or
                $live.activity.in_flight -gt 1) { throw "$($prompt.id): live service identity/state changed during task." }
            Start-Sleep -Seconds 1
            $runner.Refresh()
        }
    } catch {
        if (-not $runner.HasExited) {
            Stop-Process -Id $runner.Id
            Wait-Process -Id $runner.Id -Timeout 10 -ErrorAction SilentlyContinue
        }
        throw
    }
    $runnerExit = $runner.ExitCode
    Get-Content -LiteralPath $runnerStdout
    Get-Content -LiteralPath $runnerStderr -ErrorAction SilentlyContinue
    $diagnosticForPrompt = Get-ZeroLogitsDiagnosticSlice $diagnosticRequestStartOffset
    $lastDiagnosticLogOffset = $diagnosticForPrompt.end_offset
    $diagnosticObserved = @($diagnosticForPrompt.lines).Count -gt 0
    if ($diagnosticObserved) {
        $null = Write-ZeroLogitsDiagnosticEvent "zero-logits-$($prompt.id).json" $prompt.id $prompt.max_tokens `
            $diagnosticRequestStartOffset $diagnosticForPrompt $runner.HasExited $runnerExit
    }
    if (-not (Test-Path -LiteralPath $requestDir -PathType Container)) { throw "$($prompt.id): runner created no request artifacts (exit $runnerExit)." }
    & $python (Join-Path $repo 'tools\hetero_quality.py') --responses-dir (Join-Path $run 'requests') `
        --manifest $manifest --report-out $reportPath
    if ($LASTEXITCODE -ne 0) { throw "$($prompt.id): offline quality report failed; preserve outputs and stop." }
    $metadata = Get-Content -LiteralPath (Join-Path $requestDir 'metadata.json') -Raw | ConvertFrom-Json
    $requests = Get-Content -LiteralPath (Join-Path $requestDir 'requests.json') -Raw | ConvertFrom-Json
    $report = Get-Content -LiteralPath $reportPath -Raw | ConvertFrom-Json
    $rows = @($report.results | Where-Object { $_.prompt_id -eq $prompt.id })
    if ($runnerExit -ne 0 -or $metadata.max_tokens -ne $prompt.max_tokens -or $metadata.repeats -ne 3 -or
        $metadata.warmup -ne 1 -or $metadata.seed -ne 42 -or $metadata.reasoning_effort -ne 'none' -or
        @($requests).Count -ne 3 -or @($rows).Count -ne 3 -or
        @($requests | Where-Object { $_.status -ne 'completed' -or $_.finish_reason -ne 'stop' -or
             $_.usage.prompt_tokens -ne $prompt.local_chat_tokens -or $_.timings.cache_n -ne 0 }).Count -gt 0 -or
        @($rows | Where-Object { $_.quality_status -ne 'pass' }).Count -gt 0) {
        throw "$($prompt.id): formal status/usage/cache/quality gate failed; outputs are preserved and longer prompts are stopped." 
    }
    $rawIdsCheck = @'
import json,sys
from pathlib import Path
sys.path.insert(0,str(Path.cwd()/"tools"))
from hetero_compare_quality import extract_final_sse
folder=Path(sys.argv[1])
rows=[extract_final_sse(folder/f"formal-{i:04d}.sse.txt") for i in range(3)]
ok=all(x.get("valid") and x.get("actual_generated_token_ids") and x.get("include_stop") is True for x in rows)
print(json.dumps({"valid":ok,"count":len(rows),"id_hashes":[x.get("actual_generated_token_ids_sha256") for x in rows],"errors":[x.get("errors") for x in rows]}))
raise SystemExit(0 if ok else 1)
'@
    $rawIdsJson = & $python -B -X utf8 -c $rawIdsCheck $requestDir
    if ($LASTEXITCODE -ne 0) { throw "$($prompt.id): raw SSE engine token-ID proof failed; stop longer prompts." }
    $rawIds = ($rawIdsJson -join "`n") | ConvertFrom-Json
    if ($rawIds.count -ne 3) { throw "$($prompt.id): raw SSE formal count is not three." }
    $null = Assert-ResourceGate "after $($prompt.id)"
    $null = Assert-ReadyAndIdle "after $($prompt.id)"
    [pscustomobject]@{prompt_id=$prompt.id;max_tokens=$prompt.max_tokens;formal=3;quality='pass';actual_sse_ids_valid=$true;raw_id_hashes=$rawIds.id_hashes;zero_logits_diagnostic_observed=$diagnosticObserved} | ConvertTo-Json -Compress
    if ($diagnosticObserved) {
        throw "$($prompt.id): request, raw SSE token IDs and offline quality classification are preserved; diagnostic observed, so stop before the next prompt."
    }
}

# This H18 stage is intentionally limited to five short prompts. Full-run comparators
# require all nine prompts and are not valid for this partial candidate; long prompts
# require a separate root authorization and are never started here.
$stagePath = Join-Path (Join-Path $run 'quality') 'short-stage-result.json'
if (Test-Path -LiteralPath $stagePath) { throw "Short-stage receipt already exists; preserving it: $stagePath" }
$stage = [ordered]@{
    schema_version = 1
    status = 'five_shorts_complete'
    run = $run
    completed_utc = [DateTime]::UtcNow.ToString('o')
    prompts = @($prompts | ForEach-Object { [ordered]@{id=$_.id; max_tokens=$_.max_tokens; warmup=1; formal=3} })
    long_prompts_started = $false
    full_nine_prompt_comparisons_run = $false
    cleanup_authorized = $false
    note = 'Five short tasks only. Full comparisons and long prompts are deferred for root review and separate authorization.'
}
$stageBytes = [Text.UTF8Encoding]::new($false).GetBytes(($stage | ConvertTo-Json -Depth 5) + "`n")
$stageFile = [IO.File]::Open($stagePath, [IO.FileMode]::CreateNew, [IO.FileAccess]::Write, [IO.FileShare]::Read)
try { $stageFile.Write($stageBytes, 0, $stageBytes.Length); $stageFile.Flush($true) } finally { $stageFile.Dispose() }
Write-Output 'H18 five-short stage complete. Long prompts, full comparisons, and model/server cleanup were not started or authorized.'
