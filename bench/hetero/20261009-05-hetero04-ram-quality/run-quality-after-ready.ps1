param(
    [Parameter(Mandatory = $true)][int]$LauncherPid,
    [Parameter(Mandatory = $true)][int]$SamplerPid,
    [Parameter(Mandatory = $true)][switch]$RootReadyConfirmed
)

$ErrorActionPreference = 'Stop'
if (-not $RootReadyConfirmed) { throw 'Wait for root to confirm this run is ready before invoking the quality sequence.' }

$repo = 'C:\Users\DC\Documents\ChatGPT\Strata-Hetero'
$run = 'C:\Users\DC\Documents\ChatGPT\Strata-Hetero\bench\hetero\20261009-05-hetero04-ram-quality'
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
    [pscustomobject]@{id='smoke_three_key_retrieval'; local_chat_tokens=114; max_tokens=128},
    [pscustomobject]@{id='long_1k'; local_chat_tokens=1036; max_tokens=128},
    [pscustomobject]@{id='long_4k'; local_chat_tokens=4109; max_tokens=128},
    [pscustomobject]@{id='long_16k'; local_chat_tokens=16396; max_tokens=128},
    [pscustomobject]@{id='long_30k7'; local_chat_tokens=30712; max_tokens=128}
)

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
$null = Assert-ReadyAndIdle 'before first prompt'
$null = Assert-ResourceGate 'before first prompt'
foreach ($prompt in $prompts) {
    $null = Assert-ReadyAndIdle "before $($prompt.id)"
    $null = Assert-ResourceGate "before $($prompt.id)"
    $promptFile = Join-Path $repo "bench\hetero\20261009-06-quality-prompts\prompts\$($prompt.id).prompt.txt"
    $requestDir = Join-Path (Join-Path $run 'requests') $prompt.id
    $reportPath = Join-Path (Join-Path $run 'quality') "h5-$($prompt.id)-quality.json"
    if (Test-Path -LiteralPath $requestDir) { throw "Prompt request directory already exists; preserving it: $requestDir" }
    if (Test-Path -LiteralPath $reportPath) { throw "Quality report already exists; preserving it: $reportPath" }
    $runnerStdout = Join-Path (Join-Path $run 'logs') "h5-$($prompt.id).stdout.log"
    $runnerStderr = Join-Path (Join-Path $run 'logs') "h5-$($prompt.id).stderr.log"
    if ((Test-Path -LiteralPath $runnerStdout) -or (Test-Path -LiteralPath $runnerStderr)) {
        throw "$($prompt.id): runner log already exists; refusing overwrite."
    }
    $runnerArguments = @((Join-Path $repo 'tools\hetero_bench.py'), '--prompt-file', $promptFile,
        '--output', (Join-Path $run 'requests'), '--run-id', $prompt.id, '--repeats', '3', '--warmup', '1',
        '--max-tokens', [string]$prompt.max_tokens, '--concurrency', '1', '--seed', '42',
        '--reasoning-effort', 'none', '--identity-json', $identityPath, '--base-url', "http://127.0.0.1:$port")
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
    [pscustomobject]@{prompt_id=$prompt.id;max_tokens=$prompt.max_tokens;formal=3;quality='pass';actual_sse_ids_valid=$true;raw_id_hashes=$rawIds.id_hashes} | ConvertTo-Json -Compress
}

# These are strict offline comparisons. Both baseline paths are absolute and refer to complete nine-prompt runs.
& $python (Join-Path $repo 'tools\hetero_compare_quality.py') --baseline-run $baselineH4 --candidate-run $run `
    --manifest $manifest --output (Join-Path $run 'quality\h5-vs-h4-full.json')
$h4CompareExit = $LASTEXITCODE
& $python (Join-Path $repo 'tools\hetero_compare_quality.py') --baseline-run $baselineUpstream03 --candidate-run $run `
    --manifest $manifest --output (Join-Path $run 'quality\h5-vs-upstream03-full.json')
$upstreamCompareExit = $LASTEXITCODE
if ($h4CompareExit -ne 0 -or $upstreamCompareExit -ne 0) {
    throw 'One or both strict comparisons did not accept; preserve evidence and leave cleanup for root authorization.'
}
Write-Output 'H5 quality sequence and both strict comparisons accepted. Model/server cleanup is not authorized by this command file.'
