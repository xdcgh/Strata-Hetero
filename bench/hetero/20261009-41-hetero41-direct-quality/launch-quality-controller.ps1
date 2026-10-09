$ErrorActionPreference = 'Stop'
Set-StrictMode -Version 2.0
$repo = 'C:\Users\DC\Documents\ChatGPT\Strata-Hetero'
$run = Join-Path $repo 'bench\hetero\20261009-41-hetero41-direct-quality'
$launchScript = Join-Path $run 'launch-owned.ps1'
$watchScript = Join-Path $run 'startup-watch.ps1'
$qualityScript = Join-Path $run 'run-quality-after-ready.ps1'
$eventsPath = Join-Path $run 'quality\launch-quality-controller-events.jsonl'
$resultPath = Join-Path $run 'quality\launch-quality-controller-result.json'
$launchEventStream = [IO.File]::Open($eventsPath, [IO.FileMode]::CreateNew, [IO.FileAccess]::Write, [IO.FileShare]::Read)
function Write-Event($Value) {
    $line = (ConvertTo-Json -InputObject ([ordered]@{utc=[DateTime]::UtcNow.ToString('o');data=$Value}) -Compress) + "`n"
    $bytes = [Text.UTF8Encoding]::new($false).GetBytes($line)
    $launchEventStream.Write($bytes, 0, $bytes.Length); $launchEventStream.Flush($true)
}
function Write-ExclusiveJson([string]$Path, $Value) {
    $bytes = [Text.UTF8Encoding]::new($false).GetBytes(($Value | ConvertTo-Json -Depth 14) + "`n")
    $stream = [IO.File]::Open($Path, [IO.FileMode]::CreateNew, [IO.FileAccess]::Write, [IO.FileShare]::Read)
    try { $stream.Write($bytes, 0, $bytes.Length); $stream.Flush($true) } finally { $stream.Dispose() }
}
function Exact-Process($Snapshot, [long]$ProcessId, [string]$Exe, [string]$CreateUtc, [long]$ParentPid, [string]$CommandContains) {
    $matches = @($Snapshot | Where-Object { $_.ProcessId -eq $ProcessId })
    if ($matches.Count -ne 1) { return $null }
    $candidate = $matches[0]
    if ($candidate.ExecutablePath -ne $Exe -or $candidate.CreationDate.ToUniversalTime().ToString('o') -ne $CreateUtc -or
        ($ParentPid -ge 0 -and $candidate.ParentProcessId -ne $ParentPid) -or
        ($CommandContains -and -not $candidate.CommandLine.Contains($CommandContains))) { return $null }
    return $candidate
}
$phase = 'preflight'
try {
    foreach ($path in @($launchScript, $watchScript, $qualityScript)) {
        if (-not (Test-Path -LiteralPath $path -PathType Leaf)) { throw "required coordinator stage missing: $path" }
    }
    foreach ($path in @((Join-Path $run 'process.json'), (Join-Path $run 'sampler-process.json'),
        (Join-Path $run 'admission.json'), (Join-Path $run 'identity-recheck.json'),
        (Join-Path $run 'resource\startup-watch-result.json'), (Join-Path $run 'quality\preflight-ready-binding.json'),
        (Join-Path $run 'quality\ready-binding.json'), (Join-Path $run 'quality\quality-progress.jsonl'),
        (Join-Path $run 'quality\F-nine-prompt-stage.json'))) {
        if (Test-Path -LiteralPath $path) { throw "run44 evidence path already exists: $path" }
    }
    Write-Event @{phase=$phase;status='passed';launch_attempts=1;quality_attempts=1}

    $phase = 'launch'
    $null = & $launchScript -Start
    $processPath = Join-Path $run 'process.json'
    if (-not (Test-Path -LiteralPath $processPath -PathType Leaf)) { throw 'launch returned without process receipt' }
    $process = Get-Content -LiteralPath $processPath -Raw | ConvertFrom-Json -DateKind String
    $sampler = Get-Content -LiteralPath (Join-Path $run 'sampler-process.json') -Raw | ConvertFrom-Json -DateKind String
    $admission = Get-Content -LiteralPath (Join-Path $run 'admission.json') -Raw | ConvertFrom-Json -DateKind String
    $identity = Get-Content -LiteralPath (Join-Path $run 'identity.json') -Raw | ConvertFrom-Json -DateKind String
    $config = Get-Content -LiteralPath (Join-Path $run 'config.json') -Raw | ConvertFrom-Json -DateKind String
    $provenance = Get-Content -LiteralPath (Join-Path $run 'provenance.json') -Raw | ConvertFrom-Json -DateKind String
    $readerPath = Join-Path $repo 'tools\hetero_jsonl_tail.ps1'
    $bridgePath = $provenance.server_bridge.path
    $sha = { param([string]$Path) (Get-FileHash -LiteralPath $Path -Algorithm SHA256).Hash.ToLowerInvariant() }
    if ($process.run_id -ne '20261009-41-hetero41-direct-quality' -or
        $process.engine_path -ne $config.exe -or $process.engine_sha256 -ne $identity.engine.sha256 -or
        $config.exe -ne $identity.engine.path -or $provenance.engine_sha256 -ne $identity.engine.sha256 -or
        $provenance.engine_cpp_sha -ne $identity.source.sha -or
        $process.server_file_sha256 -ne $provenance.server_file_sha256 -or
        (& $sha $config.exe) -ne $identity.engine.sha256 -or
        (& $sha $readerPath) -ne $provenance.jsonl_tail_reader_sha256 -or
        (& $sha $bridgePath) -ne $provenance.server_file_sha256) { throw 'post-launch binary/source/config/bridge/reader identity binding failed' }
    if ($config.env.STRATA_PREFILL_CPU_SHARE -ne '0' -or $config.env.STRATA_STAGE_PIN -ne '0' -or
        $config.args[([array]::IndexOf($config.args, '--resident-budget-gib') + 1)] -ne '20' -or
        $config.args[([array]::IndexOf($config.args, '--max-context') + 1)] -ne '32768' -or
        $config.args[([array]::IndexOf($config.args, '--kv') + 1)] -ne 'int8' -or
        $config.args[([array]::IndexOf($config.args, '--ple-io') + 1)] -ne 'direct') { throw 'matched F placement or v1.41 opt-out controls changed' }
    if ($admission.pass -ne $true -or $admission.status -ne 'pass' -or
        $admission.raw.windows_memory.available_bytes -lt 12GB -or
        $admission.raw.windows_memory.available_commit_bytes -lt 4GB -or
        $admission.chosen_gates.vram.pass -ne $true -or $admission.chosen_gates.vram.available_mib -lt 46000) {
        throw 'fresh launch admission did not pass 12/4 GiB and 46000 MiB VRAM'
    }
    Write-Event @{phase=$phase;status='launched_not_ready';launcher_pid=$process.launcher_pid;
        launcher_create_utc=$process.launcher_create_utc;server_child_pid=$process.actual_server_child_pid;
        server_child_create_utc=$process.actual_server_child_create_utc;sampler_launcher_pid=$sampler.sampler_launcher_pid;
        sampler_child_pid=$sampler.actual_child_pid;sampler_child_create_utc=$sampler.actual_child_create_utc;
        engine_path=$process.engine_path;engine_sha256=$process.engine_sha256;reader_sha256=$provenance.jsonl_tail_reader_sha256}

    $phase = 'startup_watch'
    $watchOutput = @(& $watchScript -Monitor -TimeoutSeconds 900)
    $watchJson = ($watchOutput | ForEach-Object { [string]$_ }) -join "`n"
    $watchResult = $watchJson | ConvertFrom-Json -DateKind String -ErrorAction Stop
    if ($watchResult.status -ne 'startup_ready' -or $watchResult.launcher_pid -ne $process.launcher_pid -or
        $watchResult.server_child_pid -ne $process.actual_server_child_pid -or
        $watchResult.sampler_child_pid -ne $sampler.actual_child_pid -or
        $watchResult.engine_exe -ne $identity.engine.path -or
        (([DateTimeOffset]::UtcNow - [DateTimeOffset]::Parse($watchResult.sample.observed_at_utc)).TotalSeconds -gt 5) -or
        $watchResult.sample.ram_gate.available_gib -lt 12 -or $watchResult.sample.commit_gate.available_gib -lt 4) {
        throw 'startup watcher did not return fresh exact-owner ready evidence'
    }
    $snapshot = @(Get-CimInstance Win32_Process)
    $launcher = Exact-Process $snapshot $process.launcher_pid $process.launcher_exe $process.launcher_create_utc -1 'serve\server.py'
    $server = Exact-Process $snapshot $process.actual_server_child_pid 'C:\Python314\python.exe' $process.actual_server_child_create_utc $process.launcher_pid 'serve\server.py'
    $samplerLauncher = Exact-Process $snapshot $sampler.sampler_launcher_pid $sampler.sampler_launcher_exe $sampler.sampler_launcher_create_utc -1 'hetero_resources.py'
    $samplerChild = Exact-Process $snapshot $sampler.actual_child_pid 'C:\Python314\python.exe' $sampler.actual_child_create_utc $sampler.sampler_launcher_pid 'hetero_resources.py'
    $engine = Exact-Process $snapshot $watchResult.engine_pid $identity.engine.path $watchResult.engine_create_utc $process.actual_server_child_pid ' --serve '
    if (-not $launcher -or -not $server -or -not $samplerLauncher -or -not $samplerChild -or -not $engine -or
        $engine.CommandLine -ne ($identity.engine.path + ' --serve ' + ($config.args -join ' '))) {
        throw 'live launcher/server/engine/sampler process snapshot no longer matches exact receipts'
    }
    Write-Event @{phase=$phase;status='startup_ready';launcher_pid=$launcher.ProcessId;server_child_pid=$server.ProcessId;
        engine_pid=$engine.ProcessId;engine_exe=$engine.ExecutablePath;engine_create_utc=$engine.CreationDate.ToUniversalTime().ToString('o');
        sampler_launcher_pid=$samplerLauncher.ProcessId;sampler_child_pid=$samplerChild.ProcessId;
        sample_age_seconds=([Math]::Round(([DateTimeOffset]::UtcNow-[DateTimeOffset]::Parse($watchResult.sample.observed_at_utc)).TotalSeconds,3));
        ram_gib=$watchResult.sample.ram_gate.available_gib;commit_gib=$watchResult.sample.commit_gate.available_gib}

    $phase = 'quality_checkready'
    $checkOutput = @(& $qualityScript -CheckReady -RootReadyConfirmed `
        -LauncherPid ([int]$process.launcher_pid) -SamplerPid ([int]$sampler.sampler_launcher_pid))
    $checkJson = ($checkOutput | ForEach-Object { [string]$_ }) -join "`n"
    $check = $checkJson | ConvertFrom-Json -DateKind String -ErrorAction Stop
    if ($check.status -ne 'live_quality_preflight_pass' -or $check.chat_requests_sent -ne 0 -or
        $check.model_requests -ne 0 -or $check.engine_pid -ne $engine.ProcessId -or -not $check.identity_bound -or
        -not (Test-Path -LiteralPath (Join-Path $run 'quality\preflight-ready-binding.json'))) {
        throw 'read-only CheckReady failed its zero-request exact identity gate'
    }
    Write-Event @{phase=$phase;status='passed';chat_requests_sent=0;model_requests=0;engine_pid=$engine.ProcessId;
        preflight_binding='quality\preflight-ready-binding.json'}

    $phase = 'quality_run'
    $qualityOutput = @(& $qualityScript -Run -RootReadyConfirmed `
        -LauncherPid ([int]$process.launcher_pid) -SamplerPid ([int]$sampler.sampler_launcher_pid))
    $stagePath = Join-Path $run 'quality\F-nine-prompt-stage.json'
    if (-not (Test-Path -LiteralPath $stagePath -PathType Leaf)) { throw 'quality controller returned without nine-prompt terminal receipt' }
    $stage = Get-Content -LiteralPath $stagePath -Raw | ConvertFrom-Json -DateKind String
    if ($stage.status -ne 'nine_prompt_matrix_complete' -or $stage.prompt_count -ne 9 -or
        @($stage.results | Where-Object { @($_.quality_statuses | Where-Object { $_ -ne 'pass' }).Count -gt 0 -or
            @($_.actual_token_ids_match_h4 | Where-Object { $_ -ne $true }).Count -gt 0 }).Count -gt 0) {
        throw 'quality result does not prove the complete fixed nine-task matrix and H4 IDs'
    }
    Write-Event @{phase=$phase;status='nine_prompt_matrix_complete';prompt_count=9;formal_requests=27;warmup_requests=9;
        sampler_stop_requested=$false;model_left_live_for_root_review=$true}
    $result = [ordered]@{schema_version=1;run_id='20261009-41-hetero41-direct-quality';
        status='nine_prompt_matrix_complete';completed_utc=[DateTime]::UtcNow.ToString('o');launch_attempts=1;
        startup_watch_status=$watchResult.status;checkready_status=$check.status;model_requests_before_quality=$check.model_requests;
        stage_status=$stage.status;prompt_count=$stage.prompt_count;formal_requests=27;warmup_requests=9;
        sampler_stop_requested=$false;model_left_live_for_root_review=$true}
    Write-ExclusiveJson $resultPath $result
    $result | ConvertTo-Json -Depth 8 -Compress
} catch {
    $failure = [ordered]@{schema_version=1;run_id='20261009-41-hetero41-direct-quality';
        status='coordinator_failed_stopped';phase=$phase;utc=[DateTime]::UtcNow.ToString('o');
        error_type=$_.Exception.GetType().FullName;error=$_.Exception.Message;
        quality_failure_receipt_exists=(Test-Path -LiteralPath (Join-Path $run 'quality\F-quality-failure.json'));
        quality_stage_exists=(Test-Path -LiteralPath (Join-Path $run 'quality\F-nine-prompt-stage.json'));
        sampler_stop_requested=$false;model_cleanup_requested=$false}
    if (-not (Test-Path -LiteralPath $resultPath)) { try { Write-ExclusiveJson $resultPath $failure } catch {} }
    try { Write-Event @{phase=$phase;status='failed';error=$_.Exception.Message} } catch {}
    throw
} finally {
    $launchEventStream.Dispose()
}
