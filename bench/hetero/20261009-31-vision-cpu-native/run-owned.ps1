$ErrorActionPreference = 'Stop'
$workspaceRoot = 'C:\Users\DC\Documents\ChatGPT\Strata-Hetero'
$benchDirectory = Join-Path $workspaceRoot 'bench\hetero\20261009-31-vision-cpu-native'
$compareDirectory = 'E:\Strata-Hetero-data\vision-compare\20261009-31-cpu-native'
$exportReceipt = 'E:\Strata-Hetero-data\vision-payloads\20261009-30-native-vision-export\export-receipt.json'
$comparePython = 'E:\Strata-Hetero-data\venv-intel\Scripts\python.exe'
$memoryCode = 'import json; from tools.hetero_resources import memory_snapshot; m,e=memory_snapshot(); print(json.dumps({"memory":m,"errors":e}))'
if (Test-Path -LiteralPath $compareDirectory) { throw 'Exclusive E comparison target already exists' }
foreach ($name in @('stdout.json','stderr.log','process.json','resource.jsonl','exit.json')) {
    if (Test-Path -LiteralPath (Join-Path $benchDirectory $name)) { throw "Exclusive bench artifact exists: $name" }
}
$defaultOutput = & $comparePython tools\hetero_vision_cpu_compare.py --export-receipt $exportReceipt
$defaultExit = $LASTEXITCODE
$defaultOutput | Set-Content -LiteralPath (Join-Path $benchDirectory 'default-contract.json') -Encoding utf8
if ($defaultExit -ne 0) { throw 'Default no-runtime contract failed' }
$memoryText = & $comparePython -c $memoryCode
$memoryExit = $LASTEXITCODE
$initial = $memoryText | ConvertFrom-Json -AsHashtable
if ($memoryExit -ne 0 -or $initial.errors.Count -gt 0 -or $initial.memory.physical_available_bytes -lt 12884901888 -or $initial.memory.commit_available_bytes -lt 4294967296) { throw 'Fresh 12/4 gate failed or unknown' }
$initial['observed_utc'] = [DateTime]::UtcNow.ToString('o')
$initial | ConvertTo-Json -Depth 7 | Set-Content -LiteralPath (Join-Path $benchDirectory 'preflight.json') -Encoding utf8
$checkoutHead = & git rev-parse HEAD
$gitExit = $LASTEXITCODE
if ($gitExit -ne 0) { throw 'Checkout identity unavailable' }
$sourcePaths = @('tools\hetero_vision_cpu_compare.py','tools\hetero_vision_qwen3vl.py','tools\hetero_vision_payload.py','tools\hetero_resources.py','bench\hetero\20261009-31-vision-cpu-native\run-owned.ps1',$comparePython,'C:\Python314\python.exe')
$sourceHashes = @($sourcePaths | ForEach-Object { [ordered]@{path=(Resolve-Path -LiteralPath $_).Path;sha256=(Get-FileHash -LiteralPath $_ -Algorithm SHA256).Hash.ToLower()} })
$env:PYTHONDONTWRITEBYTECODE = '1'
$env:OPENBLAS_NUM_THREADS = '1'
$env:MKL_NUM_THREADS = '1'
$env:OMP_NUM_THREADS = '1'
$env:CUDA_VISIBLE_DEVICES = '-1'
$compareArgs = @('-u','tools\hetero_vision_cpu_compare.py','--run','--export-receipt',$exportReceipt,'--output-dir',$compareDirectory)
$resourceStream = [System.IO.FileStream]::new((Join-Path $benchDirectory 'resource.jsonl'),[System.IO.FileMode]::CreateNew,[System.IO.FileAccess]::Write,[System.IO.FileShare]::Read)
$resourceWriter = [System.IO.StreamWriter]::new($resourceStream,[System.Text.UTF8Encoding]::new($false))
$compareProcess = $null
$stopReason = 'process_terminal'
$monitorClock = [System.Diagnostics.Stopwatch]::StartNew()
$previousSampleMs = $null
$nextSampleMs = 0
try {
    $launchUtc = [DateTime]::UtcNow.ToString('o')
    $compareProcess = Start-Process -FilePath $comparePython -ArgumentList $compareArgs -WorkingDirectory $workspaceRoot -WindowStyle Hidden -RedirectStandardOutput (Join-Path $benchDirectory 'stdout.json') -RedirectStandardError (Join-Path $benchDirectory 'stderr.log') -PassThru
    $creationUtc = $compareProcess.StartTime.ToUniversalTime().ToString('o')
    $processRecord = [ordered]@{schema_version=1;status='started';scope='Exact CPU native vision comparison only';launch_utc=$launchUtc;pid=$compareProcess.Id;process_create_utc=$creationUtc;python=$comparePython;argv=$compareArgs;checkout_head=[string]$checkoutHead;source_hashes=$sourceHashes;device='CPU';required_precision='f32';required_execution_mode='ACCURACY';warmups_per_image=1;formal_repeats_per_image=3;external_monitor_interval_seconds=1;external_monitor_source='PowerShell-owned process handle plus tools.hetero_resources.memory_snapshot';stop_scope='Only this exact Start-Process-owned handle';npu_authorized=$false;vision_api_authorized=$false}
    $processRecord | ConvertTo-Json -Depth 7 | Set-Content -LiteralPath (Join-Path $benchDirectory 'process.json') -Encoding utf8
    while (-not $compareProcess.HasExited) {
        $sampleStartMs = $monitorClock.Elapsed.TotalMilliseconds
        $memoryText = & $comparePython -c $memoryCode
        $memoryExit = $LASTEXITCODE
        $sample = $null
        try { $sample = $memoryText | ConvertFrom-Json -AsHashtable } catch { }
        $passed = $memoryExit -eq 0 -and $null -ne $sample -and $sample.errors.Count -eq 0 -and $sample.memory.physical_available_bytes -ge 12884901888 -and $sample.memory.commit_available_bytes -ge 4294967296
        $compareProcess.Refresh()
        $ownerRss = $ownerPrivate = $null
        try { if (-not $compareProcess.HasExited) { $ownerRss=$compareProcess.WorkingSet64; $ownerPrivate=$compareProcess.PrivateMemorySize64 } } catch { }
        $row = [ordered]@{schema_version=1;observed_utc=[DateTime]::UtcNow.ToString('o');owner_pid=$compareProcess.Id;owner_creation_utc=$creationUtc;sample_start_ms=$sampleStartMs;sample_gap_ms=if($null -eq $previousSampleMs){$null}else{$sampleStartMs-$previousSampleMs};probe_exit_code=$memoryExit;gate_pass=$passed;memory=$sample;owner_has_exited=$compareProcess.HasExited;owner_rss_bytes=$ownerRss;owner_private_bytes=$ownerPrivate}
        $resourceWriter.WriteLine(($row | ConvertTo-Json -Compress -Depth 8))
        $resourceWriter.Flush()
        $resourceStream.Flush($true)
        $previousSampleMs = $sampleStartMs
        if (-not $passed) {
            $stopReason = 'external_memory_gate_failed_or_unknown'
            if (-not $compareProcess.HasExited) { $compareProcess.Kill(); $compareProcess.WaitForExit() }
            break
        }
        $nextSampleMs += 1000
        $delayMs = [Math]::Max(0,$nextSampleMs-$monitorClock.Elapsed.TotalMilliseconds)
        if ($delayMs -gt 0) { Start-Sleep -Milliseconds ([int]$delayMs) }
    }
    $compareProcess.WaitForExit()
} catch {
    $stopReason = 'owned_launcher_or_monitor_error: ' + $_.Exception.Message
    if ($null -ne $compareProcess -and -not $compareProcess.HasExited) { $compareProcess.Kill(); $compareProcess.WaitForExit() }
    throw
} finally {
    $resourceWriter.Dispose()
    if ($null -ne $compareProcess) {
        $terminal = [ordered]@{schema_version=1;pid=$compareProcess.Id;process_create_utc=$creationUtc;launch_utc=$launchUtc;ended_utc=[DateTime]::UtcNow.ToString('o');exit_code=$compareProcess.ExitCode;stop_reason=$stopReason;owned_process_only=$true;npu_run=$false;vision_api_run=$false}
        $terminal | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath (Join-Path $benchDirectory 'exit.json') -Encoding utf8
        $terminal | ConvertTo-Json -Compress
    }
}
exit $compareProcess.ExitCode
