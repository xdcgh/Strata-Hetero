$ErrorActionPreference = 'Stop'

$repo = 'C:\Users\DC\Documents\ChatGPT\Strata-Hetero'
$runRoot = Join-Path $repo 'bench\hetero\20261009-10-native-cpu\runs\native-cpu-01'
$compareScript = Join-Path $runRoot 'compare_reference.py'
$inputsReceiptPath = Join-Path $repo 'bench\hetero\20261009-10-native-cpu\inputs-receipt.json'
$buildReceiptPath = Join-Path $repo 'bench\hetero\20261009-10-native-cpu\build\build-receipt.json'
$f32ReceiptPath = Join-Path $repo 'bench\hetero\20261009-02-arc-real-ffn\cpu-f32.json'
$arcComparisonPath = Join-Path $repo 'bench\hetero\20261009-02-arc-real-ffn\comparison.json'
$exe = 'E:\Strata-Hetero-data\build\native-cpu-harness-20261009-01\hetero_native_expert.exe'
$python = 'E:\Strata-Hetero-data\venv\Scripts\python.exe'
$outputRoot = 'E:\Strata-Hetero-data\experts\20261009-10-native-outputs-01'
$rowsToRun = @(1, 2, 4, 8, 16, 32, 64, 128, 256)
$minFreeRam = 12GB
$minCommitHeadroom = 4GB

function Get-ResourceSample {
    $os = Get-CimInstance Win32_OperatingSystem
    $memory = Get-CimInstance Win32_PerfFormattedData_PerfOS_Memory -ErrorAction SilentlyContinue
    $available = [int64]$os.FreePhysicalMemory * 1024
    $commit = $null
    $limit = $null
    if ($memory) { $commit = [int64]$memory.CommittedBytes; $limit = [int64]$memory.CommitLimit }
    $headroom = if ($null -ne $commit -and $null -ne $limit) { $limit - $commit } else { $null }
    return [ordered]@{
        time_utc = [DateTimeOffset]::UtcNow.ToString('o')
        available_physical_bytes = $available
        committed_bytes = $commit
        commit_limit_bytes = $limit
        commit_headroom_bytes = $headroom
    }
}

function Assert-ResourceGate {
    param($Sample, [string]$Phase)
    if ($Sample.available_physical_bytes -lt $minFreeRam) { throw "$Phase admission failed: RAM available below 12 GiB." }
    if ($null -eq $Sample.commit_headroom_bytes -or $Sample.commit_headroom_bytes -lt $minCommitHeadroom) {
        throw "$Phase admission failed: commit headroom unavailable or below 4 GiB."
    }
}

function Get-FileRecord {
    param([string]$Path)
    if (-not (Test-Path -LiteralPath $Path -PathType Leaf)) { throw "Required file is missing: $Path" }
    $item = Get-Item -LiteralPath $Path
    return [ordered]@{
        path = $item.FullName
        bytes = [int64]$item.Length
        sha256 = (Get-FileHash -LiteralPath $Path -Algorithm SHA256).Hash.ToLowerInvariant()
    }
}

function Convert-CimCreateTimeUtc {
    param($Process)
    $value = $Process.CreationDate
    if ($value -is [DateTimeOffset]) { return $value.ToUniversalTime().ToString('o') }
    if ($value -is [DateTime]) { return ([DateTimeOffset]$value.ToUniversalTime()).ToString('o') }
    $text = [string]$value
    $digits = $text.Substring(0, 14) + $text.Substring(15, 6)
    $local = [DateTime]::ParseExact($digits, 'yyyyMMddHHmmssffffff', [Globalization.CultureInfo]::InvariantCulture)
    $minutes = [int]$text.Substring(22, 3)
    if ($text[21] -eq '-') { $minutes = -$minutes }
    return ([DateTimeOffset]::new($local, [TimeSpan]::FromMinutes($minutes))).ToUniversalTime().ToString('o')
}

function Get-ProcessIdentity {
    param([int]$ProcessId, [string]$ExePath, [string[]]$OwnedArguments)
    $cim = Get-CimInstance Win32_Process -Filter "ProcessId = $ProcessId" -ErrorAction SilentlyContinue
    if (-not $cim) { return $null }
    $actualExe = [IO.Path]::GetFullPath([string]$cim.ExecutablePath)
    $expectedExe = [IO.Path]::GetFullPath($ExePath)
    if (-not [string]::Equals($actualExe, $expectedExe, [StringComparison]::OrdinalIgnoreCase)) { return $null }
    $commandLine = [string]$cim.CommandLine
    foreach ($argument in $OwnedArguments) {
        if ($commandLine.IndexOf($argument, [StringComparison]::OrdinalIgnoreCase) -lt 0) { return $null }
    }
    return [ordered]@{
        process_id = [int]$cim.ProcessId
        create_time_utc = Convert-CimCreateTimeUtc $cim
        executable_path = $actualExe
        command_line = $commandLine
        parent_process_id = [int]$cim.ParentProcessId
    }
}

function Stop-OwnedHarnessProcess {
    param($Identity, [string[]]$OwnedArguments)
    if (-not $Identity) { return 'No verified process identity; did not terminate any process.' }
    $current = Get-CimInstance Win32_Process -Filter "ProcessId = $($Identity.process_id)" -ErrorAction SilentlyContinue
    if (-not $current) { return 'Verified process already exited.' }
    $now = Get-ProcessIdentity -ProcessId $Identity.process_id -ExePath $Identity.executable_path -OwnedArguments $OwnedArguments
    if (-not $now -or $now.create_time_utc -ne $Identity.create_time_utc) {
        return 'Process identity changed; did not terminate an unverified PID.'
    }
    Stop-Process -Id $Identity.process_id -Force -ErrorAction Stop
    return 'Stopped only the verified owned harness process.'
}

function Invoke-NativeRow {
    param($InputEntry, [System.Collections.Generic.List[object]]$ResourceSamples)
    $rows = [int]$InputEntry.rows
    $tag = '{0:D3}' -f $rows
    $rowArchive = Join-Path $runRoot "rows-$tag"
    if (Test-Path -LiteralPath $rowArchive) { throw "Refusing to reuse row archive: $rowArchive" }
    New-Item -ItemType Directory -Path $rowArchive | Out-Null

    $outputPath = Join-Path $outputRoot "output-$tag.f32"
    $nativeReceiptPath = Join-Path $rowArchive 'native-receipt.json'
    $stdoutPath = Join-Path $rowArchive 'native.stdout.log'
    $stderrPath = Join-Path $rowArchive 'native.stderr.log'
    $processLogPath = Join-Path $rowArchive 'process.json'
    if ((Test-Path -LiteralPath $outputPath) -or (Test-Path -LiteralPath $nativeReceiptPath)) {
        throw "Refusing to overwrite an output or receipt for rows=$rows"
    }

    $sample = Get-ResourceSample
    $null = $ResourceSamples.Add($sample)
    Assert-ResourceGate $sample "rows=$rows pre-launch"

    $arguments = '--run --blob "{0}" --input "{1}" --rows {2} --hidden 2560 --intermediate 640 --gu-type 12 --down-type 7 --output "{3}" --receiptJSON "{4}" --warmup 1 --repeat 5' -f `
        $inputManifest.blob.path, $InputEntry.path, $rows, $outputPath, $nativeReceiptPath
    $process = Start-Process -FilePath $exe -ArgumentList $arguments -WorkingDirectory $rowArchive `
        -PassThru -WindowStyle Hidden -RedirectStandardOutput $stdoutPath -RedirectStandardError $stderrPath
    try {
        $process.Refresh()
        if (-not $process.HasExited) { $process.PriorityClass = [System.Diagnostics.ProcessPriorityClass]::Normal }
    } catch { }
    $owned = Get-ProcessIdentity -ProcessId $process.Id -ExePath $exe -OwnedArguments @($outputPath, $nativeReceiptPath, "--rows $rows")
    [ordered]@{
        rows = $rows
        pid = $process.Id
        process_priority = 'Normal'
        process_identity = $owned
        argv = @('--run', '--blob', $inputManifest.blob.path, '--input', $InputEntry.path, '--rows', "$rows",
                 '--hidden', '2560', '--intermediate', '640', '--gu-type', '12', '--down-type', '7',
                 '--output', $outputPath, '--receiptJSON', $nativeReceiptPath, '--warmup', '1', '--repeat', '5')
        start_utc = if ($owned) { $owned.create_time_utc } else { $process.StartTime.ToUniversalTime().ToString('o') }
        output_path = $outputPath
        receipt_path = $nativeReceiptPath
        stdout_path = $stdoutPath
        stderr_path = $stderrPath
    } | ConvertTo-Json -Depth 6 | Set-Content -LiteralPath $processLogPath -Encoding utf8

    $stopReason = $null
    while ($true) {
        $process.Refresh()
        if ($process.HasExited) { break }
        Start-Sleep -Seconds 1
        $sample = Get-ResourceSample
        $null = $ResourceSamples.Add($sample)
        $process.Refresh()
        $inside = $sample.available_physical_bytes -ge $minFreeRam -and
                  $null -ne $sample.commit_headroom_bytes -and $sample.commit_headroom_bytes -ge $minCommitHeadroom
        if (-not $inside -and -not $process.HasExited) {
            $stopReason = Stop-OwnedHarnessProcess -Identity $owned -OwnedArguments @($outputPath, $nativeReceiptPath, "--rows $rows")
            break
        }
    }
    $process.Refresh()
    if (-not $process.HasExited) { $process.WaitForExit() }
    $exitCode = [int]$process.ExitCode
    $finishSample = Get-ResourceSample
    $null = $ResourceSamples.Add($finishSample)
    if ($stopReason) { "Resource gate stop: $stopReason" | Set-Content -LiteralPath (Join-Path $rowArchive 'resource-gate-stop.txt') -Encoding utf8 }

    $processRecord = Get-Content -LiteralPath $processLogPath -Raw | ConvertFrom-Json
    $processRecord | Add-Member -NotePropertyName exit_code -NotePropertyValue $exitCode
    $processRecord | Add-Member -NotePropertyName resource_gate_stop -NotePropertyValue $stopReason
    $processRecord | Add-Member -NotePropertyName finish_utc -NotePropertyValue $finishSample.time_utc
    $processRecord | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $processLogPath -Encoding utf8
    if ($exitCode -ne 0) { throw "Native CPU process failed for rows=$rows with exit $exitCode; retained logs and receipt at $rowArchive" }

    if (-not (Test-Path -LiteralPath $nativeReceiptPath -PathType Leaf) -or -not (Test-Path -LiteralPath $outputPath -PathType Leaf)) {
        throw "Native CPU process for rows=$rows exited 0 without both output and receipt; retained $rowArchive"
    }

    # The NumPy/NPZ comparison starts only after this native process has exited.
    $comparePreSample = Get-ResourceSample
    $null = $ResourceSamples.Add($comparePreSample)
    Assert-ResourceGate $comparePreSample "rows=$rows NumPy comparison pre-launch"
    $compareReceipt = Join-Path $rowArchive 'comparison.json'
    $compareStdout = Join-Path $rowArchive 'comparison.stdout.log'
    $compareStderr = Join-Path $rowArchive 'comparison.stderr.log'
    $compareArgs = '"{0}" --rows {1} --input-receipt "{2}" --input "{3}" --native-receipt "{4}" --output "{5}" --cpu-f32-receipt "{6}" --report "{7}"' -f `
        $compareScript, $rows, $inputsReceiptPath, $InputEntry.path, $nativeReceiptPath, $outputPath, $f32ReceiptPath, $compareReceipt
    $compareProcess = Start-Process -FilePath $python -ArgumentList $compareArgs -WorkingDirectory $repo `
        -PassThru -WindowStyle Hidden -RedirectStandardOutput $compareStdout -RedirectStandardError $compareStderr
    $compareProcess.PriorityClass = [System.Diagnostics.ProcessPriorityClass]::Idle
    $compareProcess.WaitForExit()
    $comparePostSample = Get-ResourceSample
    $null = $ResourceSamples.Add($comparePostSample)
    if ($compareProcess.ExitCode -ne 0) { throw "NumPy reference comparison failed for rows=$rows with exit $($compareProcess.ExitCode); retained $rowArchive" }
    Assert-ResourceGate $comparePostSample "rows=$rows NumPy comparison post-run"
    $quality = Get-Content -LiteralPath $compareReceipt -Raw | ConvertFrom-Json
    $native = Get-Content -LiteralPath $nativeReceiptPath -Raw | ConvertFrom-Json
    $arcRow = @($arcComparison.row_samples | Where-Object { $_.rows -eq $rows } | Select-Object -First 1)[0]
    if (-not $arcRow) { throw "Saved Arc F32 median is missing for rows=$rows" }
    $nativeMedianMs = [double]$native.timings_ns.median / 1000000.0
    $rowResult = [ordered]@{
        rows = $rows
        native_process_exit = $exitCode
        native_latency_median_ms = $nativeMedianMs
        native_formal_times_ns = @($native.timings_ns.formal_iterations)
        arc_f32_median_ms = $arcRow.arc_f32_median_ms
        native_vs_arc_median_ratio = $nativeMedianMs / [double]$arcRow.arc_f32_median_ms
        quality = $quality.quality
        quality_pass = [bool]$quality.quality.pass
        output_path = $outputPath
        native_receipt_path = $nativeReceiptPath
        comparison_receipt_path = $compareReceipt
    }
    $rowResult | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath (Join-Path $rowArchive 'row-summary.json') -Encoding utf8
    return $rowResult
}

if (Test-Path -LiteralPath $outputRoot) { throw "Output directory exists; refusing overwrite: $outputRoot" }
foreach ($file in @($inputsReceiptPath, $buildReceiptPath, $f32ReceiptPath, $arcComparisonPath, $exe, $python, $compareScript)) {
    if (-not (Test-Path -LiteralPath $file -PathType Leaf)) { throw "Required run input missing: $file" }
}
if (Test-Path -LiteralPath (Join-Path $runRoot 'run-summary.json')) { throw 'Run summary exists; refusing overwrite.' }

$inputsReceiptRaw = Get-Content -LiteralPath $inputsReceiptPath -Raw
$inputManifest = $inputsReceiptRaw | ConvertFrom-Json
$inputsReceiptSha = (Get-FileHash -LiteralPath $inputsReceiptPath -Algorithm SHA256).Hash.ToLowerInvariant()
$buildReceipt = Get-Content -LiteralPath $buildReceiptPath -Raw | ConvertFrom-Json
$exeFileRecord = @($buildReceipt.binary_artifacts | Where-Object { [IO.Path]::GetFileName($_.path) -eq 'hetero_native_expert.exe' } | Select-Object -First 1)[0]
if (-not $exeFileRecord -or $buildReceipt.status -ne 'success') { throw 'Native harness build receipt lacks a successful executable identity.' }
$exeSha = (Get-FileHash -LiteralPath $exe -Algorithm SHA256).Hash.ToLowerInvariant()
if ($exeSha -ne $exeFileRecord.sha256 -or $exeSha -ne '9cd95dae9c92b50a6487a3a564322576cc2f9aaeb9f16a72946e97afa2ab90b6') {
    throw "Native harness EXE SHA mismatch: $exeSha"
}
if ($inputManifest.schema_version -ne 1 -or $inputManifest.status -ne 'prepared' -or $inputManifest.device_created -or
    $inputManifest.quantization.gate_up -ne 'Q4_K' -or $inputManifest.quantization.down -ne 'Q5_1' -or $inputManifest.quantization.requantized) {
    throw 'Inputs receipt does not describe the approved raw Q4_K/Q5_1 payload and fixed F32 inputs.'
}
$sourceReceipt = Get-Content -LiteralPath $inputManifest.source_receipt -Raw | ConvertFrom-Json
$sourceIdentity = Get-Content -LiteralPath $inputManifest.source_identity -Raw | ConvertFrom-Json
if ((Get-FileHash -LiteralPath $inputManifest.source_receipt -Algorithm SHA256).Hash.ToLowerInvariant() -ne $inputManifest.source_receipt_sha256 -or
    (Get-FileHash -LiteralPath $inputManifest.source_identity -Algorithm SHA256).Hash.ToLowerInvariant() -ne $inputManifest.source_identity_sha256) {
    throw 'Source extraction receipt or identity SHA mismatch.'
}
if ($sourceReceipt.status -ne 'extracted' -or $sourceReceipt.selection.layer -ne 28 -or $sourceReceipt.selection.expert -ne 288 -or
    $inputManifest.selection.layer -ne 28 -or $inputManifest.selection.expert -ne 288) { throw 'Unexpected layer/expert selection in source receipts.' }
foreach ($role in @('gate', 'up', 'down')) {
    $payload = @($inputManifest.payloads | Where-Object { $_.role -eq $role } | Select-Object -First 1)[0]
    if (-not $payload -or $sourceReceipt.raw_selected_payload_sha256.$role -ne $payload.sha256) { throw "Source raw payload identity mismatch for $role" }
}
$blobRecord = Get-FileRecord -Path $inputManifest.blob.path
if ($blobRecord.bytes -ne $inputManifest.blob.bytes -or $blobRecord.sha256 -ne $inputManifest.blob.sha256 -or $blobRecord.bytes -ne 3072000) {
    throw 'Native raw blob size/SHA mismatch.'
}
$inputRecords = [System.Collections.Generic.List[object]]::new()
foreach ($entry in $inputManifest.inputs) {
    $record = Get-FileRecord -Path $entry.path
    if ($record.bytes -ne $entry.bytes -or $record.sha256 -ne $entry.sha256) { throw "F32 input mismatch for rows=$($entry.rows)" }
    $null = $inputRecords.Add($record)
}
if (@($inputManifest.inputs | ForEach-Object { [int]$_.rows }) -join ',' -ne '1,2,4,8,16,32,64,128,256') {
    throw 'Input receipt row set/order differs from the requested sweep.'
}
if ([Environment]::GetEnvironmentVariable('STRATA_FORCE_AVX2', 'Process') -eq '1') {
    throw 'STRATA_FORCE_AVX2=1 would disable the model-default Q8_K AVX2 path; refusing this run.'
}
$f32Receipt = Get-Content -LiteralPath $f32ReceiptPath -Raw | ConvertFrom-Json
if ($f32Receipt.status -ne 'complete' -or -not $f32Receipt.weights_loaded_and_hash_verified -or
    $f32Receipt.weights_identity.layer -ne 28 -or $f32Receipt.weights_identity.expert -ne 288) {
    throw 'Prior CPU F32 reference receipt is not a complete, hash-verified match for layer 28 expert 288.'
}
foreach ($role in @('gate', 'up', 'down')) {
    if ($sourceIdentity.weights_sha256.$role -ne $f32Receipt.weights_identity.weights_sha256.$role) {
        throw "Decoded F32 reference identity differs from the selected native source for $role"
    }
}
if ($sourceIdentity.layer -ne 28 -or $sourceIdentity.expert -ne 288 -or $sourceIdentity.model -ne $f32Receipt.weights_identity.model) {
    throw 'Decoded F32 reference identity differs from the selected native source metadata.'
}
$arcComparison = Get-Content -LiteralPath $arcComparisonPath -Raw | ConvertFrom-Json
$resourceSamples = [System.Collections.Generic.List[object]]::new()
$preflightSample = Get-ResourceSample
$null = $resourceSamples.Add($preflightSample)
Assert-ResourceGate $preflightSample 'initial real-native-run'
$processes = @(Get-Process strata, hetero_native_expert, cmake, ninja, cl -ErrorAction SilentlyContinue)
if ($processes.Count -ne 0) { throw "Refusing benchmark while build/engine process exists: $($processes.Name -join ',')" }
if (Get-NetTCPConnection -State Listen -LocalPort 8081 -ErrorAction SilentlyContinue) { throw 'Refusing benchmark while the model API listener is active.' }

$preflight = [ordered]@{
    schema = 'strata-hetero-native-cpu-preflight-v1'
    created_utc = [DateTimeOffset]::UtcNow.ToString('o')
    inputs_receipt = [ordered]@{ path = $inputsReceiptPath; sha256 = $inputsReceiptSha; status = $inputManifest.status }
    source_receipt_sha256 = $inputManifest.source_receipt_sha256
    source_identity_sha256 = $inputManifest.source_identity_sha256
    native_blob = $blobRecord
    inputs = @($inputRecords)
    harness_binary = [ordered]@{ path = $exe; bytes = (Get-Item -LiteralPath $exe).Length; sha256 = $exeSha; build_head = $buildReceipt.source.head_at_start }
    weights_reference_metadata = [ordered]@{ cpu_f32_receipt = $f32ReceiptPath; weights_npz_path = $f32Receipt.weights_header_info.path; npz_arrays_not_loaded_yet = $true }
    rows_order = $rowsToRun
    warmup = 1
    repeat = 5
    avx2_force_disabled = $false
    output_root = $outputRoot
    gpu_executed = $false
    model_or_engine_cli_executed = $false
    initial_resource_sample = $preflightSample
}
$preflight | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath (Join-Path $runRoot 'preflight.json') -Encoding utf8
New-Item -ItemType Directory -Path $outputRoot | Out-Null

$rowResults = [System.Collections.Generic.List[object]]::new()
$overallFailure = $null
$runStart = [DateTimeOffset]::UtcNow
try {
    foreach ($rowCount in $rowsToRun) {
        $entry = @($inputManifest.inputs | Where-Object { [int]$_.rows -eq $rowCount } | Select-Object -First 1)[0]
        if (-not $entry) { throw "Approved input is missing for rows=$rowCount" }
        Write-Output "Starting native rows=$rowCount"
        $result = Invoke-NativeRow -InputEntry $entry -ResourceSamples $resourceSamples
        $null = $rowResults.Add($result)
        [ordered]@{
            schema = 'strata-hetero-native-cpu-progress-v1'
            status = 'in_progress'
            completed_rows = @($rowResults)
            next_rows = @($rowsToRun | Where-Object { $_ -gt $rowCount })
            resource_samples = @($resourceSamples)
            created_utc = [DateTimeOffset]::UtcNow.ToString('o')
        } | ConvertTo-Json -Depth 10 | Set-Content -LiteralPath (Join-Path $runRoot 'progress.json') -Encoding utf8
        Write-Output "Completed native rows=$rowCount quality_pass=$($result.quality_pass) median_ms=$($result.native_latency_median_ms)"
    }
} catch {
    $overallFailure = $_.Exception.ToString()
}

$finishSample = Get-ResourceSample
$null = $resourceSamples.Add($finishSample)
$arcRows = @($arcComparison.row_samples)
$summary = [ordered]@{
    schema = 'strata-hetero-native-cpu-run-v1'
    status = if ($overallFailure) { 'failed_or_incomplete' } elseif (@($rowResults | Where-Object { -not $_.quality_pass }).Count -gt 0) { 'complete_with_quality_failures' } else { 'complete_quality_pass' }
    failure_reason = $overallFailure
    started_utc = $runStart.ToString('o')
    finished_utc = [DateTimeOffset]::UtcNow.ToString('o')
    rows_requested_in_order = $rowsToRun
    completed_rows = @($rowResults)
    all_quality_pass = (@($rowResults).Count -eq $rowsToRun.Count -and @($rowResults | Where-Object { -not $_.quality_pass }).Count -eq 0)
    inputs_manifest_sha256 = $inputsReceiptSha
    raw_blob_sha256 = $blobRecord.sha256
    native_harness_sha256 = $exeSha
    weights_reference = $f32Receipt.weights_header_info.path
    weights_sha256 = $f32Receipt.weights_identity.weights_sha256
    strict_tolerances = $f32Receipt.quality_tolerances
    arc_comparison_receipt = $arcComparisonPath
    scope = 'Single-thread native Q4_K/Q5_1 expert with default Q8_K activation path. The measured interval includes x-to-Q8 quantization, full gate/up plus SiLU-times-up, intermediate Q8 requantization, and full down rows. It excludes engine pool/IPC, expert retrieval, weight movement, and model routing. Arc F32 medians include copy/retrieval; ratios are descriptive and not an apples-to-apples ranking.'
    resource_samples = @($resourceSamples)
    gpu_executed = $false
    model_or_api_called = $false
    engine_cli_executed = $false
}
$summary | ConvertTo-Json -Depth 12 | Set-Content -LiteralPath (Join-Path $runRoot 'run-summary.json') -Encoding utf8
if ($overallFailure) { throw "Native CPU run stopped; partial receipts/logs retained in $runRoot. $overallFailure" }
