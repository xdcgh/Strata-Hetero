[CmdletBinding()]
param(
    [switch]$H4Stopped,
    [switch]$StartBuild,
    [string]$RunId = 'native-cpu-harness-20261009-01'
)

$ErrorActionPreference = 'Stop'
$ggmlPin = '3cf03257f219afbe7334045ff7c6a06ac68c627d'
$repo = 'C:\Users\DC\Documents\ChatGPT\Strata-Hetero'
$ggml = 'E:\Strata-Hetero-data\source\llama-3cf0325'
$buildRoot = 'E:\Strata-Hetero-data\build'
$cmake = 'E:\Strata-Hetero-data\venv\Lib\site-packages\cmake\data\bin\cmake.exe'
$ninja = 'E:\Strata-Hetero-data\venv\Scripts\ninja.exe'
$ctest = Join-Path (Split-Path -Parent $cmake) 'ctest.exe'
$vcvars = 'C:\Program Files (x86)\Microsoft Visual Studio\18\BuildTools\VC\Auxiliary\Build\vcvars64.bat'
$expectedCmakeVersion = 'cmake version 4.4.3'
$expectedNinjaVersion = '1.13.2.git.kitware.jobserver-pipe-1'
$minFreeRam = 12GB
$minCommitHeadroom = 4GB

if (-not $H4Stopped -or -not $StartBuild) {
    throw 'Refusing to create a build directory or launch tools. After root explicitly confirms H4 is stopped, invoke with both -H4Stopped and -StartBuild.'
}
if ($RunId -notmatch '^[a-z0-9][a-z0-9-]{2,63}$') { throw 'RunId must be a simple lowercase alphanumeric/hyphen ID.' }

$build = Join-Path $buildRoot $RunId
$batch = Join-Path $build 'run-build.cmd'
$receiptPath = Join-Path $build 'build-receipt.json'
$repoSourceFiles = @(
    'CMakeLists.txt',
    'src/kernels/cpu/expert.cpp',
    'src/kernels/cpu/pool.cpp',
    'src/kernels/cpu/native_expert.cpp',
    'src/kernels/cpu/expert_layout.cpp',
    'src/kernels/cpu/iq_avx2.cpp',
    'src/kernels/cpu/iq_avx512.cpp',
    'src/kernels/cpu/kq_avx2.cpp',
    'src/kernels/cpu/kq_avx1.cpp',
    'include/strata/kernels/cpu/expert.hpp',
    'include/strata/kernels/cpu/pool.hpp',
    'include/strata/kernels/cpu/native_expert.hpp',
    'include/strata/kernels/cpu/expert_layout.hpp',
    'include/strata/kernels/cpu/iq_avx2.hpp',
    'include/strata/kernels/cpu/iq_avx512.hpp',
    'include/strata/kernels/cpu/kq_avx2.hpp',
    'src/hetero/native_expert_bench.cpp',
    'src/hetero/native_expert_fixture.cpp',
    'tools/hetero_native_cpu/CMakeLists.txt',
    'tools/hetero_native_cpu/sha256.hpp',
    'tools/hetero_native_cpu/cli_fixtures.cmake',
    'tools/hetero_native_cpu/run_build.ps1'
)
$ggmlSourceFiles = @(
    'ggml/include/ggml.h',
    'ggml/include/ggml-cpu.h',
    'ggml/src/ggml-common.h',
    'ggml/src/ggml.c',
    'ggml/src/ggml-cpu/ggml-cpu.c',
    'ggml/src/ggml-cpu/quants.c',
    'ggml/src/ggml-cpu/arch/x86/quants.c'
)

function Get-SourceSnapshot {
    param([string]$SourceRoot, [string[]]$RelativePaths)
    $records = [System.Collections.Generic.List[object]]::new()
    foreach ($relativePath in $RelativePaths) {
        $path = Join-Path $SourceRoot ($relativePath -replace '/', '\')
        if (-not (Test-Path -LiteralPath $path -PathType Leaf)) { throw "Source input is missing: $path" }
        $item = Get-Item -LiteralPath $path
        $sha = (Get-FileHash -LiteralPath $path -Algorithm SHA256).Hash.ToLowerInvariant()
        $gitRoot = $SourceRoot
        $gitPath = $relativePath
        $blob = $null
        $status = $null
        $blobOutput = & git -C $gitRoot rev-parse "HEAD:$gitPath" 2>$null
        $blobExit = $LASTEXITCODE
        if ($blobExit -eq 0) { $blob = ([string]$blobOutput).Trim() }
        $statusOutput = & git -C $gitRoot status --porcelain -- $gitPath 2>$null
        $statusExit = $LASTEXITCODE
        if ($statusExit -eq 0) { $status = (@($statusOutput) -join "`n").TrimEnd() }
        $null = $records.Add([ordered]@{
            path = $path
            bytes = [int64]$item.Length
            sha256 = $sha
            git_blob_at_head = $blob
            git_status = $status
        })
    }
    return @($records)
}

function Get-HostResourceSample {
    $os = Get-CimInstance Win32_OperatingSystem
    $memory = Get-CimInstance Win32_PerfFormattedData_PerfOS_Memory -ErrorAction SilentlyContinue
    $available = [int64]$os.FreePhysicalMemory * 1024
    $commit = $null
    $limit = $null
    if ($memory) {
        $commit = [int64]$memory.CommittedBytes
        $limit = [int64]$memory.CommitLimit
    }
    $headroom = if ($null -ne $commit -and $null -ne $limit) { $limit - $commit } else { $null }
    return [ordered]@{
        time_utc = [DateTimeOffset]::UtcNow.ToString('o')
        available_physical_bytes = $available
        committed_bytes = $commit
        commit_limit_bytes = $limit
        commit_headroom_bytes = $headroom
    }
}

function Get-ProcessCreateTimeUtc {
    param($CimProcess)
    $value = $CimProcess.CreationDate
    if ($value -is [DateTimeOffset]) { return $value.ToUniversalTime().ToString('o') }
    if ($value -is [DateTime]) { return ([DateTimeOffset]$value.ToUniversalTime()).ToString('o') }
    $text = [string]$value
    if ($text.Length -lt 25) { throw "Invalid CIM process CreationDate: $text" }
    $dateDigits = $text.Substring(0, 14) + $text.Substring(15, 6)
    $localTime = [DateTime]::ParseExact($dateDigits, 'yyyyMMddHHmmssffffff', [Globalization.CultureInfo]::InvariantCulture)
    $offsetMinutes = [int]$text.Substring(22, 3)
    if ($text[21] -eq '-') { $offsetMinutes = -$offsetMinutes }
    $offset = [TimeSpan]::FromMinutes($offsetMinutes)
    return ([DateTimeOffset]::new($localTime, $offset)).ToUniversalTime().ToString('o')
}

function Normalize-ProcessPath {
    param([string]$Path)
    if ([string]::IsNullOrWhiteSpace($Path)) { return '' }
    return [System.IO.Path]::GetFullPath($Path).TrimEnd('\')
}

function Get-OwnedRootIdentity {
    param([int]$RootProcessId, [string]$BatchPath, [string]$ExpectedExecutable, [string]$ExpectedStartTimeUtc)
    $root = Get-CimInstance Win32_Process -Filter "ProcessId = $RootProcessId" -ErrorAction SilentlyContinue
    if (-not $root) { throw "Cannot capture CIM identity for launched cmd.exe PID $RootProcessId" }
    $exePath = Normalize-ProcessPath $root.ExecutablePath
    $expectedExePath = Normalize-ProcessPath $ExpectedExecutable
    $absoluteBatch = [System.IO.Path]::GetFullPath($BatchPath)
    $commandLine = [string]$root.CommandLine
    if (-not [string]::Equals($exePath, $expectedExePath, [StringComparison]::OrdinalIgnoreCase)) {
        throw "Launched PID $RootProcessId is not the expected cmd.exe ($exePath)"
    }
    if ($commandLine.IndexOf($absoluteBatch, [StringComparison]::OrdinalIgnoreCase) -lt 0) {
        throw "Launched cmd.exe PID $RootProcessId command line does not name owned batch $absoluteBatch"
    }
    $createdUtc = Get-ProcessCreateTimeUtc $root
    $startDelta = [math]::Abs(([DateTimeOffset]::Parse($createdUtc) - [DateTimeOffset]::Parse($ExpectedStartTimeUtc)).TotalSeconds)
    if ($startDelta -gt 2) { throw "CIM creation time for cmd.exe PID $RootProcessId does not match the newly launched process object." }
    return [ordered]@{
        process_id = [int]$root.ProcessId
        create_time_utc = $createdUtc
        executable_path = $exePath
        command_line = $commandLine
        batch_path = $absoluteBatch
    }
}

function Stop-OwnedProcessTree {
    param([System.Collections.IDictionary]$RootIdentity)
    $diagnostics = [System.Collections.Generic.List[string]]::new()
    $stopped = [System.Collections.Generic.List[int]]::new()
    $rootProcessId = [int]$RootIdentity.process_id
    $snapshot = @(Get-CimInstance Win32_Process)
    $root = @($snapshot | Where-Object { [int]$_.ProcessId -eq $rootProcessId }) | Select-Object -First 1
    if (-not $root) {
        $null = $diagnostics.Add("Root PID $rootProcessId is absent from the CIM snapshot; no process was stopped.")
        return [pscustomobject]@{ complete = $false; stopped_process_ids = @(); diagnostics = @($diagnostics) }
    }
    $rootMatches = (Get-ProcessCreateTimeUtc $root) -eq $RootIdentity.create_time_utc -and
                   [string]::Equals((Normalize-ProcessPath $root.ExecutablePath), $RootIdentity.executable_path, [StringComparison]::OrdinalIgnoreCase) -and
                   ([string]$root.CommandLine).IndexOf($RootIdentity.batch_path, [StringComparison]::OrdinalIgnoreCase) -ge 0
    if (-not $rootMatches) {
        $null = $diagnostics.Add("Root PID $rootProcessId identity no longer matches its recorded creation time, executable, and owned batch; no process was stopped.")
        return [pscustomobject]@{ complete = $false; stopped_process_ids = @(); diagnostics = @($diagnostics) }
    }

    $depth = @{}
    $depth[[string]$rootProcessId] = 0
    do {
        $changed = $false
        foreach ($process in $snapshot) {
            $processId = [int]$process.ProcessId
            $key = [string]$processId
            $parentKey = [string][int]$process.ParentProcessId
            if (-not $depth.ContainsKey($key) -and $depth.ContainsKey($parentKey)) {
                $depth[$key] = [int]$depth[$parentKey] + 1
                $changed = $true
            }
        }
    } while ($changed)

    $descendants = @(
        foreach ($process in $snapshot) {
            $key = [string][int]$process.ProcessId
            if ($key -ne [string]$rootProcessId -and $depth.ContainsKey($key)) {
                [pscustomobject]@{ process = $process; depth = [int]$depth[$key] }
            }
        }
    ) | Sort-Object -Property @{ Expression = { $_.depth }; Descending = $true }, @{ Expression = { [int]$_.process.ProcessId }; Descending = $true }

    $complete = $true
    foreach ($entry in $descendants) {
        $expected = $entry.process
        $processId = [int]$expected.ProcessId
        if ([string]::IsNullOrWhiteSpace([string]$expected.ExecutablePath)) {
            $complete = $false
            $null = $diagnostics.Add("Skipped PID $processId because its CIM snapshot has no executable path.")
            continue
        }
        $current = Get-CimInstance Win32_Process -Filter "ProcessId = $processId" -ErrorAction SilentlyContinue
        if (-not $current) { continue }
        $sameIdentity = (Get-ProcessCreateTimeUtc $current) -eq (Get-ProcessCreateTimeUtc $expected) -and
                        [string]::Equals((Normalize-ProcessPath $current.ExecutablePath), (Normalize-ProcessPath $expected.ExecutablePath), [StringComparison]::OrdinalIgnoreCase) -and
                        [int]$current.ParentProcessId -eq [int]$expected.ParentProcessId
        if (-not $sameIdentity) {
            $complete = $false
            $null = $diagnostics.Add("Skipped PID $processId because its current creation time, executable, or parent differs from the same CIM snapshot.")
            continue
        }
        try {
            Stop-Process -Id $processId -Force -ErrorAction Stop
            $null = $stopped.Add($processId)
        } catch {
            $complete = $false
            $null = $diagnostics.Add("Could not stop verified owned child PID ${processId}: $($_.Exception.Message)")
        }
    }

    $rootNow = Get-CimInstance Win32_Process -Filter "ProcessId = $rootProcessId" -ErrorAction SilentlyContinue
    if ($rootNow) {
        $rootStillMatches = (Get-ProcessCreateTimeUtc $rootNow) -eq $RootIdentity.create_time_utc -and
                            [string]::Equals((Normalize-ProcessPath $rootNow.ExecutablePath), $RootIdentity.executable_path, [StringComparison]::OrdinalIgnoreCase) -and
                            ([string]$rootNow.CommandLine).IndexOf($RootIdentity.batch_path, [StringComparison]::OrdinalIgnoreCase) -ge 0
        if ($rootStillMatches) {
            try {
                Stop-Process -Id $rootProcessId -Force -ErrorAction Stop
                $null = $stopped.Add($rootProcessId)
            } catch {
                $complete = $false
                $null = $diagnostics.Add("Could not stop verified owned cmd.exe PID ${rootProcessId}: $($_.Exception.Message)")
            }
        } else {
            $complete = $false
            $null = $diagnostics.Add("Skipped root PID $rootProcessId because its identity changed before termination.")
        }
    }
    return [pscustomobject]@{ complete = $complete; stopped_process_ids = @($stopped); diagnostics = @($diagnostics) }
}

function Read-ExitCodeFile {
    param([string]$Path)
    if (-not (Test-Path -LiteralPath $Path -PathType Leaf)) { return $null }
    $number = 0
    $content = (Get-Content -LiteralPath $Path -Raw).Trim()
    if ([int]::TryParse($content, [ref]$number)) { return $number }
    return $null
}

function Get-BinarySnapshot {
    param([string]$BuildPath)
    $wanted = @('hetero_native_expert.exe', 'hetero_native_expert_metadata_fixture.exe', 'strata_kernels_cpu.lib', 'ggml-cpu.lib', 'ggml-base.lib')
    $files = @(Get-ChildItem -LiteralPath $BuildPath -File -Recurse -ErrorAction SilentlyContinue | Where-Object { $wanted -contains $_.Name })
    $result = [System.Collections.Generic.List[object]]::new()
    foreach ($file in $files) {
        $sha = (Get-FileHash -LiteralPath $file.FullName -Algorithm SHA256).Hash.ToLowerInvariant()
        $null = $result.Add([ordered]@{ path = $file.FullName; bytes = [int64]$file.Length; sha256 = $sha })
    }
    return @($result)
}

if (-not (Test-Path -LiteralPath $repo -PathType Container)) { throw "Repository path is missing: $repo" }
foreach ($tool in @($cmake, $ninja, $ctest, $vcvars)) {
    if (-not (Test-Path -LiteralPath $tool -PathType Leaf)) { throw "Pinned build tool is missing: $tool" }
}
$cmakeVersionOutput = & $cmake --version
$cmakeVersionExit = $LASTEXITCODE
$cmakeVersion = ([string]$cmakeVersionOutput[0]).Trim()
if ($cmakeVersionExit -ne 0 -or $cmakeVersion -ne $expectedCmakeVersion) { throw "CMake version mismatch: expected $expectedCmakeVersion, found $cmakeVersion" }
$ninjaVersionOutput = & $ninja --version
$ninjaVersionExit = $LASTEXITCODE
$ninjaVersion = (@($ninjaVersionOutput) -join '').Trim()
if ($ninjaVersionExit -ne 0 -or $ninjaVersion -ne $expectedNinjaVersion) { throw "Ninja version mismatch: expected $expectedNinjaVersion, found $ninjaVersion" }
if (-not (Test-Path -LiteralPath $ggml -PathType Container)) { throw "Pinned ggml checkout is missing: $ggml" }
$ggmlHead = (& git -C $ggml rev-parse HEAD).Trim()
if ($LASTEXITCODE -ne 0 -or $ggmlHead -ne $ggmlPin) { throw "ggml checkout must be exactly $ggmlPin, found $ggmlHead" }
$ggmlStatus = @(& git -C $ggml status --porcelain)
if ($LASTEXITCODE -ne 0 -or $ggmlStatus.Count -ne 0) { throw 'Pinned ggml checkout must be clean before the build.' }
if (Test-Path -LiteralPath $build) { throw "Refusing to reuse build directory: $build" }

$initialResources = Get-HostResourceSample
if ($initialResources.available_physical_bytes -lt $minFreeRam) { throw 'Admission failed: available RAM is below 12 GiB.' }
if ($null -eq $initialResources.commit_headroom_bytes -or $initialResources.commit_headroom_bytes -lt $minCommitHeadroom) {
    throw 'Admission failed: commit headroom is unavailable or below 4 GiB.'
}

$headBefore = (& git -C $repo rev-parse HEAD).Trim()
if ($LASTEXITCODE -ne 0) { throw 'Cannot identify the current Strata source HEAD.' }
$branchBefore = (& git -C $repo branch --show-current).Trim()
$sourceBefore = Get-SourceSnapshot -SourceRoot $repo -RelativePaths $repoSourceFiles
$ggmlBefore = Get-SourceSnapshot -SourceRoot $ggml -RelativePaths $ggmlSourceFiles

New-Item -ItemType Directory -Path $build | Out-Null
$buildDirCreated = $true
$runStart = [DateTimeOffset]::UtcNow
$helperProcess = Get-Process -Id $PID
$helperProcess.PriorityClass = [System.Diagnostics.ProcessPriorityClass]::Idle
$samples = [System.Collections.Generic.List[object]]::new()
$buildProcess = $null
$rootIdentity = $null
$stopResult = $null
$resourceGateStopped = $false
$scriptFailure = $null
$batchExit = $null
$prioritySet = $false

try {
    $sourceBefore | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath (Join-Path $build 'source-before.json') -Encoding utf8
    $ggmlBefore | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath (Join-Path $build 'ggml-before.json') -Encoding utf8
    $null = $samples.Add($initialResources)

$batchText = @"
@echo off
setlocal
set /a identity_wait=0
:wait_for_process_identity
if exist "$build\process-identity-verified.txt" goto process_identity_verified
set /a identity_wait+=1
if %identity_wait% GEQ 60 exit /b 92
timeout /t 1 /nobreak >nul
goto wait_for_process_identity
:process_identity_verified
call "$vcvars" > "$build\vcvars.log" 2>&1
if errorlevel 1 (
  >"$build\configure.exit.txt" echo 1
  exit /b 1
)
set "CUDA_PATH="
set "CUDA_HOME="
set "CUDACXX="
set "CUDA_VISIBLE_DEVICES="
set "HIP_PATH="
set "ROCM_PATH="
set "ONEAPI_ROOT="
set "ONEAPI_DEVICE_SELECTOR="
set "SYCL_DEVICE_FILTER="
where cl > "$build\tool-paths.log" 2>&1
where cmake >> "$build\tool-paths.log" 2>&1
where ninja >> "$build\tool-paths.log" 2>&1
cl /Bv > "$build\cl-version.log" 2>&1
"$cmake" --version > "$build\cmake-version.log" 2>&1
"$ninja" --version > "$build\ninja-version.log" 2>&1
"$cmake" -S "$repo\tools\hetero_native_cpu" -B "$build" -G Ninja "-DCMAKE_MAKE_PROGRAM=$ninja" -DCMAKE_BUILD_TYPE=Release -DSTRATA_ENABLE_CUDA=OFF -DSTRATA_ENABLE_HIP=OFF -DSTRATA_HIP_GFX906=OFF -DSTRATA_ENABLE_SYCL=OFF -DSTRATA_PREFILL_MMQ=OFF -DSTRATA_MMQ_KQUANTS=OFF -DSTRATA_ORCA_Q4KS_MMQ=OFF -DSTRATA_NATIVE_EXPERTS=ON -DSTRATA_BUILD_TESTS=OFF -DSTRATA_PORTABLE=ON "-DSTRATA_ISA_FLOOR=" "-DSTRATA_GGML_DIR=$ggml" "-DHETERO_NATIVE_GGML_DIR=$ggml" > "$build\configure.log" 2>&1
set "configure_rc=%errorlevel%"
>"$build\configure.exit.txt" echo %configure_rc%
if not "%configure_rc%"=="0" exit /b %configure_rc%
"$cmake" --build "$build" --target hetero_native_expert hetero_native_expert_metadata_fixture --parallel 1 > "$build\build.log" 2>&1
set "build_rc=%errorlevel%"
>"$build\build.exit.txt" echo %build_rc%
if not "%build_rc%"=="0" exit /b %build_rc%
"$cmake" -E env --unset=CUDA_VISIBLE_DEVICES --unset=ONEAPI_DEVICE_SELECTOR --unset=SYCL_DEVICE_FILTER "$ctest" --test-dir "$build" --output-on-failure --parallel 1 -R "^hetero_native_expert_" > "$build\ctest.log" 2>&1
set "ctest_rc=%errorlevel%"
>"$build\ctest.exit.txt" echo %ctest_rc%
exit /b %ctest_rc%
"@
    Set-Content -LiteralPath $batch -Value $batchText -Encoding ascii

    $preLaunchResources = Get-HostResourceSample
    $null = $samples.Add($preLaunchResources)
    if ($preLaunchResources.available_physical_bytes -lt $minFreeRam -or
        $null -eq $preLaunchResources.commit_headroom_bytes -or $preLaunchResources.commit_headroom_bytes -lt $minCommitHeadroom) {
        throw 'Fresh admission failed immediately before launch: require >=12 GiB available RAM and >=4 GiB commit headroom.'
    }
    $buildProcess = Start-Process -FilePath $env:ComSpec -ArgumentList ('/d /c "' + $batch + '"') -PassThru -WindowStyle Hidden
    $buildProcess.PriorityClass = [System.Diagnostics.ProcessPriorityClass]::Idle
    $prioritySet = $true
    $rootIdentity = Get-OwnedRootIdentity -RootProcessId $buildProcess.Id -BatchPath $batch `
        -ExpectedExecutable $env:ComSpec -ExpectedStartTimeUtc $buildProcess.StartTime.ToUniversalTime().ToString('o')
    [ordered]@{
        run_id = $RunId
        process_id = $rootIdentity.process_id
        create_time_utc = $rootIdentity.create_time_utc
        executable_path = $rootIdentity.executable_path
        command_line = $rootIdentity.command_line
        owned_batch_absolute_path = $rootIdentity.batch_path
        priority = 'Idle'
        cmake_target = @('hetero_native_expert', 'hetero_native_expert_metadata_fixture')
        parallel_jobs = 1
        gpu_executed = $false
        model_cli_executed = $false
        build_directory = $build
    } | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath (Join-Path $build 'process.json') -Encoding utf8
    [DateTimeOffset]::UtcNow.ToString('o') | Set-Content -LiteralPath (Join-Path $build 'process-identity-verified.txt') -Encoding ascii

    while ($true) {
        $buildProcess.Refresh()
        if ($buildProcess.HasExited) { break }
        Start-Sleep -Seconds 5
        $sample = Get-HostResourceSample
        $null = $samples.Add($sample)
        $buildProcess.Refresh()
        $gateOk = $sample.available_physical_bytes -ge $minFreeRam -and
                  $null -ne $sample.commit_headroom_bytes -and $sample.commit_headroom_bytes -ge $minCommitHeadroom
        if (-not $gateOk -and -not $buildProcess.HasExited) {
            $resourceGateStopped = $true
            try { $stopResult = Stop-OwnedProcessTree -RootIdentity $rootIdentity }
            catch { $stopResult = [pscustomobject]@{ complete = $false; stopped_process_ids = @(); diagnostics = @("Identity-checked stop failed without killing an unverified process: $($_.Exception.Message)") } }
            break
        }
    }
    $buildProcess.Refresh()
    if ($buildProcess.HasExited) { $batchExit = [int]$buildProcess.ExitCode }
    else { $batchExit = 90 }
    if ($resourceGateStopped) {
        'Resource gate crossed; owned build process tree was stopped. Logs and partial outputs retained.' |
            Set-Content -LiteralPath (Join-Path $build 'resource-gate-stop.txt') -Encoding utf8
    }
} catch {
    $scriptFailure = $_.Exception.ToString()
    if ($buildProcess -and -not $buildProcess.HasExited) {
        if ($rootIdentity) {
            try { $stopResult = Stop-OwnedProcessTree -RootIdentity $rootIdentity }
            catch { $stopResult = [pscustomobject]@{ complete = $false; stopped_process_ids = @(); diagnostics = @("Identity-checked stop failed without killing an unverified process: $($_.Exception.Message)") } }
        } else { $stopResult = [pscustomobject]@{ complete = $false; stopped_process_ids = @(); diagnostics = @('No verified process identity was captured; no process was terminated.') } }
    }
}

$runFinish = [DateTimeOffset]::UtcNow
$finishedSample = Get-HostResourceSample
$null = $samples.Add($finishedSample)
$headAfter = (& git -C $repo rev-parse HEAD).Trim()
$branchAfter = (& git -C $repo branch --show-current).Trim()
$ggmlHeadAfter = (& git -C $ggml rev-parse HEAD).Trim()
$sourceAfter = @()
$ggmlAfter = @()
$binarySnapshot = @()
try {
    $sourceAfter = Get-SourceSnapshot -SourceRoot $repo -RelativePaths $repoSourceFiles
    $ggmlAfter = Get-SourceSnapshot -SourceRoot $ggml -RelativePaths $ggmlSourceFiles
    $sourceAfter | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath (Join-Path $build 'source-after.json') -Encoding utf8
    $ggmlAfter | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath (Join-Path $build 'ggml-after.json') -Encoding utf8
    $binarySnapshot = Get-BinarySnapshot -BuildPath $build
    $binarySnapshot | ConvertTo-Json -Depth 6 | Set-Content -LiteralPath (Join-Path $build 'binary-hashes.json') -Encoding utf8
} catch {
    if (-not $scriptFailure) { $scriptFailure = "Post-build receipt collection failed: $($_.Exception.Message)" }
}

$sourceChanged = $false
if ($sourceAfter.Count -eq $sourceBefore.Count) {
    for ($i = 0; $i -lt $sourceBefore.Count; $i++) {
        if ($sourceBefore[$i].path -ne $sourceAfter[$i].path -or $sourceBefore[$i].sha256 -ne $sourceAfter[$i].sha256) { $sourceChanged = $true }
    }
} else { $sourceChanged = $true }
$headChanged = $headBefore -ne $headAfter
$ggmlChanged = $false
if ($ggmlAfter.Count -eq $ggmlBefore.Count) {
    for ($i = 0; $i -lt $ggmlBefore.Count; $i++) {
        if ($ggmlBefore[$i].path -ne $ggmlAfter[$i].path -or $ggmlBefore[$i].sha256 -ne $ggmlAfter[$i].sha256) { $ggmlChanged = $true }
    }
} else { $ggmlChanged = $true }
$ggmlHeadChanged = $ggmlHead -ne $ggmlHeadAfter

$configureExit = Read-ExitCodeFile (Join-Path $build 'configure.exit.txt')
$buildExit = Read-ExitCodeFile (Join-Path $build 'build.exit.txt')
$ctestExit = Read-ExitCodeFile (Join-Path $build 'ctest.exit.txt')
$allStagesSucceeded = $configureExit -eq 0 -and $buildExit -eq 0 -and $ctestExit -eq 0 -and $batchExit -eq 0
$clPath = $null
$clSha = $null
$toolPathsLog = Join-Path $build 'tool-paths.log'
if (Test-Path -LiteralPath $toolPathsLog) {
    $clCandidate = (Get-Content -LiteralPath $toolPathsLog -TotalCount 1).Trim()
    if (Test-Path -LiteralPath $clCandidate -PathType Leaf) {
        $clPath = $clCandidate
        $clSha = (Get-FileHash -LiteralPath $clCandidate -Algorithm SHA256).Hash.ToLowerInvariant()
    }
}
$status = if ($resourceGateStopped -and $stopResult -and $stopResult.complete) { 'stopped_by_resource_gate' }
          elseif ($resourceGateStopped) { 'resource_gate_stop_identity_failure' }
          elseif ($scriptFailure) { 'failed' }
          elseif ($allStagesSucceeded -and -not $sourceChanged -and -not $headChanged -and -not $ggmlChanged -and -not $ggmlHeadChanged) { 'success' }
          elseif ($sourceChanged -or $headChanged -or $ggmlChanged -or $ggmlHeadChanged) { 'source_changed_during_build' }
          else { 'failed' }
$ramValues = @($samples | ForEach-Object { [int64]$_.available_physical_bytes })
$headroomValues = @($samples | Where-Object { $null -ne $_.commit_headroom_bytes } | ForEach-Object { [int64]$_.commit_headroom_bytes })
$osAtFinish = Get-CimInstance Win32_OperatingSystem
$totalRam = [int64]$osAtFinish.TotalVisibleMemorySize * 1024
$receipt = [ordered]@{
    schema_version = 1
    run_id = $RunId
    status = $status
    failure_reason = $scriptFailure
    start_utc = $runStart.ToString('o')
    finish_utc = $runFinish.ToString('o')
    duration_seconds = [math]::Round(($runFinish - $runStart).TotalSeconds, 3)
    source = [ordered]@{
        repo = $repo
        branch_at_start = $branchBefore
        branch_at_finish = $branchAfter
        head_at_start = $headBefore
        head_at_finish = $headAfter
        head_changed_during_build = $headChanged
        source_changed_during_build = $sourceChanged
        tracked_and_untracked_source_hashes_before = $sourceBefore
        tracked_and_untracked_source_hashes_after = $sourceAfter
    }
    ggml = [ordered]@{
        path = $ggml
        pinned_head = $ggmlPin
        head_at_start = $ggmlHead
        head_at_finish = $ggmlHeadAfter
        head_changed_during_build = $ggmlHeadChanged
        clean_at_start = $true
        changed_during_build = $ggmlChanged
        source_hashes_before = $ggmlBefore
        source_hashes_after = $ggmlAfter
    }
    toolchain = [ordered]@{
        cmake_path = $cmake
        cmake_version = $cmakeVersion
        cmake_sha256 = (Get-FileHash -LiteralPath $cmake -Algorithm SHA256).Hash.ToLowerInvariant()
        ninja_path = $ninja
        ninja_version = $ninjaVersion
        ninja_sha256 = (Get-FileHash -LiteralPath $ninja -Algorithm SHA256).Hash.ToLowerInvariant()
        ctest_path = $ctest
        ctest_sha256 = (Get-FileHash -LiteralPath $ctest -Algorithm SHA256).Hash.ToLowerInvariant()
        vcvars64_path = $vcvars
        vcvars64_sha256 = (Get-FileHash -LiteralPath $vcvars -Algorithm SHA256).Hash.ToLowerInvariant()
        cl_path = $clPath
        cl_sha256 = $clSha
        cl_version_log = Join-Path $build 'cl-version.log'
        local_dll_changes = $false
        global_environment_changed = $false
        cuda_or_hip_compiler_invoked = $false
        process_priority = if ($prioritySet) { 'Idle' } else { 'not_started' }
    }
    process_identity = $rootIdentity
    stop_result = $stopResult
    cmake_options = [ordered]@{
        generator = 'Ninja'
        configuration = 'Release'
        targets = @('hetero_native_expert', 'hetero_native_expert_metadata_fixture')
        parallel_jobs = 1
        STRATA_ENABLE_CUDA = 'OFF'
        STRATA_ENABLE_HIP = 'OFF'
        STRATA_HIP_GFX906 = 'OFF'
        STRATA_ENABLE_SYCL = 'OFF'
        STRATA_PREFILL_MMQ = 'OFF'
        STRATA_MMQ_KQUANTS = 'OFF'
        STRATA_ORCA_Q4KS_MMQ = 'OFF'
        STRATA_NATIVE_EXPERTS = 'ON'
        STRATA_BUILD_TESTS = 'OFF'
        STRATA_PORTABLE = 'ON'
        STRATA_ISA_FLOOR = ''
        STRATA_GGML_DIR = $ggml
    }
    actual_exit_codes = [ordered]@{ configure = $configureExit; build = $buildExit; ctest = $ctestExit; command_shell = $batchExit }
    host = [ordered]@{
        ram_total_bytes = $totalRam
        ram_min_available_sampled_bytes = ($ramValues | Measure-Object -Minimum).Minimum
        commit_headroom_min_sampled_bytes = if ($headroomValues.Count) { ($headroomValues | Measure-Object -Minimum).Minimum } else { $null }
        ram_floor_bytes = $minFreeRam
        commit_headroom_floor_bytes = $minCommitHeadroom
        sample_interval_seconds = 5
        sample_count = $samples.Count
        samples = @($samples)
    }
    binary_artifacts = $binarySnapshot
    logs = [ordered]@{
        configure = (Join-Path $build 'configure.log')
        build = (Join-Path $build 'build.log')
        ctest = (Join-Path $build 'ctest.log')
        vcvars = (Join-Path $build 'vcvars.log')
        tool_paths = (Join-Path $build 'tool-paths.log')
        cl_version = (Join-Path $build 'cl-version.log')
        cmake_version = (Join-Path $build 'cmake-version.log')
        ninja_version = (Join-Path $build 'ninja-version.log')
    }
    build_directory = $build
    gpu_executed = $false
    model_or_engine_cli_executed = $false
}

if (Test-Path -LiteralPath $receiptPath) { throw "Receipt unexpectedly exists; preserving all files: $receiptPath" }
$receipt | ConvertTo-Json -Depth 12 | Set-Content -LiteralPath $receiptPath -Encoding utf8
Write-Output ([ordered]@{ run_id = $RunId; status = $status; build_directory = $build; receipt = $receiptPath } | ConvertTo-Json -Compress)
if ($status -ne 'success') { throw "Native CPU build helper finished with status '$status'; see retained logs and receipt at $receiptPath" }
