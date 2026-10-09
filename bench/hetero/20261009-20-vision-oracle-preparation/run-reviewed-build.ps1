$ErrorActionPreference = 'Stop'
$root = 'C:\Users\DC\Documents\ChatGPT\Strata-Hetero'
$evidence = Join-Path $root 'bench\hetero\20261009-20-vision-oracle-preparation\build-evidence'
$build = 'E:\Strata-Hetero-data\build\vision-oracle-cpu-20261009-20'
$cmake = 'E:\Strata-Hetero-data\venv\Lib\site-packages\cmake\data\bin\cmake.exe'
$ninja = 'E:\Strata-Hetero-data\venv\Scripts\ninja.exe'
$python = 'E:\Strata-Hetero-data\venv\Scripts\python.exe'
$vcvars = 'C:\Program Files (x86)\Microsoft Visual Studio\18\BuildTools\VC\Auxiliary\Build\vcvars64.bat'
$llama = 'E:\Strata-Hetero-data\source\llama-3cf0325'
$source = Join-Path $root 'tools\vision'

function Get-Sample {
  $code = 'import json; from tools.hetero_resources import windows_memory_global; print(json.dumps(windows_memory_global()))'
  $raw = & $python -B -c $code
  if ($LASTEXITCODE -ne 0) {
    $failure = [ordered]@{ utc = [DateTime]::UtcNow.ToString('o'); phase = 'resource-preflight';
      error = "tools.hetero_resources sampler exited $LASTEXITCODE" }
    $failure | ConvertTo-Json -Compress | Set-Content -LiteralPath (Join-Path $evidence 'resource-sampler-failure.json') -Encoding UTF8
    throw $failure.error
  }
  $memory = $raw | ConvertFrom-Json
  if ($null -eq $memory.physical_available_bytes -or $null -eq $memory.commit_available_bytes) {
    $failure = [ordered]@{ utc = [DateTime]::UtcNow.ToString('o'); phase = 'resource-preflight';
      error = 'existing hetero_resources sampler returned incomplete physical/commit data'; telemetry = $memory }
    $failure | ConvertTo-Json -Depth 6 -Compress | Set-Content -LiteralPath (Join-Path $evidence 'preflight-failure.json') -Encoding UTF8
    throw 'Existing resource sampler did not provide required physical/commit telemetry'
  }
  [pscustomobject]@{
    utc = [DateTime]::UtcNow.ToString('o')
    available_physical_bytes = [UInt64]$memory.physical_available_bytes
    available_commit_bytes = [UInt64]$memory.commit_available_bytes
  }
}

function Write-Sample($sample) {
  $sample | ConvertTo-Json -Compress | Add-Content -LiteralPath (Join-Path $evidence 'resource-samples.jsonl') -Encoding UTF8
  return ($sample.available_physical_bytes -ge 12GB -and $sample.available_commit_bytes -ge 4GB)
}

function Get-TreeIds([int]$rootPid) {
  $all = @(Get-CimInstance Win32_Process)
  $ids = [System.Collections.Generic.HashSet[int]]::new()
  [void]$ids.Add($rootPid)
  $changed = $true
  while ($changed) {
    $changed = $false
    foreach ($p in $all) {
      if ($ids.Contains([int]$p.ParentProcessId) -and $ids.Add([int]$p.ProcessId)) { $changed = $true }
    }
  }
  return @($ids)
}

function Invoke-Phase([string]$name, [string]$arguments) {
  $log = Join-Path $evidence "$name.log"
  $exitFile = Join-Path $evidence "$name.exit.txt"
  $batch = Join-Path $evidence "$name.cmd"
  @"
@echo off
call "$vcvars" >nul 2>&1
"$cmake" $arguments >> "$log" 2>&1
set "RC=%ERRORLEVEL%"
>"$exitFile" echo %RC%
exit /b %RC%
"@ | Set-Content -LiteralPath $batch -Encoding ASCII

  $pre = Get-Sample
  if (-not (Write-Sample $pre)) { throw "resource gate failed before ${name}: $($pre | ConvertTo-Json -Compress)" }
  $p = Start-Process -FilePath "$env:WINDIR\System32\cmd.exe" -ArgumentList @('/d','/c','call',"`"$batch`"") -WindowStyle Hidden -PassThru
  $p.PriorityClass = 'Idle'
  while (-not $p.HasExited) {
    foreach ($id in (Get-TreeIds $p.Id)) {
      try { (Get-Process -Id $id -ErrorAction Stop).PriorityClass = 'Idle' } catch { }
    }
    $sample = Get-Sample
    $gate = Write-Sample $sample
    if (-not $gate) {
      # Do not kill unrelated work; mark the run and wait for this owned phase
      # to finish before refusing any next phase.
      Set-Content -LiteralPath (Join-Path $evidence "$name.resource-gate-fell.txt") -Value ($sample | ConvertTo-Json -Compress)
    }
    Start-Sleep -Seconds 5
    $p.Refresh()
  }
  $p.WaitForExit()
  $exit = [int]$p.ExitCode
  if (-not (Test-Path -LiteralPath $exitFile)) { Set-Content -LiteralPath $exitFile -Value $exit }
  return $exit
}

New-Item -ItemType Directory -Path $evidence -Force | Out-Null
if (Test-Path -LiteralPath $build) { throw "Build directory already exists; refusing reuse: $build" }
foreach ($path in @($cmake,$ninja,$python,$vcvars,$llama,(Join-Path $source 'CMakeLists.txt'),(Join-Path $source 'strata_vision.cpp'))) {
  if (-not (Test-Path -LiteralPath $path)) { throw "Required pinned path missing: $path" }
}

$configureArgs = "-S `"$source`" -B `"$build`" -G Ninja -DCMAKE_MAKE_PROGRAM=`"$ninja`" -DCMAKE_BUILD_TYPE=Release -DLLAMA_DIR=`"$llama`" -DLLAMA_BUILD_COMMON=OFF -DLLAMA_BUILD_MTMD=ON -DLLAMA_BUILD_TOOLS=OFF -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF -DLLAMA_BUILD_SERVER=OFF -DLLAMA_BUILD_APP=OFF -DSTRATA_VISION_CUDA=OFF -DSTRATA_PORTABLE=ON -DGGML_CUDA=OFF -DGGML_HIP=OFF -DGGML_SYCL=OFF -DGGML_VULKAN=OFF -DGGML_OPENCL=OFF -DGGML_NATIVE=OFF -DGGML_OPENMP=OFF"
$configureExit = Invoke-Phase 'configure-retry04' $configureArgs
if ($configureExit -ne 0) { exit $configureExit }
$cache = Join-Path $build 'CMakeCache.txt'
$ninjaFile = Join-Path $build 'build.ninja'
if (-not (Test-Path -LiteralPath $cache) -or -not (Test-Path -LiteralPath $ninjaFile)) {
  throw 'Configure returned success without CMakeCache.txt and build.ninja; refusing to claim configure success'
}
$gateMarker = Join-Path $evidence 'configure-retry04.resource-gate-fell.txt'
if (Test-Path -LiteralPath $gateMarker) { throw 'Resource gate fell during configure; refusing to start build phase' }

$buildExit = Invoke-Phase 'build-retry04' "--build `"$build`" --target strata-vision --parallel 1"
if ($buildExit -ne 0) { exit $buildExit }

$exe = Join-Path $build 'bin\strata-vision.exe'
if (-not (Test-Path -LiteralPath $exe)) { throw "Expected helper missing after successful build: $exe" }
$binary = Get-Item -LiteralPath $exe
$sha = (Get-FileHash -LiteralPath $exe -Algorithm SHA256).Hash.ToLowerInvariant()
$llamaHead = (& git -C $llama rev-parse HEAD).Trim()
$strataHead = (& git -C $root rev-parse HEAD).Trim()
$receipt = [ordered]@{
  schema = 'strata-vision-oracle-build-evidence-v1'
  status = 'built_not_run'
  source_head = $strataHead
  pinned_llama_head = $llamaHead
  target = 'strata-vision'
  configure_exit = $configureExit
  build_exit = $buildExit
  binary = [ordered]@{ path = $exe; size_bytes = $binary.Length; sha256 = $sha }
  run = [ordered]@{ helper_started = $false; model_or_mmproj_loaded = $false; core_created = $false; gpu_executed = $false }
  configure_log = 'configure.log'
  build_log = 'build.log'
  resource_samples = 'resource-samples.jsonl'
}
$receipt | ConvertTo-Json -Depth 6 | Set-Content -LiteralPath (Join-Path $evidence 'build-receipt.json') -Encoding UTF8
