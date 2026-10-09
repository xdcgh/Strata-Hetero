$ErrorActionPreference='Stop'
Set-StrictMode -Version 2.0
$repo='C:\Users\DC\Documents\ChatGPT\Strata-Hetero'
$run=Join-Path $repo 'bench\hetero\20261009-46-upstream41-merged-regression'
$python='E:\Strata-Hetero-data\venv\Scripts\python.exe'
$pwsh='C:\Users\DC\.cache\codex-runtimes\codex-primary-runtime\dependencies\native\powershell\pwsh.exe'
$commandsPath=Join-Path $run 'commands.jsonl'
$resultsPath=Join-Path $run 'results.json'
$summaryPath=Join-Path $run 'summary.md'
$failedPath=Join-Path $run 'failure.json'
if((Test-Path $commandsPath) -or (Test-Path $resultsPath) -or (Test-Path $summaryPath) -or (Test-Path $failedPath)){throw 'run46 result evidence exists; refusing overwrite'}
if((Get-FileHash $python -Algorithm SHA256).Hash.ToLowerInvariant() -ne '16a377d348c3a0c8d3a58d222d96807065dd0dcd9438c6025900c842ec836603'){throw 'main venv Python hash differs from validated run33 environment'}
$headBefore=(git -C $repo rev-parse HEAD).Trim()
$statusBefore=@(git -C $repo status --short)
$pythonInfoRaw=& $python -B -c "import sys,json,importlib.util,numpy,psutil,PIL;print(json.dumps({'executable':sys.executable,'version':sys.version.split()[0],'numpy':numpy.__version__,'psutil':psutil.__version__,'pillow':PIL.__version__,'openvino_available':importlib.util.find_spec('openvino') is not None,'yaml_available':importlib.util.find_spec('yaml') is not None},separators=(',',':')))"
if($LASTEXITCODE -ne 0){throw 'main venv package metadata probe failed'}
$pythonInfo=($pythonInfoRaw -join "`n")|ConvertFrom-Json -ErrorAction Stop
$receipt=[ordered]@{schema_version=1;run_id='20261009-46-upstream41-merged-regression';status='running';source_head=$headBefore;branch=(git -C $repo branch --show-current).Trim();python=$python;python_sha256=(Get-FileHash $python -Algorithm SHA256).Hash.ToLowerInvariant();python_environment=$pythonInfo;pwsh=$pwsh;pwsh_sha256=(Get-FileHash $pwsh -Algorithm SHA256).Hash.ToLowerInvariant();execution_scope='sequential pure unit/fake-fixture commands only';intel_venv_used=$false;openvino_core_initialized=$false;model_api_gpu_download_compile=false;start_utc=[DateTime]::UtcNow.ToString('o');suites=[Collections.Generic.List[object]]::new()}
$repoToolsPythonPath=$repo+';'+(Join-Path $repo 'tools')
function Record-CommandEvent($Event){$bytes=[Text.UTF8Encoding]::new($false).GetBytes((ConvertTo-Json -InputObject $Event -Depth 10 -Compress)+"`n");$f=[IO.File]::Open($commandsPath,[IO.FileMode]::Append,[IO.FileAccess]::Write,[IO.FileShare]::Read);try{$f.Write($bytes,0,$bytes.Length);$f.Flush($true)}finally{$f.Dispose()}}
function Invoke-PythonSuite([string]$Name,[string[]]$SuiteArgs,[string]$ProcessPythonPath=''){
    $log=Join-Path (Join-Path $run 'logs') ($Name+'.log')
    if(Test-Path $log){throw "log path exists: $log"}
    $oldPythonPath=$env:PYTHONPATH
    $started=[DateTime]::UtcNow
    Write-Output "SUITE_START $Name"
    Record-CommandEvent @{event='start';suite=$Name;utc=$started.ToString('o');exe=$python;argv=$SuiteArgs;cwd=$repo;pythonpath=$ProcessPythonPath}
    try{
        if($ProcessPythonPath){$env:PYTHONPATH=$ProcessPythonPath}
        & $python @SuiteArgs *> $log
        $exitCode=[int]$LASTEXITCODE
    }finally{$env:PYTHONPATH=$oldPythonPath}
    $ended=[DateTime]::UtcNow;$text=Get-Content -LiteralPath $log -Raw
    $ranMatch=[regex]::Match($text,'Ran\s+(\d+)\s+tests?\s+in')
    $ran=if($ranMatch.Success){[int]$ranMatch.Groups[1].Value}else{$null}
    $skipMatch=[regex]::Match($text,'OK\s+\(skipped=(\d+)\)')
    $skipped=if($skipMatch.Success){[int]$skipMatch.Groups[1].Value}else{0}
    $failed=([regex]::Matches($text,'(?m)^(FAIL|ERROR):')).Count
    $passed=if($null -ne $ran){[Math]::Max(0,$ran-$skipped-$failed)}else{$null}
    $entry=[ordered]@{suite=$Name;exe=$python;argv=$SuiteArgs;cwd=$repo;process_local_pythonpath=$ProcessPythonPath;started_utc=$started.ToString('o');ended_utc=$ended.ToString('o');duration_seconds=[Math]::Round(($ended-$started).TotalSeconds,6);exit_code=$exitCode;tests_run=$ran;tests_passed=$passed;tests_skipped=$skipped;tests_failed=$failed;log=$log;log_sha256=(Get-FileHash $log -Algorithm SHA256).Hash.ToLowerInvariant()}
    $receipt.suites.Add($entry);Record-CommandEvent @{event='finish';suite=$Name;utc=$ended.ToString('o');exit_code=$exitCode;tests_run=$ran;tests_passed=$passed;tests_skipped=$skipped;tests_failed=$failed;log_sha256=$entry.log_sha256}
    Write-Output "SUITE_END $Name exit=$exitCode tests=$ran passed=$passed skipped=$skipped failed=$failed duration=$($entry.duration_seconds)s"
    if($exitCode -ne 0 -or $null -eq $ran -or $failed -ne 0){$receipt.status='failed_stopped';$receipt.failure_suite=$Name;Write-FinalEvidence;throw "suite failed or had no unittest count: $Name exit=$exitCode"}
}
function Invoke-PwshFixture([string]$Name,[string]$FixturePath){
    $log=Join-Path (Join-Path $run 'logs') ($Name+'.log');if(Test-Path $log){throw "log path exists: $log"}
    $started=[DateTime]::UtcNow;Record-CommandEvent @{event='start';suite=$Name;utc=$started.ToString('o');exe=$pwsh;argv=@('-NoProfile','-File',$FixturePath);cwd=$repo;pythonpath=''}
    Write-Output "SUITE_START $Name"
    & $pwsh -NoProfile -File $FixturePath *> $log;$exitCode=[int]$LASTEXITCODE;$ended=[DateTime]::UtcNow;$text=Get-Content -LiteralPath $log -Raw
    $out=$text|ConvertFrom-Json -ErrorAction Stop;if($exitCode -ne 0 -or $out.status -ne 'passed'){throw "PowerShell fixture failed: $Name"}
    $entry=[ordered]@{suite=$Name;exe=$pwsh;argv=@('-NoProfile','-File',$FixturePath);cwd=$repo;started_utc=$started.ToString('o');ended_utc=$ended.ToString('o');duration_seconds=[Math]::Round(($ended-$started).TotalSeconds,6);exit_code=$exitCode;fixture_cases=$out.fixture_count;fixture_summary=$out.cases;tests_failed=0;log=$log;log_sha256=(Get-FileHash $log -Algorithm SHA256).Hash.ToLowerInvariant()}
    $receipt.suites.Add($entry);Record-CommandEvent @{event='finish';suite=$Name;utc=$ended.ToString('o');exit_code=$exitCode;fixture_cases=$out.fixture_count;log_sha256=$entry.log_sha256}
    Write-Output "SUITE_END $Name exit=$exitCode fixture_cases=$($out.fixture_count) duration=$($entry.duration_seconds)s"
}
function Write-FinalEvidence {
    if(Test-Path $resultsPath){return}
    $receipt.ended_utc=[DateTime]::UtcNow.ToString('o');$receipt.source_head_after=(git -C $repo rev-parse HEAD).Trim();$receipt.status_after=@(git -C $repo status --short)
    $receipt.tracked_diff_paths=@(git -C $repo diff --name-only);$receipt.staged_diff_paths=@(git -C $repo diff --cached --name-only)
    $receipt.source_head_unchanged=($receipt.source_head_after -eq $headBefore);$receipt.tracked_source_unchanged=($receipt.tracked_diff_paths.Count -eq 0 -and $receipt.staged_diff_paths.Count -eq 0)
    $pySuites=@($receipt.suites|Where-Object{$null -ne $_.tests_run});$receipt.python_tests_run=($pySuites|Measure-Object tests_run -Sum).Sum;$receipt.python_tests_passed=($pySuites|Measure-Object tests_passed -Sum).Sum;$receipt.python_tests_skipped=($pySuites|Measure-Object tests_skipped -Sum).Sum;$receipt.python_tests_failed=($pySuites|Measure-Object tests_failed -Sum).Sum
    $receipt.custom_fixture_cases=(@($receipt.suites|Where-Object{$null -ne $_.fixture_cases}|Measure-Object fixture_cases -Sum).Sum)
    $json=$receipt|ConvertTo-Json -Depth 16
    $b=[Text.UTF8Encoding]::new($false).GetBytes($json+"`n");$f=[IO.File]::Open($resultsPath,[IO.FileMode]::CreateNew,[IO.FileAccess]::Write,[IO.FileShare]::Read);try{$f.Write($b,0,$b.Length);$f.Flush($true)}finally{$f.Dispose()}
    $md=@("# Upstream41 merged pure regression","","Status: $($receipt.status)","Source HEAD: ``$headBefore``","Python tests: $($receipt.python_tests_run) run, $($receipt.python_tests_passed) passed, $($receipt.python_tests_skipped) skipped, $($receipt.python_tests_failed) failed.","Additional fixture cases: $($receipt.custom_fixture_cases).","No compiler/model/API/GPU/Core/download path was run.","","Commands and complete logs are recorded in ``commands.jsonl`` and ``logs/``.") -join "`n"
    $mb=[Text.UTF8Encoding]::new($false).GetBytes($md+"`n");$mf=[IO.File]::Open($summaryPath,[IO.FileMode]::CreateNew,[IO.FileAccess]::Write,[IO.FileShare]::Read);try{$mf.Write($mb,0,$mb.Length);$mf.Flush($true)}finally{$mf.Dispose()}
}
try{
    $header=[ordered]@{event='run_start';utc=[DateTime]::UtcNow.ToString('o');source_head=$headBefore;branch=$receipt.branch;python=$python;python_sha256=$receipt.python_sha256;python_environment=$pythonInfo}
    $hb=[Text.UTF8Encoding]::new($false).GetBytes((ConvertTo-Json -InputObject $header -Compress)+"`n");$cf=[IO.File]::Open($commandsPath,[IO.FileMode]::CreateNew,[IO.FileAccess]::Write,[IO.FileShare]::Read);try{$cf.Write($hb,0,$hb.Length);$cf.Flush($true)}finally{$cf.Dispose()}
    $serve=@('-B','-m','unittest','-v','serve.test_parallel','serve.test_request_hardening','serve.test_responses','serve.test_server','serve.test_slot_alloc','serve.test_tool_name_space','serve.test_vision_busy','serve.test_runconfig','serve.test_security','serve.test_monitor','serve.test_prometheus')
    Invoke-PythonSuite 'serve-unit-modules' $serve
    $setup=@('-B','-m','unittest','-v','tools.test_calibrate','tools.test_setup_choices','tools.test_setup_compute_mode','tools.test_setup_engine_keep','tools.test_setup_hotfix_tag','tools.test_setup_remote_opt','tools.test_setup_risk','tools.test_setup_unsloth','tools.test_check_isa_guard')
    Invoke-PythonSuite 'calibrate-setup-pure-modules' $setup
    Invoke-PythonSuite 'tokenizer-piece-cache' @('-B','tools/test_strata_tokenizer.py','-v')
    $hetero=@('admission','bench','compare_quality','copy_verified','cpu_topology','extract_expert','inventory','native_activation','native_inputs','quality','resources','vision_preparation','vision_qwen3vl','xpu_service','xpu_transport','xpu_worker')
    foreach($name in $hetero){Invoke-PythonSuite ("hetero-$name") @('-B',(Join-Path 'tools' ("test_hetero_{0}.py" -f $name)),'-v') $repoToolsPythonPath}
    Invoke-PythonSuite 'hetero-prepare-cuda-mocked-archives' @('-B','tools/test_hetero_prepare_cuda.py','-v') $repoToolsPythonPath
    Invoke-PythonSuite 'hetero-planner' @('-B','tools/test_hetero_planner.py','-v') $repoToolsPythonPath
    Invoke-PythonSuite 'hetero-runtime' @('-B','tools/test_hetero_runtime.py','-v') $repoToolsPythonPath
    Invoke-PwshFixture 'hetero-jsonl-tail-pure-fixtures' (Join-Path $repo 'tools\test_hetero_jsonl_tail.ps1')
    $receipt.status='passed';Write-FinalEvidence
} catch {
    if($receipt.status -eq 'running'){$receipt.status='failed_stopped';$receipt.failure_type=$_.Exception.GetType().FullName;$receipt.failure=$_.Exception.Message;try{Write-FinalEvidence}catch{}}
    throw
}
