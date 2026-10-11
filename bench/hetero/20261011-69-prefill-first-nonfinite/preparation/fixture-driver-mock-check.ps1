$ErrorActionPreference='Stop'
Set-StrictMode -Version 2.0
$driverPath=Join-Path (Get-Location) 'bench/hetero/20261011-69-prefill-first-nonfinite/preparation/run_probe_fixture69.ps1'
$expectedDriverSha='1885cbbe9326eac2e369cfaafaa5a6f37400e2d9cc4aeecb6a4c6b41f7a6ba08'
$receiptPath=Join-Path (Get-Location) 'bench/hetero/20261011-69-prefill-first-nonfinite/preparation/fixture-driver-mock-check.receipt.json'
$tests=[System.Collections.Generic.List[object]]::new()
$script:MockLiveIdentity=$null
$script:MockStopCalls=[System.Collections.Generic.List[int]]::new()
$script:MockStopFailure=$false
function Assert-Check([bool]$condition,[string]$name) { if(-not $condition){throw "assertion failed: $name"}; $tests.Add([ordered]@{name=$name;passed=$true}) }
function Identity([int]$processId) { if($null -ne $script:MockLiveIdentity -and $script:MockLiveIdentity.pid -eq $processId){return $script:MockLiveIdentity}; return $null }
function Stop-Process { param([int]$Id,[string]$ErrorAction); $script:MockStopCalls.Add($Id); if($script:MockStopFailure){throw 'mock stop denied'} }
$status='failed';$errorText=$null;$sourceSha=$null
try {
    $sourceSha=(Get-FileHash -LiteralPath $driverPath -Algorithm SHA256).Hash.ToLowerInvariant()
    if($sourceSha -ne $expectedDriverSha){throw 'Pinned driver SHA changed; refusing extraction'}
    $tokens=$null;$parseErrors=$null
    $ast=[System.Management.Automation.Language.Parser]::ParseFile($driverPath,[ref]$tokens,[ref]$parseErrors)
    if($parseErrors.Count){throw 'Pinned source no longer parses'}
    $wanted=@('Same','ExpectedWorker','StopProven')
    foreach($name in $wanted){
        $defs=@($ast.EndBlock.Statements | Where-Object {$_ -is [System.Management.Automation.Language.FunctionDefinitionAst] -and $_.Name -eq $name})
        if($defs.Count -ne 1){throw "Expected exactly one $name definition"}
        Invoke-Expression $defs[0].Extent.Text
    }
    $owned=[pscustomobject]@{pid=481;parent=400;exe='C:\fixture\prefill_finite_probe_test.exe';created='2026-10-11T03:00:00.0000000Z';command='C:\fixture\prefill_finite_probe_test.exe --mock-run-69'}
    $required='--mock-run-69'
    Assert-Check (-not (ExpectedWorker $null $owned.exe $owned.parent $required)) 'nil identity is not approved'
    $bad=$owned.PSObject.Copy();$bad.exe='C:\other\strata.exe'
    Assert-Check (-not (ExpectedWorker $bad $owned.exe $owned.parent $required)) 'executable mismatch is rejected'
    $bad=$owned.PSObject.Copy();$bad.parent=401
    Assert-Check (-not (ExpectedWorker $bad $owned.exe $owned.parent $required)) 'parent mismatch is rejected'
    $bad=$owned.PSObject.Copy();$bad.command='C:\fixture\prefill_finite_probe_test.exe --other-run'
    Assert-Check (-not (ExpectedWorker $bad $owned.exe $owned.parent $required)) 'unique argv mismatch is rejected'
    Assert-Check (ExpectedWorker $owned $owned.exe $owned.parent $required) 'exact expected worker identity is approved'

    $script:MockLiveIdentity=$null
    $r=StopProven $null
    Assert-Check ($r.status -eq 'unproven_no_stop' -and $script:MockStopCalls.Count -eq 0) 'nil unproven identity never stops'
    $script:MockLiveIdentity=$null
    $r=StopProven $owned
    Assert-Check ($r.status -eq 'already_terminal' -and $script:MockStopCalls.Count -eq 0) 'already-terminal process is not stopped'

    foreach($field in @('parent','exe','created','command')) {
        $live=$owned.PSObject.Copy()
        switch($field){
            'parent' {$live.parent=402}
            'exe' {$live.exe='C:\other\prefill_finite_probe_test.exe'}
            'created' {$live.created='2026-10-11T03:00:01.0000000Z'}
            'command' {$live.command='C:\fixture\prefill_finite_probe_test.exe --different-run'}
        }
        $script:MockLiveIdentity=$live
        $r=StopProven $owned
        Assert-Check ($r.status -eq 'identity_mismatch_no_stop' -and $script:MockStopCalls.Count -eq 0) "live $field change prevents stop"
    }
    $script:MockLiveIdentity=$owned
    $r=StopProven $owned
    Assert-Check ($r.status -eq 'stopped_exact_owned' -and $script:MockStopCalls.Count -eq 1 -and $script:MockStopCalls[0] -eq 481) 'exact match stops only the proven PID once'
    $script:MockStopFailure=$true
    $r=StopProven $owned
    Assert-Check ($r.status -eq 'stop_failed' -and $r.error -eq 'mock stop denied' -and $script:MockStopCalls.Count -eq 2 -and $script:MockStopCalls[1] -eq 481) 'mock stop failure remains explicit'
    $status='pass'
} catch { $errorText=$_.Exception.ToString() }
$receipt=[ordered]@{schema_version=1;status=$status;driver_path=$driverPath;driver_sha256=$sourceSha;ast_parse_errors=0;functions_extracted=@('Same','ExpectedWorker','StopProven');test_count=$tests.Count;tests=@($tests);mocked_commands=@('Identity','Stop-Process');real_cim_accessed=$false;real_process_stop_attempted=$false;driver_executed=$false;model_or_engine_executed=$false;gpu_or_device_used=$false;build_or_test_binary_executed=$false;error=$errorText}
$json=$receipt|ConvertTo-Json -Depth 8
$b=[Text.UTF8Encoding]::new($false).GetBytes($json+"`n")
$f=[IO.File]::Open($receiptPath,[IO.FileMode]::CreateNew,[IO.FileAccess]::Write,[IO.FileShare]::Read)
try{$f.Write($b,0,$b.Length);$f.Flush($true)}finally{$f.Dispose()}
$receipt|ConvertTo-Json -Depth 8 -Compress
if($status -ne 'pass'){exit 1}