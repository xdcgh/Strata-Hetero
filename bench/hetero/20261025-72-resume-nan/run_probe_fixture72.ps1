param(
    [Parameter(Mandatory)][string]$BinaryPath,
    [Parameter(Mandatory)][string]$ExpectedBinarySha,
    [Parameter(Mandatory)][ValidatePattern('^[0-9a-f]{40}$')][string]$ExpectedSourceHead,
    [Parameter(Mandatory)][string]$BuildReceiptPath,
    [Parameter(Mandatory)][string]$OutputPath,
    [switch]$Execute
)
$ErrorActionPreference='Stop'
Set-StrictMode -Version 2.0
$repo='C:\Users\DC\Documents\ChatGPT\Strata-Hetero'
$py='E:\Strata-Hetero-data\venv\Scripts\python.exe'
$stdlibPy='C:\Python314\python.exe'
$sdk='E:\Strata-Hetero-data\toolchains\cuda-13.3.1'
$source='E:\Strata-Hetero-data\source\hetero-moe45-probe-71'
$sourceHead=$ExpectedSourceHead
$sampler=$null;$samplerIdentity=$null;$samplerChild=$null;$worker=$null;$workerIdentity=$null;$failure=$null;$runs=@();$samples=@();$syntheticStarted=$false;$stopActions=@()
function Sha([string]$path){(Get-FileHash -LiteralPath $path -Algorithm SHA256).Hash.ToLowerInvariant()}
function WriteNew([string]$path,$value){
    $bytes=[Text.UTF8Encoding]::new($false).GetBytes(($value|ConvertTo-Json -Depth 15)+"`n")
    $file=[IO.File]::Open($path,[IO.FileMode]::CreateNew,[IO.FileAccess]::Write,[IO.FileShare]::Read)
    try{$file.Write($bytes,0,$bytes.Length);$file.Flush($true)}finally{$file.Dispose()}
}
function Identity([int]$processId){
    $row=Get-CimInstance Win32_Process -Filter "ProcessId=$processId" -ErrorAction SilentlyContinue
    if(-not $row){return $null}
    [pscustomobject]@{pid=[int]$row.ProcessId;parent=[int]$row.ParentProcessId;exe=[string]$row.ExecutablePath;created=$row.CreationDate.ToUniversalTime().ToString('o');command=[string]$row.CommandLine}
}
function Same($a,$b){$a -and $b -and $a.pid -eq $b.pid -and $a.parent -eq $b.parent -and $a.exe -eq $b.exe -and $a.created -eq $b.created -and $a.command -eq $b.command}
function ExpectedWorker($identity,[string]$expectedExe,[int]$expectedParent,[string]$requiredCommand) {
    $identity -and $identity.exe -eq $expectedExe -and $identity.parent -eq $expectedParent -and $identity.command.Contains($requiredCommand)
}
function StopProven($provenIdentity) {
    if(-not $provenIdentity){return [ordered]@{status='unproven_no_stop'}}
    $live=Identity $provenIdentity.pid
    if(-not $live){return [ordered]@{status='already_terminal';identity=$provenIdentity}}
    if(-not(Same $live $provenIdentity)){return [ordered]@{status='identity_mismatch_no_stop';identity=$provenIdentity;live=$live}}
    try { Stop-Process -Id $provenIdentity.pid -ErrorAction Stop; return [ordered]@{status='stopped_exact_owned';identity=$provenIdentity} }
    catch { return [ordered]@{status='stop_failed';identity=$provenIdentity;error=$_.Exception.Message} }
}
function FreshMemory {
    if(-not(Test-Path -LiteralPath $resource)){return $false}
    $row=Get-Content -LiteralPath $resource -Tail 1|ConvertFrom-Json -DateKind String
    $age=([DateTimeOffset]::UtcNow-[DateTimeOffset]::Parse($row.sampled_at_utc)).TotalSeconds
    return -not $row.error -and $age -ge -2 -and $age -le 5 -and $null -ne $row.physical_available_bytes -and $null -ne $row.commit_available_bytes -and [int64]$row.physical_available_bytes -ge 12GB -and [int64]$row.commit_available_bytes -ge 4GB
}
$BinaryPath=[IO.Path]::GetFullPath($BinaryPath);$OutputPath=[IO.Path]::GetFullPath($OutputPath)
$expectedFixture='E:\Strata-Hetero-data\build\cuda-moe45-probe71-01\prefill_finite_probe_test.exe'
if($BinaryPath -ne $expectedFixture){throw 'Only the pinned synthetic prefill_finite_probe_test.exe may be executed'}
if(Test-Path -LiteralPath $OutputPath){throw 'Fixture output path must be absent'}
$build=Get-Content -LiteralPath $BuildReceiptPath -Raw|ConvertFrom-Json -DateKind String
if($build.status -ne 'success' -or $build.source_after.head -ne $sourceHead -or -not $build.source_after.clean -or $build.test_binary.sha256 -ne $ExpectedBinarySha -or $build.test_binary.path -ne $BinaryPath -or (Sha $BinaryPath) -ne $ExpectedBinarySha){throw 'Fixture binary/build receipt identity mismatch'}
$head=(& git -C $source rev-parse HEAD).Trim();if($LASTEXITCODE -ne 0){throw 'source HEAD read failed'}
$dirty=@(& git -C $source status --porcelain=v1);if($LASTEXITCODE -ne 0){throw 'source status read failed'}
if($head -ne $sourceHead -or $dirty.Count){throw 'Source changed since build'}
$samplerScript=Join-Path $build.build_directory 'resource_sampler.py'
if(-not(Test-Path -LiteralPath $samplerScript)){throw 'Build sampler script missing'}
$samplerScriptSha=Sha $samplerScript
if(-not $Execute){[ordered]@{status='fixture_prepared_no_start';binary=$BinaryPath;sha256=$ExpectedBinarySha;source_head=$head;output_absent=$true;execute_switch_present=$false;sampler_script_sha256=$samplerScriptSha}|ConvertTo-Json -Compress;exit 0}
[void][IO.Directory]::CreateDirectory($OutputPath)
$resource=Join-Path $OutputPath 'resources.jsonl';$stop=Join-Path $OutputPath 'resources.STOP'
try {
    & wsl.exe --shutdown
    if($LASTEXITCODE -ne 0){throw 'WSL shutdown failed before fixture admission'}
    & $stdlibPy -I -B -X utf8 (Join-Path $repo 'tools\hetero_admission.py') --observe --output (Join-Path $OutputPath 'admission.json') --required-vram-mib 46000 --minimum-ram-gib 12 --check-wsl-current --inspect-running-wsl
    if($LASTEXITCODE -ne 0){throw 'Strict fixture admission failed; no fixture process started'}
    $env:PATH="$sdk\bin\x64;$sdk\bin;$env:PATH"
    $env:CUDA_VISIBLE_DEVICES='0'
    $sampler=Start-Process -FilePath $py -ArgumentList @('-u','-B','-X','utf8',$samplerScript,$repo,$resource,$stop) -WorkingDirectory $repo -WindowStyle Hidden -RedirectStandardOutput (Join-Path $OutputPath 'sampler.stdout.log') -RedirectStandardError (Join-Path $OutputPath 'sampler.stderr.log') -PassThru
    $observed=Identity $sampler.Id
    if(-not(ExpectedWorker $observed $py $PID $resource)){throw 'Sampler launcher identity mismatch'}
    $samplerIdentity=$observed
    $limit=[DateTimeOffset]::UtcNow.AddSeconds(10)
    do {
        $child=Get-CimInstance Win32_Process | Where-Object {$_.ParentProcessId -eq $sampler.Id -and $_.ExecutablePath -eq $stdlibPy -and $_.CommandLine.Contains($resource)} | Select-Object -First 1
        if($child){
            $observedChild=Identity ([int]$child.ProcessId)
            if(ExpectedWorker $observedChild $stdlibPy $sampler.Id $resource){$samplerChild=$observedChild}
        }
        if($samplerChild -and (FreshMemory)){break}
        Start-Sleep -Milliseconds 100
    }while([DateTimeOffset]::UtcNow -lt $limit)
    if(-not $samplerChild -or -not(FreshMemory)){throw 'Fresh sampler identity or 12/4 reserve missing'}
    WriteNew (Join-Path $OutputPath 'sampler-process.json') ([ordered]@{launcher=$samplerIdentity;child=$samplerChild;script_sha256=$samplerScriptSha})
    foreach($mode in @('disabled','synthetic')) {
        if(-not(FreshMemory) -or (Sha $BinaryPath) -ne $ExpectedBinarySha){throw 'Fixture fresh memory/binary gate failed'}
        $out=Join-Path $OutputPath ($mode+'.stdout.log');$err=Join-Path $OutputPath ($mode+'.stderr.log')
        $options=@{FilePath=$BinaryPath;WorkingDirectory=$repo;WindowStyle='Hidden';RedirectStandardOutput=$out;RedirectStandardError=$err;PassThru=$true}
        if($mode -eq 'disabled'){$options.ArgumentList=@('--disabled-only')}
        $worker=Start-Process @options
        if($mode -eq 'synthetic'){$syntheticStarted=$true}
        $workerIdentity=$null
        $observed=Identity $worker.Id
        # Very short fixtures may finish before CIM can bind them. The process
        # handle, actual PID, binary hash, argv and exit code remain recorded.
        if($observed) {
            if(-not(ExpectedWorker $observed $BinaryPath $PID $BinaryPath)){throw 'Fixture worker identity mismatch'}
            $workerIdentity=$observed
        }
        WriteNew (Join-Path $OutputPath ($mode+'.process.json')) ([ordered]@{pid=$worker.Id;identity=$workerIdentity;binary_sha256=$ExpectedBinarySha;argv=if($mode -eq 'disabled'){@('--disabled-only')}else{@()}})
        $limit=[DateTimeOffset]::UtcNow.AddSeconds(120)
        while(-not $worker.HasExited) {
            if(-not(FreshMemory)){throw 'Fixture resource telemetry stale, unknown or below 12/4'}
            if([DateTimeOffset]::UtcNow -gt $limit){throw 'Fixture exceeded 120-second bound'}
            Start-Sleep -Milliseconds 100;$worker.Refresh()
        }
        $worker.WaitForExit();$worker.Refresh();$rc=$worker.ExitCode
        $runs+=@([ordered]@{mode=$mode;pid=$worker.Id;exit_code=$rc;stdout_sha256=(Sha $out);stderr_sha256=(Sha $err);identity=$workerIdentity})
        WriteNew (Join-Path $OutputPath ($mode+'.terminal.json')) $runs[-1]
        if($rc -ne 0){throw "Fixture $mode failed with exit $rc"}
    }
}catch{$failure=$_.Exception.ToString()}
finally {
    if($worker -and -not $worker.HasExited) {
        $stopActions+=@(StopProven $workerIdentity)
        if(-not $worker.WaitForExit(5000)){$failure=[string]$failure+"`nActive worker not terminal; no unmatched process stopped"}
    }
    if($sampler) {
        if(-not(Test-Path -LiteralPath $stop)){[IO.File]::WriteAllBytes($stop,[byte[]]@())}
        if(-not $sampler.WaitForExit(10000)) {
            $failure=[string]$failure+"`nSampler did not exit after STOP"
            # A previously unbound child may now be visible. Bind only the
            # exact launcher descendant with this run's unique output argv.
            if(-not $samplerChild -and (Same (Identity $sampler.Id) $samplerIdentity)) {
                $candidate=Get-CimInstance Win32_Process | Where-Object {$_.ParentProcessId -eq $sampler.Id -and $_.ExecutablePath -eq $stdlibPy -and $_.CommandLine.Contains($resource)} | Select-Object -First 1
                if($candidate){
                    $observedChild=Identity ([int]$candidate.ProcessId)
                    if(ExpectedWorker $observedChild $stdlibPy $sampler.Id $resource){$samplerChild=$observedChild}
                }
            }
            $stopActions+=@(StopProven $samplerChild)
            $stopActions+=@(StopProven $samplerIdentity)
            [void]$sampler.WaitForExit(5000)
        }
    }
    if(Test-Path -LiteralPath $resource){$samples=@(Get-Content -LiteralPath $resource | ForEach-Object {$_|ConvertFrom-Json -DateKind String})}
    $resourcePass=$samples.Count -gt 0 -and @($samples|Where-Object {$_.error -or $null -eq $_.physical_available_bytes -or $null -eq $_.commit_available_bytes -or [int64]$_.physical_available_bytes -lt 12GB -or [int64]$_.commit_available_bytes -lt 4GB}).Count -eq 0
    $samplerTerminal=if($null -eq $sampler){$true}elseif($null -eq $samplerChild){$false}else{$sampler.HasExited -and -not(Same (Identity $samplerChild.pid) $samplerChild)}
    $workerTerminal=$null -eq $worker -or $worker.HasExited
    $status=if(-not $failure -and $runs.Count -eq 2 -and $resourcePass -and $samplerTerminal -and $workerTerminal){'pass'}else{'failed'}
    $gpuStatus=if(@($runs|Where-Object {$_.mode -eq 'synthetic' -and $_.exit_code -eq 0}).Count -eq 1){'executed_exit_zero'}elseif($syntheticStarted){'started_execution_extent_unknown'}else{'not_started'}
    $physicalMinimum=if($samples.Count){($samples|Measure-Object physical_available_bytes -Minimum).Minimum}else{$null}
    $commitMinimum=if($samples.Count){($samples|Measure-Object commit_available_bytes -Minimum).Minimum}else{$null}
    $receipt=[ordered]@{schema_version=1;status=$status;failure=$failure;source_head=$head;binary=@{path=$BinaryPath;sha256=$ExpectedBinarySha};build_receipt=@{path=$BuildReceiptPath;sha256=(Sha $BuildReceiptPath)};runs=$runs;resource=@{count=$samples.Count;all_pass=$resourcePass;physical_min_bytes=$physicalMinimum;commit_min_bytes=$commitMinimum};sampler_terminal=$samplerTerminal;sampler_launcher_identity=$samplerIdentity;sampler_child_identity=$samplerChild;stop_actions=$stopActions;worker_terminal=$workerTerminal;model_executed=$false;synthetic_process_started=$syntheticStarted;gpu_synthetic_status=$gpuStatus;performance_claim=$false}
    WriteNew (Join-Path $OutputPath 'receipt.json') $receipt
    $receipt|ConvertTo-Json -Depth 8 -Compress
}
if($status -ne 'pass'){exit 1}
