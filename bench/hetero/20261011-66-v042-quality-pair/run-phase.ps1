param(
    [ValidateSet('upstream42','hetero42')][string]$Side='upstream42',
    [ValidateSet('launch','watch','quality','cleanup')][string]$Phase='launch',
    [ValidateRange(1,3)][int]$Attempt=1,
    [switch]$Execute,
    [string]$Authorization
)
$ErrorActionPreference='Stop'
Set-StrictMode -Version 2.0
$repo='C:\Users\DC\Documents\ChatGPT\Strata-Hetero'
$root=$PSScriptRoot
$pwsh='C:\Users\DC\.cache\codex-runtimes\codex-primary-runtime\dependencies\native\powershell\pwsh.exe'
function Sha([string]$Path){
    $stream=[IO.File]::Open($Path,[IO.FileMode]::Open,[IO.FileAccess]::Read,[IO.FileShare]::ReadWrite)
    $hasher=[Security.Cryptography.SHA256]::Create()
    try{[Convert]::ToHexString($hasher.ComputeHash($stream)).ToLowerInvariant()}
    finally{$hasher.Dispose();$stream.Dispose()}
}
function WriteNew([string]$Path,$Value){
    $raw=[Text.UTF8Encoding]::new($false).GetBytes(($Value|ConvertTo-Json -Depth 15)+"`n")
    $file=[IO.File]::Open($Path,[IO.FileMode]::CreateNew,[IO.FileAccess]::Write,[IO.FileShare]::Read)
    try{$file.Write($raw,0,$raw.Length);$file.Flush($true)}finally{$file.Dispose()}
}
if((Sha $pwsh) -ne '362a356ce7f0940ec74f73a8fc2c990a2cc24a38a11c90bbd8eca947110ad139'){throw 'Pinned PowerShell changed'}
$matrixPath=Join-Path $root 'matrix.json'
$matrix=Get-Content -LiteralPath $matrixPath -Raw|ConvertFrom-Json -DateKind String
$matches=@($matrix.sides|Where-Object id -eq $Side)
if($matches.Count -ne 1){throw 'Side is not uniquely bound in matrix'}
$sideRecord=$matches[0]
$arm=Join-Path $root $sideRecord.arm_path
foreach($kind in @('config','identity','provenance')){
    $path=Join-Path $root $sideRecord.($kind+'_path')
    if((Sha $path) -ne $sideRecord.($kind+'_sha256')){throw "$kind binding changed"}
}
$scriptPath=$sideRecord.rendered_scripts.$Phase.rendered_path
if((Sha $scriptPath) -ne $sideRecord.rendered_scripts.$Phase.rendered_sha256){throw 'Rendered phase script changed'}
if(-not $Execute){
    [ordered]@{status='phase_wrapper_validated_no_start';side=$Side;phase=$Phase;script=$scriptPath;script_sha256=(Sha $scriptPath);start_performed=$false;api_called=$false}|ConvertTo-Json -Compress
    exit 0
}
if(-not $Authorization -or -not [IO.Path]::IsPathRooted($Authorization)){throw 'Absolute root authorization required'}
$auth=Get-Content -LiteralPath $Authorization -Raw|ConvertFrom-Json -DateKind String
if($auth.status -ne 'root_reviewed_authorized' -or $auth.matrix_sha256 -ne (Sha $matrixPath) -or $auth.wrapper_sha256 -ne (Sha $PSCommandPath) -or $Side -notin $auth.sides -or $auth.execution_mode -ne 'direct_powershell_controllers'){throw 'Root execution binding mismatch'}
if($Attempt -gt 1 -and ($Side+'.'+$Phase+'.'+$Attempt) -notin $auth.corrected_preflight_attempts){throw 'This corrected preflight attempt is not explicitly authorized'}
if($Phase -eq 'launch' -and $auth.PSObject.Properties.Name -contains 'adopted_active_sides' -and $Side -in $auth.adopted_active_sides){throw 'This side already has an adopted active launch; do not relaunch'}
foreach($pin in $auth.source_pins){if((Sha $pin.path) -ne $pin.sha256){throw "Runtime source changed: $($pin.path)"}}
if($Phase -eq 'launch'){
    foreach($prompt in $matrix.quality_recipe.prompt_ids){if(Test-Path -LiteralPath (Join-Path $arm ('requests\'+$prompt))){throw 'Request output leaf must be absent before launch'}}
    foreach($relative in @('process.json','sampler-process.json','admission.json','logs\engine.log','resource\samples.jsonl','resource\STOP')){if(Test-Path -LiteralPath (Join-Path $arm $relative)){throw "Lifecycle output already exists: $relative"}}
}
$phaseFolder=if($Attempt -eq 1){$Phase}else{$Phase+'-attempt-'+$Attempt.ToString('00')}
$evidence=Join-Path $root ('execution\'+$Side+'\'+$phaseFolder)
if(Test-Path -LiteralPath $evidence){throw 'Phase evidence path already exists; no implicit retry'}
[void][IO.Directory]::CreateDirectory($evidence)
$out=Join-Path $evidence 'stdout.log';$err=Join-Path $evidence 'stderr.log'
$argv=@('-NoLogo','-NoProfile','-NonInteractive','-File',$scriptPath)
switch($Phase){
    'launch'{$argv+='-Start'}
    'watch'{$argv+=@('-Monitor','-TimeoutSeconds','1800')}
    'quality'{
        $owner=Get-Content -LiteralPath (Join-Path $arm 'process.json') -Raw|ConvertFrom-Json -DateKind String
        $sampler=Get-Content -LiteralPath (Join-Path $arm 'sampler-process.json') -Raw|ConvertFrom-Json -DateKind String
        $argv+=@('-Run','-RootReadyConfirmed','-LauncherPid',[string]$owner.launcher_pid,'-SamplerPid',[string]$sampler.sampler_launcher_pid)
    }
    'cleanup'{$argv+='-Execute'}
}
$child=Start-Process -FilePath $pwsh -ArgumentList $argv -WorkingDirectory $repo -WindowStyle Hidden -RedirectStandardOutput $out -RedirectStandardError $err -PassThru
$ownerRow=Get-CimInstance Win32_Process -Filter "ProcessId=$($child.Id)"
if(-not $ownerRow -or $ownerRow.ExecutablePath -ne $pwsh){throw 'Phase controller identity missing; inspect exact process before further work'}
WriteNew (Join-Path $evidence 'controller.json') ([ordered]@{pid=$child.Id;parent_pid=$ownerRow.ParentProcessId;exe=$ownerRow.ExecutablePath;create_utc=$ownerRow.CreationDate.ToUniversalTime().ToString('o');command_line=$ownerRow.CommandLine;argv=$argv;script_sha256=(Sha $scriptPath);matrix_sha256=(Sha $matrixPath)})
$child.WaitForExit();$child.Refresh();$exitCode=$child.ExitCode
WriteNew (Join-Path $evidence 'terminal.json') ([ordered]@{side=$Side;phase=$Phase;controller_pid=$child.Id;exit_code=$exitCode;finished_utc=[DateTimeOffset]::UtcNow.ToString('o');stdout_sha256=(Sha $out);stderr_sha256=(Sha $err);log_hash_scope='snapshot after controller exit; descendant handles may remain open until model cleanup';source_script_sha256=(Sha $scriptPath)})
[ordered]@{side=$Side;phase=$Phase;exit_code=$exitCode;evidence=$evidence}|ConvertTo-Json -Compress
exit $exitCode
