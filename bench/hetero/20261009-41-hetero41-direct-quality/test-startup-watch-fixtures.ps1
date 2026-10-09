param()
$ErrorActionPreference='Stop';Set-StrictMode -Version 2.0
$watcher=Join-Path $PSScriptRoot 'startup-watch.ps1'
$tokens=$null;$parseErrors=$null;$ast=[System.Management.Automation.Language.Parser]::ParseFile($watcher,[ref]$tokens,[ref]$parseErrors)
if($parseErrors.Count){throw 'watcher parser error'}
$needed=@('Exact','Stop-ExactTree','Assert-NewEvidencePaths','Get-LatestSample')
$funcs=@($ast.FindAll({param($n) $n -is [System.Management.Automation.Language.FunctionDefinitionAst] -and $needed -contains $n.Name},$true))
if($funcs.Count -ne $needed.Count){throw "expected $($needed.Count) actual watcher functions, found $($funcs.Count)"}
. ([scriptblock]::Create(($funcs|ForEach-Object{$_.Extent.Text}) -join "`n"))
$script:FixtureProcesses=@();$script:StopIds=@();$script:NaturalExit=$false;$script:ReuseServerPid=$false
function Get-CimInstance { param([string]$ClassName,[string]$Filter,[string]$ErrorAction); return @($script:FixtureProcesses) }
function Stop-Process { param([int]$Id,[string]$ErrorAction);$script:StopIds+=,$Id;if($script:NaturalExit -and $Id -eq 303){$script:FixtureProcesses=@()}elseif($script:ReuseServerPid -and $Id -eq 303){$script:FixtureProcesses=@($script:FixtureProcesses|Where-Object{$_.ProcessId -ne 303});$server=New-Fake 202 101 'C:\Python314\python.exe' '2026-10-09T10:00:02.0000000Z' 'serve\server.py C:\run\config.json' 'python.exe';$script:FixtureProcesses+=,$server}else{$script:FixtureProcesses=@($script:FixtureProcesses|Where-Object{$_.ProcessId -ne $Id})} }
function Start-Sleep { param([int]$Milliseconds) }
function New-Fake([int]$Id,[int]$Parent,[string]$Exe,[string]$Created,[string]$Command,[string]$Name){[pscustomobject]@{ProcessId=$Id;ParentProcessId=$Parent;ExecutablePath=$Exe;CreationDate=[DateTime]::Parse($Created,[Globalization.CultureInfo]::InvariantCulture,[Globalization.DateTimeStyles]::RoundtripKind);CommandLine=$Command;Name=$Name}}
# Receipt and sampler ISO strings must survive ConvertFrom-Json without DateTime reformatting.
$stamp='2026-10-09T10:00:01.1234560Z';$parsed=('{"created":"'+$stamp+'"}'|ConvertFrom-Json -DateKind String)
if($parsed.created -isnot [string] -or $parsed.created -cne $stamp){throw 'full-precision UTC receipt string was coerced'}
$sampleDir=Join-Path ([IO.Path]::GetTempPath()) ('strata-startup-fixture-'+[guid]::NewGuid().ToString('N'));[IO.Directory]::CreateDirectory($sampleDir)|Out-Null
try{$script:samplePath=Join-Path $sampleDir 'sample.jsonl';$obs='2026-10-09T10:00:01.7654321Z';[IO.File]::WriteAllText($samplePath,(ConvertTo-Json -Compress @{record_type='resource_sample';observed_at_utc=$obs})+"`n");$got=Get-LatestSample;if($got.observed_at_utc -isnot [string] -or $got.observed_at_utc -cne $obs){throw 'sampler timestamp lost exact UTC text'}}finally{[IO.File]::Delete($samplePath);[IO.Directory]::Delete($sampleDir)}
$launcher=New-Fake 101 1 'E:\venv\python.exe' $stamp 'E:\venv\python.exe -B E:\serve\server.py --config C:\run\config.json' 'python.exe'
$server=New-Fake 202 101 'C:\Python314\python.exe' '2026-10-09T10:00:01.2345670Z' 'C:\Python314\python.exe E:\serve\server.py --config C:\run\config.json' 'python.exe'
$engine=New-Fake 303 202 'E:\build\strata.exe' '2026-10-09T10:00:01.3456780Z' 'E:\build\strata.exe --serve --resident-budget-gib 20' 'strata.exe'
$receipt=[pscustomobject]@{launcher_pid=101;launcher_exe=$launcher.ExecutablePath;launcher_create_utc=$launcher.CreationDate.ToUniversalTime().ToString('o')}
# Exact helper matches complete fractional CreateTime, executable, parent and command marker.
$all=@($launcher,$server,$engine);if(-not(Exact $all 202 $server.ExecutablePath $server.CreationDate.ToUniversalTime().ToString('o') 101 'serve\server.py')){throw 'Exact rejected valid fractional-UTC owner'}
# Child-first engine stop may naturally tear down the Python server and launcher; this is success, not PID reuse.
$script:FixtureProcesses=@($all);$script:StopIds=@();$script:NaturalExit=$true;$script:ReuseServerPid=$false
$natural=Stop-ExactTree $receipt $server $engine $all
if(($script:StopIds -join ',') -ne '303' -or $natural.Count -ne 3 -or $natural[0].result -ne 'stop_requested_after_exact_revalidation' -or $natural[1].result -ne 'already_exited_naturally_after_child_stop' -or $natural[2].result -ne 'already_exited_naturally_after_child_stop'){throw 'natural child-first exit handling failed'}
# Reused server PID with different CreateTime must abort before stopping it or the launcher.
$script:FixtureProcesses=@($all);$script:StopIds=@();$script:NaturalExit=$false;$script:ReuseServerPid=$true;$reused=$false
try{$null=Stop-ExactTree $receipt $server $engine $all}catch{if($_.Exception.Message -like '*owned-tree revalidation failed for PID 202*'){$reused=$true}else{throw}}
if(-not $reused -or ($script:StopIds -join ',') -ne '303'){throw 'PID reuse was not rejected safely'}
# Evidence path preflight must reject a preexisting file and leave it intact.
$sentinel=Join-Path ([IO.Path]::GetTempPath()) ('strata-startup-existing-'+[guid]::NewGuid().ToString('N'));[IO.File]::WriteAllText($sentinel,'keep')
try{$blocked=$false;try{Assert-NewEvidencePaths $sentinel ($sentinel+'.absent')}catch{$blocked=$_.Exception.Message -like '*exists; refusing overwrite*'};if(-not $blocked -or [IO.File]::ReadAllText($sentinel) -cne 'keep'){throw 'existing watcher evidence was not safely rejected'}}finally{[IO.File]::Delete($sentinel)}
$result=[ordered]@{status='passed';watcher_functions_extracted=@('Exact','Stop-ExactTree','Assert-NewEvidencePaths','Get-LatestSample');cases=@('fractional UTC receipt string preserved','fractional UTC sample string preserved','exact process identity match','child-first natural parent exit accepted','reused PID with changed CreateTime rejected before parent stop','existing evidence rejected without overwrite');real_processes_stopped=0;hardware_or_model_calls=0;utc=[DateTime]::UtcNow.ToString('o')}
$receiptPath=Join-Path $PSScriptRoot 'startup-watch-fixtures-result.json';$bytes=[Text.UTF8Encoding]::new($false).GetBytes(($result|ConvertTo-Json -Depth 5)+"`n");$out=[IO.File]::Open($receiptPath,[IO.FileMode]::CreateNew,[IO.FileAccess]::Write,[IO.FileShare]::Read);try{$out.Write($bytes,0,$bytes.Length);$out.Flush($true)}finally{$out.Dispose()}
$result|ConvertTo-Json -Depth 5 -Compress
