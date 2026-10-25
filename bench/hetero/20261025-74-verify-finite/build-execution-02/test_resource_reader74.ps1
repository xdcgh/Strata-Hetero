$ErrorActionPreference='Stop'
Set-StrictMode -Version 2.0
$driver=Join-Path $PSScriptRoot 'run_cuda_build_verify74_v2.ps1'
$text=[IO.File]::ReadAllText($driver)
$a=$text.IndexOf('function LatestSample {');$b=$text.IndexOf('function RunPhase',$a)
Invoke-Expression $text.Substring($a,$b-$a)
$MinRam=12GB;$MinCommit=4GB
$script:ResourcePath=Join-Path $PSScriptRoot ('reader-fixture-'+[guid]::NewGuid().ToString('N')+'.jsonl')
$cases=@()
function WriteBytes([byte[]]$bytes){[IO.File]::WriteAllBytes($script:ResourcePath,$bytes)}
function Assert([bool]$pass,[string]$name){if(-not $pass){throw "FAIL $name"};$script:cases+=@($name)}
function Raw([string]$time,[int64]$ram=12GB,[int64]$commit=4GB){'{"sampled_at_utc":"'+$time+'","physical_available_bytes":'+$ram+',"commit_available_bytes":'+$commit+',"error":null}'}
$utf8=[Text.UTF8Encoding]::new($false)
$now=[DateTimeOffset]::UtcNow.ToString('o')
$valid=Raw $now
WriteBytes ([byte[]]@());Assert ($null -eq (LatestSample)) 'empty file'
WriteBytes ($utf8.GetBytes($valid));Assert ($null -eq (LatestSample)) 'partial only'
WriteBytes ($utf8.GetBytes($valid+"`n"));Assert ((LatestSample).ok) 'final LF'
WriteBytes ($utf8.GetBytes($valid+"`r`n"));Assert ((LatestSample).ok) 'CRLF'
WriteBytes ($utf8.GetBytes($valid+"`n"+'{"part'));Assert ((LatestSample).ok) 'partial JSON suffix'
WriteBytes ($utf8.GetBytes($valid+"`n")+[byte[]]@(0xE4,0xB8));Assert ((LatestSample).ok) 'split UTF8 suffix'
WriteBytes ($utf8.GetBytes($valid+"`n`n"));$thrown=$false;try{LatestSample|Out-Null}catch{$thrown=$true};Assert $thrown 'empty complete tail fail closed'
WriteBytes ($utf8.GetBytes($valid+"`nBAD`n"));$thrown=$false;try{LatestSample|Out-Null}catch{$thrown=$true};Assert $thrown 'complete bad JSON fail closed'
WriteBytes ($utf8.GetBytes($valid+"`n")+[byte[]]@(0xE4,10));$thrown=$false;try{LatestSample|Out-Null}catch{$thrown=$true};Assert $thrown 'complete bad UTF8 fail closed'
WriteBytes ($utf8.GetBytes((Raw ([DateTimeOffset]::UtcNow.AddSeconds(-10).ToString('o')))+"`n"));Assert (-not (LatestSample).ok) 'stale fails'
WriteBytes ($utf8.GetBytes((Raw $now (12GB-1) 4GB)+"`n"));Assert (-not (LatestSample).ok) 'physical threshold unchanged'
WriteBytes ($utf8.GetBytes((Raw $now 12GB (4GB-1))+"`n"));Assert (-not (LatestSample).ok) 'commit threshold unchanged'
$tokens=$null;$errors=$null;[void][Management.Automation.Language.Parser]::ParseFile($driver,[ref]$tokens,[ref]$errors);Assert ($errors.Count -eq 0) 'build driver AST'
$fixture=Join-Path $PSScriptRoot 'run_verify_fixture74.ps1';[void][Management.Automation.Language.Parser]::ParseFile($fixture,[ref]$tokens,[ref]$errors);Assert ($errors.Count -eq 0) 'graph fixture driver AST'
[ordered]@{status='pass';count=$cases.Count;cases=$cases;fixture_data_path=$script:ResourcePath;gpu_executed=$false;model_executed=$false}|ConvertTo-Json -Depth 4
