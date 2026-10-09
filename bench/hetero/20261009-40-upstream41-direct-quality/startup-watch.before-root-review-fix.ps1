param([switch]$Monitor,[int]$TimeoutSeconds=600)
# Read-only startup observer; only an explicit -Monitor enables sampling/status reads and gate-triggered owned-tree stop.
$ErrorActionPreference='Stop'; Set-StrictMode -Version 2.0
$repo='C:\Users\DC\Documents\ChatGPT\Strata-Hetero'
$runId='20261009-40-upstream41-direct-quality'; $run=Join-Path $repo ('bench\hetero\'+$runId)
$serverScript='E:\Strata-Hetero-data\source\hetero-0-1-41\serve\server.py'
$processPath=Join-Path $run 'process.json';$samplerPath=Join-Path $run 'sampler-process.json';$configPath=Join-Path $run 'config.json';$identityPath=Join-Path $run 'identity.json';$provPath=Join-Path $run 'provenance.json'
$samplePath=Join-Path $run 'resource\samples.jsonl';$eventsPath=Join-Path $run 'resource\startup-watch.jsonl';$resultPath=Join-Path $run 'resource\startup-watch-result.json'
function Sha([string]$p){(Get-FileHash -LiteralPath $p -Algorithm SHA256).Hash.ToLowerInvariant()}
function WriteExclusiveJson([string]$p,$v){$b=[Text.UTF8Encoding]::new($false).GetBytes(($v|ConvertTo-Json -Depth 12)+"`n");$f=[IO.File]::Open($p,[IO.FileMode]::CreateNew,[IO.FileAccess]::Write,[IO.FileShare]::Read);try{$f.Write($b,0,$b.Length);$f.Flush($true)}finally{$f.Dispose()}}
function Exact($all,[int]$targetPid,[string]$exe,[string]$create,[int]$parent,[string]$contains){$x=@($all|Where-Object{$_.ProcessId -eq $targetPid});if($x.Count -ne 1){return $null};$p=$x[0];if($p.ExecutablePath -ne $exe -or $p.CreationDate.ToUniversalTime().ToString('o') -ne $create -or ($parent -ge 0 -and $p.ParentProcessId -ne $parent) -or ($contains -and -not $p.CommandLine.Contains($contains))){return $null};return $p}
function Get-LatestSample {if(-not(Test-Path -LiteralPath $samplePath)){return $null};$line=Get-Content -LiteralPath $samplePath -Tail 1;if(-not $line){return $null};try{return ($line|ConvertFrom-Json -ErrorAction Stop)}catch{return $null}}
function Stop-ExactTree($receipt,$server,$engine,$all){
  # Child-first; re-query and verify PID, image, parent and full CreateTime immediately before each Stop-Process.
  $items=@();if($engine){$items+=,[pscustomobject]@{pid=[int]$engine.ProcessId;exe=$engine.ExecutablePath;created=$engine.CreationDate.ToUniversalTime().ToString('o');parent=[int]$server.ProcessId;needle=' --serve '}}
  if($server){$items+=,[pscustomobject]@{pid=[int]$server.ProcessId;exe='C:\Python314\python.exe';created=$server.CreationDate.ToUniversalTime().ToString('o');parent=[int]$receipt.launcher_pid;needle='serve\server.py'}}
  $launcher=Exact $all ([int]$receipt.launcher_pid) $receipt.launcher_exe $receipt.launcher_create_utc -1 'serve\server.py'
  if($launcher){$items+=,[pscustomobject]@{pid=[int]$launcher.ProcessId;exe=$launcher.ExecutablePath;created=$launcher.CreationDate.ToUniversalTime().ToString('o');parent=-1;needle='serve\server.py'}}
  foreach($i in $items){$fresh=@(Get-CimInstance Win32_Process);$x=Exact $fresh $i.pid $i.exe $i.created $i.parent $i.needle;if(-not $x){throw "owned-tree revalidation failed for PID $($i.pid); refusing further termination"};Stop-Process -Id $i.pid -ErrorAction Stop;Start-Sleep -Milliseconds 150}
  return @($items|ForEach-Object{@{pid=$_.pid;exe=$_.exe;create_utc=$_.created;parent_pid=$_.parent;terminated=$true}})
}
foreach($p in @($configPath,$identityPath,$provPath)){if(-not(Test-Path -LiteralPath $p -PathType Leaf)){throw "missing prepared config/identity: $p"}}
$config=Get-Content -LiteralPath $configPath -Raw|ConvertFrom-Json;$identity=Get-Content -LiteralPath $identityPath -Raw|ConvertFrom-Json;$prov=Get-Content -LiteralPath $provPath -Raw|ConvertFrom-Json
if($identity.engine.sha256 -ne (Sha $config.exe) -or $prov.engine_sha256 -ne $identity.engine.sha256 -or $identity.engine.version -ne '0.1.41'){throw 'prepared binary/config identity mismatch'}
if(-not $Monitor){[ordered]@{status='startup_watch_prepared';monitor_started=$false;run_id=$runId;physical_floor_gib=12;commit_floor_gib=4;sampling_interval_s=1;exact_process_tree_check=$true;on_gate_failure='stop only exact revalidated owned model tree; preserve read-only sampler'}|ConvertTo-Json -Compress;exit 0}
foreach($p in @($processPath,$samplerPath,$samplePath)){if(-not(Test-Path -LiteralPath $p -PathType Leaf)){throw "missing live-launch receipt/sample: $p"}}
$process=Get-Content -LiteralPath $processPath -Raw|ConvertFrom-Json;$sampler=Get-Content -LiteralPath $samplerPath -Raw|ConvertFrom-Json
if($process.run_id -ne $runId -or $sampler.run_id -ne $runId -or $process.config_sha256 -ne (Sha $configPath)){throw 'launch/binary/config receipt binding mismatch'}
if($TimeoutSeconds -lt 30 -or $TimeoutSeconds -gt 1800){throw 'TimeoutSeconds must be 30..1800'}
if(Test-Path -LiteralPath $eventsPath -or Test-Path -LiteralPath $resultPath){throw 'startup watcher evidence exists; refusing overwrite'}
$log=[IO.File]::Open($eventsPath,[IO.FileMode]::CreateNew,[IO.FileAccess]::Write,[IO.FileShare]::Read)
function Event($x){$line=(ConvertTo-Json -InputObject ([ordered]@{utc=[DateTime]::UtcNow.ToString('o');data=$x}) -Compress)+"`n";$b=[Text.UTF8Encoding]::new($false).GetBytes($line);$log.Write($b,0,$b.Length);$log.Flush($true)}
$started=[DateTimeOffset]::UtcNow;$final=$null
try{
  Event @{phase='attached';launcher_pid=$process.launcher_pid;server_child_pid=$process.actual_server_child_pid;sampler_launcher_pid=$sampler.sampler_launcher_pid;sampler_child_pid=$sampler.actual_child_pid}
  while(([DateTimeOffset]::UtcNow-$started).TotalSeconds -lt $TimeoutSeconds){
    $all=@(Get-CimInstance Win32_Process);$launcher=Exact $all ([int]$process.launcher_pid) $process.launcher_exe $process.launcher_create_utc -1 'serve\server.py';$server=Exact $all ([int]$process.actual_server_child_pid) 'C:\Python314\python.exe' $process.actual_server_child_create_utc ([int]$process.launcher_pid) 'serve\server.py';$samplerProc=Exact $all ([int]$sampler.actual_child_pid) 'C:\Python314\python.exe' $sampler.actual_child_create_utc ([int]$sampler.sampler_launcher_pid) 'hetero_resources.py'
    if(-not $launcher -or -not $server -or -not $samplerProc){throw 'exact launcher/server/sampler child identity changed or exited'}
    $engine=@($all|Where-Object{$_.ParentProcessId -eq $server.ProcessId -and $_.ExecutablePath -eq $identity.engine.path})|Select-Object -First 1
    if($engine){$expected=$identity.engine.path+' --serve '+($config.args -join ' ');if($engine.CommandLine -ne $expected){throw 'engine child argv does not equal frozen config'};if(-not $engine.CreationDate){throw 'engine child has no CreateTime'}; $engCreate=$engine.CreationDate.ToUniversalTime().ToString('o')}
    $sample=Get-LatestSample;if(-not $sample -or $sample.record_type -ne 'resource_sample' -or $sample.owner_pid -ne $process.launcher_pid){throw 'resource sampler missing/stale terminal/owner-mismatched sample'}
    $age=([DateTimeOffset]::UtcNow-[DateTimeOffset]::Parse($sample.observed_at_utc)).TotalSeconds
    $ramOk=($sample.ram_gate.status -eq 'pass' -and [double]$sample.ram_gate.available_gib -ge 12);$commitOk=($sample.commit_gate.status -eq 'pass' -and [double]$sample.commit_gate.available_gib -ge 4)
    if($age -lt -2 -or $age -gt 5 -or @($sample.alerts).Count -gt 0 -or @($sample.errors).Count -gt 0 -or -not $ramOk -or -not $commitOk){
      Event @{phase='gate_failure';sample_age_seconds=$age;ram=$sample.ram_gate;commit=$sample.commit_gate;alerts=$sample.alerts;errors=$sample.errors;engine_pid=if($engine){$engine.ProcessId}else{$null}}
      $stopped=Stop-ExactTree $process $server $engine $all
      $final=[ordered]@{schema_version=1;run_id=$runId;status='stopped_owned_tree_on_memory_gate_failure';utc=[DateTime]::UtcNow.ToString('o');sample=$sample;owned_processes_stopped=$stopped;sampler_termination_requested=$false}
      break
    }
    $listener=@(Get-NetTCPConnection -LocalPort ([int]$config.port) -State Listen -ErrorAction SilentlyContinue|Where-Object{$_.LocalAddress -eq '127.0.0.1' -and $_.OwningProcess -eq $server.ProcessId})
    if($listener.Count -eq 1){try{$status=Invoke-RestMethod -Uri ('http://127.0.0.1:'+ $config.port +'/v1/status') -Method Get -TimeoutSec 3}catch{$status=$null};if($status -and $status.loaded -eq $true -and $status.engine -eq '0.1.41' -and $status.activity.requests -eq 0 -and $status.activity.in_flight -eq 0 -and $engine){$final=[ordered]@{schema_version=1;run_id=$runId;status='startup_ready';utc=[DateTime]::UtcNow.ToString('o');sample=$sample;launcher_pid=$launcher.ProcessId;server_child_pid=$server.ProcessId;engine_pid=$engine.ProcessId;engine_exe=$engine.ExecutablePath;engine_create_utc=$engCreate;sampler_child_pid=$samplerProc.ProcessId;sampler_child_exe=$samplerProc.ExecutablePath;sampler_child_create_utc=$samplerProc.CreationDate.ToUniversalTime().ToString('o');sampler_stop_requested=$false};break}}
    Event @{phase='starting';sample_age_seconds=$age;available_ram_gib=$sample.ram_gate.available_gib;available_commit_gib=$sample.commit_gate.available_gib;launcher_pid=$launcher.ProcessId;server_child_pid=$server.ProcessId;engine_pid=if($engine){$engine.ProcessId}else{$null};sampler_child_pid=$samplerProc.ProcessId}
    Start-Sleep -Seconds 1
  }
  if(-not $final){$final=[ordered]@{schema_version=1;run_id=$runId;status='timeout_no_automatic_stop';utc=[DateTime]::UtcNow.ToString('o');timeout_seconds=$TimeoutSeconds;sampler_stop_requested=$false}}
  WriteExclusiveJson $resultPath $final
  if($final.status -eq 'startup_ready'){ $final|ConvertTo-Json -Depth 10 -Compress;exit 0 }
  $final|ConvertTo-Json -Depth 10 -Compress;exit 2
} catch {
  $err=[ordered]@{schema_version=1;run_id=$runId;status='watch_failed';utc=[DateTime]::UtcNow.ToString('o');error_type=$_.Exception.GetType().FullName;error=$_.Exception.Message;sampler_stop_requested=$false}
  try{WriteExclusiveJson $resultPath $err}catch{};try{Event @{phase='failed';error=$_.Exception.Message}}catch{};throw
} finally {$log.Dispose()}
