$ErrorActionPreference='Stop'
$repo='C:\Users\DC\Documents\ChatGPT\Strata-Hetero'
$runDir=Join-Path $repo 'bench\hetero\20261009-14-real-ple-parity'
$admissionPath=Join-Path $runDir 'real-parity-01-admission.json'
$outputPath=Join-Path $runDir 'real-parity-01.json'
$stdoutPath=Join-Path $runDir 'real-parity-01.stdout.log'
$stderrPath=Join-Path $runDir 'real-parity-01.stderr.log'
$resourcePath=Join-Path $runDir 'real-parity-01-resource.jsonl'
$processPath=Join-Path $runDir 'real-parity-01-process.json'
$stopPath=Join-Path $runDir 'real-parity-01-sampler-stop.json'
$binary='E:\Strata-Hetero-data\build\host-ple-parity-20261009-01\strata-ple-table-parity.exe'
$model='F:\Strata-data\models\unsloth-UD-Q4_K_XL\Qwen3.8-Flash-Next-UD-Q4_K_XL-00002-of-00004.gguf'
$integrityPath=Join-Path $runDir '..\20261008-00-admission\model-integrity-02.json'
$freshIdentityPath=Join-Path $runDir '..\20261009-03-upstream04-quality\fresh-identities-before-request.json'
$buildReceiptPath=Join-Path $runDir 'build-evidence\build-receipt.json'
$expectedSha='3f342f1c1580473f1ee94ddd5b28206e8c07a70fa1a366f59d1d6c922919a6c9'
$expectedExeSha='5e830b82fabcd877e311adc1b0f3560f2256d5ee15c2b2d1a4371d5390119245'
$expectedBytes=[uint64]49859583136
$tableRows=[uint64]320001536
$rowBytes=[uint64]90
$tableBytes=$tableRows*$rowBytes
$reserveBytes=[uint64](12GB)
$physicalFloor=[math]::Max([double](12GB),[double]($tableBytes+$reserveBytes))
$commitFloor=[double](4GB)
$timeoutSeconds=1280
$created=[DateTime]::UtcNow.ToString('o')
$gates=[ordered]@{}
$blocked=$false
$blockReasons=[System.Collections.Generic.List[string]]::new()
function Write-NewJson([string]$path,$object){$json=$object|ConvertTo-Json -Depth 12; $stream=[IO.File]::Open($path,[IO.FileMode]::CreateNew,[IO.FileAccess]::Write,[IO.FileShare]::Read); try{$bytes=[Text.UTF8Encoding]::new($false).GetBytes($json+[Environment]::NewLine);$stream.Write($bytes,0,$bytes.Length);$stream.Flush($true)}finally{$stream.Dispose()}}
Add-Type -TypeDefinition @"
using System;
using System.Runtime.InteropServices;
using Microsoft.Win32.SafeHandles;
public static class RealPleGate14 {
 [StructLayout(LayoutKind.Sequential)] public class MEMORYSTATUSEX {
  public uint dwLength; public uint dwMemoryLoad; public ulong ullTotalPhys; public ulong ullAvailPhys;
  public ulong ullTotalPageFile; public ulong ullAvailPageFile; public ulong ullTotalVirtual; public ulong ullAvailVirtual; public ulong ullAvailExtendedVirtual;
  public MEMORYSTATUSEX(){dwLength=(uint)Marshal.SizeOf(typeof(MEMORYSTATUSEX));}
 }
 [StructLayout(LayoutKind.Sequential)] struct BY_HANDLE_FILE_INFORMATION {
  public uint FileAttributes; public System.Runtime.InteropServices.ComTypes.FILETIME CreationTime,LastAccessTime,LastWriteTime;
  public uint VolumeSerialNumber,FileSizeHigh,FileSizeLow,NumberOfLinks,FileIndexHigh,FileIndexLow;
 }
 [StructLayout(LayoutKind.Sequential)] struct FILE_BASIC_INFO { public long CreationTime,LastAccessTime,LastWriteTime,ChangeTime; public uint FileAttributes,Reserved; }
 [DllImport("kernel32.dll", CharSet=CharSet.Unicode, SetLastError=true)] static extern SafeFileHandle CreateFileW(string name,uint access,uint share,IntPtr security,uint creation,uint flags,IntPtr template);
 [DllImport("kernel32.dll", SetLastError=true)] static extern bool GetFileInformationByHandle(SafeFileHandle h,out BY_HANDLE_FILE_INFORMATION info);
 [DllImport("kernel32.dll", SetLastError=true)] static extern bool GetFileInformationByHandleEx(SafeFileHandle h,int infoClass,out FILE_BASIC_INFO info,uint size);
 [DllImport("kernel32.dll", SetLastError=true)] public static extern bool GlobalMemoryStatusEx([In,Out] MEMORYSTATUSEX s);
 public static long[] FileIdentity(string path){using(var h=CreateFileW(path,0x80,0x7,IntPtr.Zero,3,0x80,IntPtr.Zero)){if(h.IsInvalid)throw new System.ComponentModel.Win32Exception(Marshal.GetLastWin32Error());BY_HANDLE_FILE_INFORMATION i;if(!GetFileInformationByHandle(h,out i))throw new System.ComponentModel.Win32Exception(Marshal.GetLastWin32Error());FILE_BASIC_INFO b;if(!GetFileInformationByHandleEx(h,0,out b,(uint)Marshal.SizeOf(typeof(FILE_BASIC_INFO))))throw new System.ComponentModel.Win32Exception(Marshal.GetLastWin32Error());long size=((long)i.FileSizeHigh<<32)|i.FileSizeLow;long index=((long)i.FileIndexHigh<<32)|i.FileIndexLow;return new long[]{i.VolumeSerialNumber,index,size,b.LastWriteTime,b.CreationTime,b.ChangeTime};}}
}
"@
foreach($p in @($admissionPath,$outputPath,$stdoutPath,$stderrPath,$resourcePath,$processPath,$stopPath)){if(Test-Path -LiteralPath $p){$blocked=$true;$blockReasons.Add("output collision: $p")}}
$manifest=Get-Content -Raw -LiteralPath $integrityPath|ConvertFrom-Json
$manifestRow=@($manifest.model_shards|Where-Object {$_.path -ieq $model})|Select-Object -First 1
$fresh=Get-Content -Raw -LiteralPath $freshIdentityPath|ConvertFrom-Json
$freshRow=@($fresh.shards|Where-Object {$_.path -ieq $model})|Select-Object -First 1
$gates.model_manifest=[ordered]@{manifest=$integrityPath;record_found=($null -ne $manifestRow);expected_bytes=if($manifestRow){[uint64]$manifestRow.expected_size_bytes}else{$null};expected_sha256=if($manifestRow){$manifestRow.expected_sha256}else{$null};prior_hash_receipt_matches=($null -ne $manifestRow -and $manifestRow.expected_sha256 -eq $expectedSha -and [uint64]$manifestRow.expected_size_bytes -eq $expectedBytes)}
$gates.fresh_identity_receipt=[ordered]@{receipt=$freshIdentityPath;record_found=($null -ne $freshRow);matches_prior_full_hash_file_identity=if($freshRow){$freshRow.matches_prior_full_hash_file_identity}else{$false};prior_sha256=if($freshRow){$freshRow.prior_verified_sha256}else{$null};current_stat=if($freshRow){$freshRow.current_stat}else{$null}}
if(-not $gates.model_manifest.prior_hash_receipt_matches){$blocked=$true;$blockReasons.Add('model-integrity-02 does not bind the expected F2 path, size, and SHA-256')}
if($null -eq $freshRow -or -not $freshRow.matches_prior_full_hash_file_identity -or $freshRow.prior_verified_sha256 -ne $expectedSha){$blocked=$true;$blockReasons.Add('fresh identity receipt does not match expected prior full-hash identity')}
try{$fid=[RealPleGate14]::FileIdentity($model);$identityNow=[ordered]@{volume=$fid[0];file_index=$fid[1];size_bytes=$fid[2];last_write_filetime=$fid[3];creation_filetime=$fid[4];change_filetime=$fid[5]};$freshStat=$freshRow.current_stat;$statMtimeNs=[long]$freshStat.mtime_ns;$statCtimeNs=[long]$freshStat.ctime_ns;$writeNs=([long]$fid[3]-116444736000000000L)*100L;$statCtimeCompatibilityNs=([long]$fid[4]-116444736000000000L)*100L;$changeNs=([long]$fid[5]-116444736000000000L)*100L;$identityMatch=($fid[0] -eq [long]$freshStat.device -and $fid[1] -eq [long]$freshStat.file_index -and $fid[2] -eq $expectedBytes -and $writeNs -eq $statMtimeNs -and $statCtimeCompatibilityNs -eq $statCtimeNs);$gates.current_model_identity=[ordered]@{method='CreateFileW + GetFileInformationByHandle/Ex metadata only; no payload read';identity=$identityNow;mtime_ns=$writeNs;ctime_compatibility_ns=$statCtimeCompatibilityNs;change_time_ns=$changeNs;matches_fresh_identity_receipt=$identityMatch}}catch{$blocked=$true;$blockReasons.Add("current file identity query failed: $($_.Exception.Message)")}
if($gates.current_model_identity -and -not $gates.current_model_identity.matches_fresh_identity_receipt){$blocked=$true;$blockReasons.Add('current model file ID/size/mtime/ctime differs from prior identity receipt')}
if(-not (Test-Path -LiteralPath $binary -PathType Leaf)){$blocked=$true;$blockReasons.Add('expected host binary missing')}
else{$exeItem=Get-Item -LiteralPath $binary;$exeSha=(Get-FileHash -Algorithm SHA256 -LiteralPath $binary).Hash.ToLowerInvariant();$buildReceipt=Get-Content -Raw -LiteralPath $buildReceiptPath|ConvertFrom-Json;$receiptExe=$buildReceipt.binary.sha256;$gates.binary=[ordered]@{path=$binary;bytes=$exeItem.Length;sha256=$exeSha;expected_sha256=$expectedExeSha;matches_user_confirmed_sha=($exeSha -eq $expectedExeSha);matches_build_receipt=($exeSha -eq $receiptExe)};if($exeSha -ne $expectedExeSha -or $exeSha -ne $receiptExe){$blocked=$true;$blockReasons.Add('binary SHA differs from root-confirmed build receipt')}}
$mem=[RealPleGate14+MEMORYSTATUSEX]::new();$memOk=[RealPleGate14]::GlobalMemoryStatusEx($mem);$gates.resource_gate=[ordered]@{sample_utc=[DateTime]::UtcNow.ToString('o');method='GlobalMemoryStatusEx PSAPI';success=$memOk;physical_available_bytes=if($memOk){[uint64]$mem.ullAvailPhys}else{$null};physical_required_bytes=$physicalFloor;table_payload_bytes=$tableBytes;ram_reserve_bytes=$reserveBytes;available_commit_bytes=if($memOk){[uint64]$mem.ullAvailPageFile}else{$null};commit_required_bytes=$commitFloor;physical_pass=($memOk -and [double]$mem.ullAvailPhys -ge $physicalFloor);commit_pass=($memOk -and [double]$mem.ullAvailPageFile -ge $commitFloor)};if(-not $gates.resource_gate.physical_pass){$blocked=$true;$blockReasons.Add('physical memory below 28.8GB table + 12GiB reserve gate')};if(-not $gates.resource_gate.commit_pass){$blocked=$true;$blockReasons.Add('available commit headroom below 4GiB gate')}
$competitors=@(Get-CimInstance Win32_Process|Where-Object {$_.Name -match '^strata.*\.exe$' -or $_.CommandLine -match 'serve[/\\]server\.py|strata[-_]?server|upstream04-quality'}|Select-Object ProcessId,ParentProcessId,Name,ExecutablePath,CommandLine);$gates.competitor_processes=[ordered]@{scope='Strata engine/server command lines only';matches=$competitors;none=($competitors.Count -eq 0)};if($competitors.Count -gt 0){$blocked=$true;$blockReasons.Add('a Strata runtime/server process is active')}
$admission=[ordered]@{schema_version=1;created_utc=$created;status=if($blocked){'blocked'}else{'pass'};task='single CPU-only real IQ4_NL PLE row/gather parity run';model_path=$model;binary_path=$binary;argv=@('--run','--file',$model,'--output',$outputPath);no_payload_rehash_performed=$true;block_reasons=@($blockReasons);gates=$gates;timeout_seconds=$timeoutSeconds;physical_floor_bytes=$physicalFloor;commit_floor_bytes=$commitFloor;monitor_sampling_seconds=1;gpu_executed=$false;virtual_lock_will_be_attempted_only_if_started=$true}
Write-NewJson $admissionPath $admission
if($blocked){throw "Admission blocked; see $admissionPath. No parity process launched."}
# Reserve the resource journal exclusively before launching the one owner child.
$journal=[IO.File]::Open($resourcePath,[IO.FileMode]::CreateNew,[IO.FileAccess]::Write,[IO.FileShare]::Read);$journal.Dispose()
$started=[DateTime]::UtcNow
$proc=Start-Process -FilePath $binary -ArgumentList @('--run','--file',$model,'--output',$outputPath) -PassThru -WindowStyle Hidden -RedirectStandardOutput $stdoutPath -RedirectStandardError $stderrPath
$proc.Refresh();$procStart=$proc.StartTime.ToUniversalTime();$procPath=$proc.Path
$cim=Get-CimInstance Win32_Process -Filter "ProcessId=$($proc.Id)"
$expectedCmdParts=@('--run','--file',$model,'--output',$outputPath)
$owner=[ordered]@{schema_version=1;started_utc=$started.ToString('o');pid=$proc.Id;executable=$procPath;executable_sha256=$exeSha;process_start_utc=$procStart.ToString('o');argv=$expectedCmdParts;cmdline=$cim.CommandLine;parent_pid=$cim.ParentProcessId;priority_before=$proc.PriorityClass.ToString();timeout_seconds=$timeoutSeconds;identity_check='PID + exact executable path + process start time + exact command line before any stop';stdout=$stdoutPath;stderr=$stderrPath;result_json=$outputPath}
if($procPath -ine $binary -or $null -eq $cim -or $cim.ExecutablePath -ine $binary){$owner.identity_valid=$false;Write-NewJson $processPath $owner;throw 'Started process identity did not match target; refusing further monitoring/termination.'}
try{$proc.PriorityClass='Idle'}catch{}
$owner.priority_after=(Get-Process -Id $proc.Id).PriorityClass.ToString();$owner.identity_valid=$true;$owner.command_line=(Get-CimInstance Win32_Process -Filter "ProcessId=$($proc.Id)").CommandLine
Write-NewJson $processPath $owner
$runner=Get-Process -Id $PID
$monitorStart=[DateTime]::UtcNow
$reason='child_terminal';$killed=$false;$minPhys=[uint64]::MaxValue;$minCommit=[uint64]::MaxValue;$samples=0;$lastMem=$null
while($true){
 $proc.Refresh()
 if($proc.HasExited){break}
 $cur=Get-Process -Id $proc.Id -ErrorAction SilentlyContinue
 if($null -eq $cur){break}
 $curCim=Get-CimInstance Win32_Process -Filter "ProcessId=$($proc.Id)" -ErrorAction SilentlyContinue
 $identityOk=($null -ne $curCim -and $cur.Path -ieq $binary -and $cur.StartTime.ToUniversalTime().Ticks -eq $procStart.Ticks -and $curCim.ExecutablePath -ieq $binary -and $curCim.CommandLine -eq $owner.command_line)
 $mem=[RealPleGate14+MEMORYSTATUSEX]::new();$memOk=[RealPleGate14]::GlobalMemoryStatusEx($mem)
 if(-not $memOk){$phys=[uint64]0;$commit=[uint64]0}else{$phys=[uint64]$mem.ullAvailPhys;$commit=[uint64]$mem.ullAvailPageFile}
 if($phys -lt $minPhys){$minPhys=$phys};if($commit -lt $minCommit){$minCommit=$commit};$samples++
 $event=$null
 if(-not $identityOk){$reason='owner_identity_lost';$event='owner_identity_lost'}
 elseif(-not $memOk){$reason='resource_sample_failed';$event='PSAPI GlobalMemoryStatusEx failed'}
 elseif($phys -lt $physicalFloor){$reason='physical_floor_breached';$event='available physical memory fell below floor'}
 elseif($commit -lt $commitFloor){$reason='commit_floor_breached';$event='available commit fell below floor'}
 elseif(([DateTime]::UtcNow-$started).TotalSeconds -ge $timeoutSeconds){$reason='hard_timeout_1280s';$event='1280 second hard timeout reached'}
 $lastMem=[ordered]@{utc=[DateTime]::UtcNow.ToString('o');pid=$proc.Id;identity_valid=$identityOk;psapi_ok=$memOk;available_physical_bytes=$phys;available_commit_bytes=$commit;total_physical_bytes=if($memOk){[uint64]$mem.ullTotalPhys}else{$null};total_pagefile_bytes=if($memOk){[uint64]$mem.ullTotalPageFile}else{$null};sample_index=$samples;event=$event}
 $line=($lastMem|ConvertTo-Json -Compress);[IO.File]::AppendAllText($resourcePath,$line+[Environment]::NewLine,[Text.UTF8Encoding]::new($false))
 if($event){
  if($identityOk -and $reason -ne 'owner_identity_lost') {Stop-Process -Id $proc.Id -Force; $killed=$true}
  break
 }
 Start-Sleep -Milliseconds 850
}
$proc.Refresh();$childExited=$proc.HasExited;$childExit=if($childExited){[int]$proc.ExitCode}else{$null}
$stop=[ordered]@{schema_version=1;sampler_pid=$PID;sampler_executable=(Get-Process -Id $PID).Path;sampler_started_utc=$monitorStart.ToString('o');stop_utc=[DateTime]::UtcNow.ToString('o');reason=$reason;child_pid=$proc.Id;child_exited=$childExited;child_exit_code=$childExit;child_killed_by_floor_or_timeout=$killed;min_available_physical_bytes=if($minPhys -eq [uint64]::MaxValue){$null}else{$minPhys};min_available_commit_bytes=if($minCommit -eq [uint64]::MaxValue){$null}else{$minCommit};resource_sample_count=$samples;sampler_exits_after_receipt=$true}
Write-NewJson $stopPath $stop
if(-not $childExited){throw "Owned target ended monitor without child terminal state: $reason"}
if($childExit -ne 0){throw "Owned parity process exited $childExit; preserve failure output and logs."}
