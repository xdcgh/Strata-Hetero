[CmdletBinding()]
param(
  [Parameter(Mandatory)][string]$SourcePath,
  [Parameter(Mandatory)][string]$BuildPath,
  [Parameter(Mandatory)][string]$ExpectedHead,
  [Parameter(Mandatory)][string]$RunName,
  [switch]$Execute
)
$ErrorActionPreference='Stop'
$SourcePath=[IO.Path]::GetFullPath($SourcePath);$BuildPath=[IO.Path]::GetFullPath($BuildPath)
$Cmake='E:\Strata-Hetero-data\venv\Lib\site-packages\cmake\data\bin\cmake.exe'
$Ninja='E:\Strata-Hetero-data\venv\Scripts\ninja.exe'
$Py='E:\Strata-Hetero-data\venv\Scripts\python.exe'
$Nvcc='E:\Strata-Hetero-data\toolchains\cuda-13.3.1\bin\nvcc.exe'
$Sdk='E:\Strata-Hetero-data\toolchains\cuda-13.3.1'
$Vcvars='C:\Program Files (x86)\Microsoft Visual Studio\18\BuildTools\VC\Auxiliary\Build\vcvars64.bat'
$ClExe='C:\Program Files (x86)\Microsoft Visual Studio\18\BuildTools\VC\Tools\MSVC\14.51.36231\bin\Hostx64\x64\cl.exe'
$Ggml='E:\Strata-Hetero-data\source\llama-3cf0325'
$ReferenceBuild='E:\Strata-Hetero-data\build\cuda-hetero-upstream04-01'
$GgmlPin='3cf03257f219afbe7334045ff7c6a06ac68c627d'
$MonitorRoot='C:\Users\DC\Documents\ChatGPT\Strata-Hetero'
$MonitorHelper=Join-Path $MonitorRoot 'tools\hetero_resources.py'
$MonitorHelperSha='5d0fc03363304f1fa2d7e6a8325775467b0a240122d78466078b4340395aa285'
$MonitorHelperBlob='a7ae863b48ae8b94a0e057195be2142045a4f22a'
$GitExe=(Get-Command git.exe -ErrorAction Stop).Source
$ComSpec=Join-Path $env:WINDIR 'System32\cmd.exe'
$MinRam=[int64](12GB);$MinCommit=[int64](4GB)
$script:PhaseResults=@();$script:ResourceStop=$false;$script:SamplerProcess=$null;$script:SamplerChild=$null
$script:StopPath=Join-Path $BuildPath 'resources.STOP';$script:ResourcePath=Join-Path $BuildPath 'resources.jsonl'
function Sha([string]$p){(Get-FileHash -LiteralPath $p -Algorithm SHA256).Hash.ToLowerInvariant()}
function InvokeGitLine([string]$root,[string[]]$argv){$o=@(& $GitExe -C $root @argv);if($LASTEXITCODE -ne 0){throw "git -C $root $($argv -join ' ') failed: $LASTEXITCODE"};if($o.Count -lt 1){return ''};return [string]$o[0]}
function InvokeGitLines([string]$root,[string[]]$argv){$o=@(& $GitExe -C $root @argv);if($LASTEXITCODE -ne 0){throw "git -C $root $($argv -join ' ') failed: $LASTEXITCODE"};return $o}
function GitBind([string]$root){
 $head=(InvokeGitLine $root @('rev-parse','HEAD')).Trim();$tree=(InvokeGitLine $root @('rev-parse','HEAD^{tree}')).Trim();$branch=(InvokeGitLine $root @('branch','--show-current')).Trim();$status=@(InvokeGitLines $root @('status','--porcelain=v1','--untracked-files=all'))
 [pscustomobject]@{path=$root;head=$head;tree=$tree;branch=$branch;clean=($status.Count -eq 0);status=$status}
}
function MonitorBind{$head=(InvokeGitLine $MonitorRoot @('rev-parse','HEAD')).Trim();$blob=(InvokeGitLine $MonitorRoot @('rev-parse','HEAD:tools/hetero_resources.py')).Trim();$status=@(InvokeGitLines $MonitorRoot @('status','--porcelain=v1','--','tools/hetero_resources.py'));$sha=Sha $MonitorHelper;[pscustomobject]@{path=$MonitorHelper;root=$MonitorRoot;head=$head;git_blob=$blob;sha256=$sha;clean=($status.Count -eq 0);status=$status}}
function WriteExclusive([string]$path,$value,[int]$depth=10){$b=[Text.UTF8Encoding]::new($false).GetBytes(($value|ConvertTo-Json -Depth $depth)+"`n");$f=[IO.File]::Open($path,[IO.FileMode]::CreateNew,[IO.FileAccess]::Write,[IO.FileShare]::Read);try{$f.Write($b,0,$b.Length);$f.Flush($true)}finally{$f.Dispose()}}
function FindChild([int]$ParentProcessId,[string]$needle){$end=(Get-Date).AddSeconds(10);do{$p=Get-CimInstance Win32_Process|Where-Object{$_.ParentProcessId -eq $ParentProcessId -and $_.CommandLine -and $_.CommandLine.Contains($needle)}|Select-Object -First 1;if($p){return $p};Start-Sleep -Milliseconds 100}while((Get-Date)-lt $end);return $null}
function Tree([int]$rootPid){
 $all=@(Get-CimInstance Win32_Process);$depth=@{};$depth[$rootPid]=0;$changed=$true
 while($changed){$changed=$false;foreach($p in $all){$parent=[int]$p.ParentProcessId;if($depth.ContainsKey($parent) -and -not $depth.ContainsKey([int]$p.ProcessId)){$depth[[int]$p.ProcessId]=$depth[$parent]+1;$changed=$true}}}
 @($all|Where-Object{$depth.ContainsKey([int]$_.ProcessId)}|ForEach-Object{[pscustomobject]@{pid=[int]$_.ProcessId;parent=[int]$_.ParentProcessId;depth=$depth[[int]$_.ProcessId];exe=[string]$_.ExecutablePath;created=$_.CreationDate.ToUniversalTime().ToString('o');command=[string]$_.CommandLine}})
}
function SetTreeIdle([int]$rootPid){foreach($n in (Tree $rootPid)){try{$c=Get-CimInstance Win32_Process -Filter "ProcessId = $($n.pid)" -ErrorAction Stop;if($c.ExecutablePath -eq $n.exe -and [int]$c.ParentProcessId -eq $n.parent -and $c.CreationDate.ToUniversalTime().ToString('o') -eq $n.created -and [string]$c.CommandLine -eq $n.command){$p=Get-Process -Id $n.pid -ErrorAction Stop;if($p.PriorityClass -ne 'Idle'){$p.PriorityClass='Idle'}}}catch{}}}
function StopOwnedTree($rootIdentity){
 $snap=@(Tree $rootIdentity.pid);$root=$snap|Where-Object pid -eq $rootIdentity.pid|Select-Object -First 1;$stopped=@();$diag=@()
 if(-not $root -or $root.exe -ne $rootIdentity.exe -or $root.created -ne $rootIdentity.created -or -not $root.command.Contains($rootIdentity.batch)){return [pscustomobject]@{complete=$false;stopped=@();diagnostics=@('root identity absent or mismatched; no process terminated')}}
 foreach($n in @($snap|Sort-Object depth -Descending)){$c=Get-CimInstance Win32_Process -Filter "ProcessId = $($n.pid)" -ErrorAction SilentlyContinue;if(-not $c){continue};$same=($c.ExecutablePath -eq $n.exe -and [int]$c.ParentProcessId -eq $n.parent -and $c.CreationDate.ToUniversalTime().ToString('o') -eq $n.created -and [string]$c.CommandLine -eq $n.command);if($same){try{Stop-Process -Id $n.pid -ErrorAction Stop;$stopped+=@($n.pid)}catch{$diag+=@("stop $($n.pid): $($_.Exception.Message)")}}else{$diag+=@("identity changed for PID $($n.pid); skipped")}}
 [pscustomobject]@{complete=($diag.Count -eq 0);stopped=$stopped;diagnostics=$diag}
}
function LatestSample {
 if(-not(Test-Path -LiteralPath $script:ResourcePath)){return $null};$line=Get-Content -LiteralPath $script:ResourcePath -Tail 1;if(-not $line){return $null};$d=[System.Text.Json.JsonDocument]::Parse([string]$line);$x=$line|ConvertFrom-Json;$at=[DateTimeOffset]::Parse($d.RootElement.GetProperty('sampled_at_utc').GetString());$age=([DateTimeOffset]::UtcNow-$at).TotalSeconds
 [pscustomobject]@{raw=$x;age=$age;ok=($age -ge -2 -and $age -le 5 -and $null -ne $x.physical_available_bytes -and $null -ne $x.commit_available_bytes -and [int64]$x.physical_available_bytes -ge $MinRam -and [int64]$x.commit_available_bytes -ge $MinCommit -and -not $x.error)}
}
function RunPhase([string]$name,[string[]]$commands){
 $batch=Join-Path $BuildPath ($name+'.cmd');$log=Join-Path $BuildPath ($name+'.log');$exitFile=Join-Path $BuildPath ($name+'.exit.txt');$procFile=Join-Path $BuildPath ($name+'.process.json')
 if((Test-Path $batch) -or (Test-Path $log) -or (Test-Path $exitFile) -or (Test-Path $procFile)){throw "refuse phase evidence overwrite: $name"}
 $lines=@('@echo off','setlocal',"call `"$Vcvars`" > `"$(Join-Path $BuildPath ($name+'.vcvars.log'))`" 2>&1",'if errorlevel 1 goto vcvars_failed',"set `"CUDA_PATH=$Sdk`"", "set `"CUDA_PATH_V13_3=$Sdk`"", "set `"CUDACXX=$Nvcc`"", "set `"PATH=$Sdk\bin;$Sdk\bin\x64;$Sdk\nvvm\bin;%PATH%`"")+ $commands + @("set `"RC=%ERRORLEVEL%`"", ">`"$exitFile`" echo %RC%",'exit /b %RC%',':vcvars_failed',"set `"RC=%ERRORLEVEL%`"",">`"$exitFile`" echo %RC%",'exit /b %RC%')
 [IO.File]::WriteAllLines($batch,$lines,[Text.Encoding]::ASCII)
 $before=LatestSample;if(-not $before -or -not $before.ok){throw "$name refused: no fresh 12GiB/4GiB resource sample"}
 $process=Start-Process -FilePath $ComSpec -ArgumentList @('/d','/c','call',"`"$batch`"") -WorkingDirectory $SourcePath -WindowStyle Hidden -PassThru
 try{$process.PriorityClass='Idle'}catch{}
 $cim=Get-CimInstance Win32_Process -Filter "ProcessId = $($process.Id)" -ErrorAction SilentlyContinue
 if(-not $cim -or $cim.ExecutablePath -ne $ComSpec -or -not [string]$cim.CommandLine.Contains($batch)){throw "$name owned cmd identity could not be bound to its batch"}
 $root=[ordered]@{pid=[int]$process.Id;exe=[string]$cim.ExecutablePath;created=$cim.CreationDate.ToUniversalTime().ToString('o');command=[string]$cim.CommandLine;batch=$batch}
 WriteExclusive $procFile $root
 $start=[DateTimeOffset]::UtcNow
 while($true){
   $process.Refresh();if($process.HasExited){break}
   SetTreeIdle $process.Id
   $sample=$null;try{$sample=LatestSample}catch{}
   if(-not $sample -or -not $sample.ok){
     $script:ResourceStop=$true;$stop=StopOwnedTree $root
     $failure=[ordered]@{phase=$name;reason='resource sample stale, unknown, or below 12GiB physical/4GiB commit gate';observed_utc=[DateTimeOffset]::UtcNow.ToString('o');sample=if($sample){$sample.raw}else{$null};owned_stop=$stop}
     WriteExclusive (Join-Path $BuildPath ($name+'.resource-gate-failure.json')) $failure
     $process.Refresh();if(-not $process.HasExited){try{$process.WaitForExit(15000)|Out-Null}catch{}}
     throw "$name terminated on resource gate; see preserved exact-owner stop receipt"
   }
   Start-Sleep -Seconds 1
 }
 $process.WaitForExit();$process.Refresh();$code=[int]$process.ExitCode;$elapsed=([DateTimeOffset]::UtcNow-$start).TotalSeconds
 if(-not(Test-Path $exitFile)){[IO.File]::WriteAllText($exitFile,"$code`n",[Text.UTF8Encoding]::new($false))}
 $declared=[int](Get-Content -LiteralPath $exitFile -Raw).Trim()
 $result=[ordered]@{phase=$name;exit_code=$code;batch_exit_file=$declared;duration_seconds=$elapsed;process=$root;log=$log;batch=$batch;last_resource=(LatestSample).raw}
 WriteExclusive (Join-Path $BuildPath ($name+'.result.json')) $result
 if($code -ne 0 -or $declared -ne 0){throw "$name failed with actual exit $code (batch record $declared); logs preserved"}
 return $result
}
function CacheValue([string]$cache,[string]$key){$m=[regex]::Match($cache,"(?m)^"+[regex]::Escape($key)+':[A-Z_]+=(.*)$');if($m.Success){return $m.Groups[1].Value.Trim()};return $null}
function NormPath([string]$v){if($null -eq $v){return ''};[IO.Path]::GetFullPath($v.Replace('/','\')).TrimEnd('\')}
function CMakeSetValue([string]$path,[string]$key){if(-not(Test-Path -LiteralPath $path -PathType Leaf)){throw "CMake compiler config missing: $path"};$txt=Get-Content -LiteralPath $path -Raw;$m=[regex]::Match($txt,'(?m)^\s*set\('+[regex]::Escape($key)+'\s+"([^"]*)"\s*\)');if(-not $m.Success){throw "CMake compiler config key missing: $key in $path"};return $m.Groups[1].Value}
function CMakeCompilerConfig([string]$dir){$d=Join-Path $dir 'CMakeFiles/4.4.3';$xx=Join-Path $d 'CMakeCXXCompiler.cmake';$cu=Join-Path $d 'CMakeCUDACompiler.cmake';[ordered]@{directory=$d;cxx_file=$xx;cxx_file_sha256=(Sha $xx);cxx_id=(CMakeSetValue $xx 'CMAKE_CXX_COMPILER_ID');cxx_version=(CMakeSetValue $xx 'CMAKE_CXX_COMPILER_VERSION');cxx_path=(CMakeSetValue $xx 'CMAKE_CXX_COMPILER');cuda_file=$cu;cuda_file_sha256=(Sha $cu);cuda_id=(CMakeSetValue $cu 'CMAKE_CUDA_COMPILER_ID');cuda_version=(CMakeSetValue $cu 'CMAKE_CUDA_COMPILER_VERSION');cuda_path=(CMakeSetValue $cu 'CMAKE_CUDA_COMPILER')}}
function QuoteArg([string]$x){'"'+$x.Replace('"','\"')+'"'}
if(-not $Execute){
 $srcBind=GitBind $SourcePath;$ggmlBind=GitBind $Ggml;$monitorBind=MonitorBind
 if($monitorBind.sha256 -ne $MonitorHelperSha -or $monitorBind.git_blob -ne $MonitorHelperBlob -or -not $monitorBind.clean){throw 'Validate-only canonical resource helper identity/clean check failed'}
 if($srcBind.head -ne $ExpectedHead -or -not $srcBind.clean){throw 'Validate-only source head/clean check failed'}
 if($ggmlBind.head -ne $GgmlPin -or -not $ggmlBind.clean){throw 'Validate-only ggml pin/clean check failed'}
 if(Test-Path -LiteralPath $BuildPath){throw 'Validate-only destination build directory already exists'}
 foreach($item in @($Cmake,$Ninja,$Py,$Nvcc,$Vcvars,$ClExe,$MonitorHelper,(Join-Path $Sdk 'include\cuda.h'))){if(-not(Test-Path -LiteralPath $item -PathType Leaf)){throw "Validate-only pinned tool missing: $item"}}
 $refCachePath=Join-Path $ReferenceBuild 'CMakeCache.txt';if(-not(Test-Path -LiteralPath $refCachePath)){throw 'Validate-only H4 CMakeCache reference is missing'}
 $refText=Get-Content -LiteralPath $refCachePath -Raw;$refKeys=@('CMAKE_PROJECT_VERSION','CMAKE_BUILD_TYPE','CMAKE_CUDA_ARCHITECTURES','CMAKE_CUDA_COMPILER','CMAKE_CXX_COMPILER','STRATA_ENABLE_CUDA','STRATA_ENABLE_HIP','STRATA_HIP_GFX906','STRATA_ENABLE_SYCL','STRATA_BUILD_TESTS','STRATA_NATIVE_EXPERTS','STRATA_PORTABLE','STRATA_MMQ_KQUANTS','STRATA_Q6K_EXPERTS','STRATA_PREFILL_MMQ','STRATA_ORCA_Q4KS_MMQ','STRATA_ISA_FLOOR','STRATA_GGML_DIR','CUDAToolkit_ROOT','CMAKE_MAKE_PROGRAM','GGML_CUDA');$refValues=[ordered]@{};foreach($k in $refKeys){$refValues[$k]=CacheValue $refText $k;if($null -eq $refValues[$k]){throw "Validate-only matched H4 CMakeCache lacks required key: $k"}}
 $refCompilers=CMakeCompilerConfig $ReferenceBuild
 $refOff=@('STRATA_ENABLE_HIP','STRATA_HIP_GFX906','STRATA_ENABLE_SYCL','STRATA_MMQ_KQUANTS','STRATA_Q6K_EXPERTS','STRATA_PREFILL_MMQ','STRATA_ORCA_Q4KS_MMQ');$refBadOff=@($refOff|Where-Object{$refValues[$_] -ne 'OFF'})
 if($refValues.CMAKE_PROJECT_VERSION -ne '0.1.40.4' -or $refValues.CMAKE_BUILD_TYPE -ne 'Release' -or $refValues.CMAKE_CUDA_ARCHITECTURES -ne '89' -or (NormPath $refValues.CMAKE_CUDA_COMPILER) -ne (NormPath $Nvcc) -or (NormPath $refValues.CMAKE_CXX_COMPILER) -ne (NormPath $ClExe) -or (NormPath $refValues.CMAKE_MAKE_PROGRAM) -ne (NormPath $Ninja) -or $refValues.STRATA_ENABLE_CUDA -ne 'ON' -or $refValues.STRATA_NATIVE_EXPERTS -ne 'ON' -or $refValues.STRATA_PORTABLE -ne 'ON' -or $refValues.STRATA_BUILD_TESTS -ne 'OFF' -or (NormPath $refValues.STRATA_GGML_DIR) -ne (NormPath $Ggml) -or (NormPath $refValues.CUDAToolkit_ROOT) -ne (NormPath $Sdk) -or $refValues.GGML_CUDA -ne 'OFF' -or $refBadOff.Count -gt 0 -or $refCompilers.cxx_id -ne 'MSVC' -or $refCompilers.cxx_version -ne '19.51.36260.0' -or (NormPath $refCompilers.cxx_path) -ne (NormPath $ClExe) -or $refCompilers.cuda_id -ne 'NVIDIA' -or $refCompilers.cuda_version -ne '13.3.73' -or (NormPath $refCompilers.cuda_path) -ne (NormPath $Nvcc)){throw 'Validate-only matched H4 cache/compiler-config parser verification failed'}
 $memCode="import json,sys;sys.path.insert(0,sys.argv[1]);from tools.hetero_resources import windows_memory_global;print(json.dumps(windows_memory_global()))"
 $memoryLines=@(& $Py -B -X utf8 -c $memCode $MonitorRoot);$memoryExit=$LASTEXITCODE
 if($memoryExit -ne 0){throw "Validate-only memory helper exit $memoryExit"};$memory=($memoryLines -join "`n")|ConvertFrom-Json
 if($null -eq $memory.physical_available_bytes -or $null -eq $memory.commit_available_bytes -or [int64]$memory.physical_available_bytes -lt $MinRam -or [int64]$memory.commit_available_bytes -lt $MinCommit){throw 'Validate-only fresh 12GiB/4GiB gate failed'}
 $cmv=((& $Cmake --version)|Select-Object -First 1);$niv=((& $Ninja --version)|Select-Object -First 1);$nvv=((& $Nvcc --version|Select-String 'release 13.3').Line)
 if($cmv -ne 'cmake version 4.4.3' -or $niv -ne '1.13.2.git.kitware.jobserver-pipe-1' -or $nvv -notmatch 'release 13\.3, V13\.3\.73'){throw 'Validate-only CMake/Ninja/NVCC version pin mismatch'}
 $result=[ordered]@{status='validated_not_started';run_name=$RunName;execute_switch_present=$false;source=$srcBind;ggml=$ggmlBind;resource_monitor=$monitorBind;build_directory_absent=$true;toolchain=@{cmake=$cmv;ninja=$niv;nvcc=$nvv;cmake_sha256=(Sha $Cmake);ninja_sha256=(Sha $Ninja);nvcc_sha256=(Sha $Nvcc);vcvars64_sha256=(Sha $Vcvars);cl_exe=$ClExe;cl_sha256=(Sha $ClExe);cuda_header_sha256=(Sha (Join-Path $Sdk 'include\cuda.h'));sdk_root=$Sdk};matched_h4_reference=@{build=$ReferenceBuild;cache_values=$refValues;compiler_configs=$refCompilers;checks_passed=$true};resource_gate=@{sample=$memory;minimum_ram_bytes=$MinRam;minimum_commit_bytes=$MinCommit;pass=$true};planned_target='strata';planned_flags=@('-DCMAKE_BUILD_TYPE=Release','-DSTRATA_ENABLE_CUDA=ON','-DCMAKE_CUDA_ARCHITECTURES=89','-DSTRATA_NATIVE_EXPERTS=ON','-DSTRATA_PORTABLE=ON','-DSTRATA_BUILD_TESTS=OFF','-DSTRATA_MMQ_KQUANTS=OFF','-DSTRATA_Q6K_EXPERTS=OFF','-DSTRATA_ENABLE_HIP=OFF','-DSTRATA_HIP_GFX906=OFF','-DSTRATA_ENABLE_SYCL=OFF','-DSTRATA_PREFILL_MMQ=OFF','-DSTRATA_ORCA_Q4KS_MMQ=OFF',"-DSTRATA_GGML_DIR=$Ggml","-DCMAKE_CUDA_COMPILER=$Nvcc","-DCUDAToolkit_ROOT=$Sdk")}
 $result|ConvertTo-Json -Depth 8 -Compress;exit 0
}
$failure=$null;$receiptWriteFailure=$null;$samplerStopOk=$false;$sourceBefore=$null;$ggmlBefore=$null;$sourceAfter=$null;$ggmlAfter=$null;$configureResult=$null;$buildResult=$null;$cacheValues=$null;$binary=$null;$toolBefore=$null;$toolAfter=$null;$sampleStats=$null;$controllerId=$PID;$controllerCim=Get-CimInstance Win32_Process -Filter "ProcessId = $controllerId" -ErrorAction SilentlyContinue
try{
 if(Test-Path -LiteralPath $BuildPath){throw "Refusing existing build directory: $BuildPath"}
 New-Item -ItemType Directory -Path $BuildPath|Out-Null
 foreach($p in @($Cmake,$Ninja,$Py,$Nvcc,$Vcvars,(Join-Path $Sdk 'include\cuda.h'),(Join-Path $Ggml 'CMakeLists.txt'))){if(-not(Test-Path -LiteralPath $p -PathType Leaf)){throw "Pinned input missing: $p"}}
 $sourceBefore=GitBind $SourcePath;if($sourceBefore.head -ne $ExpectedHead -or -not $sourceBefore.clean){throw "Source identity/clean gate failed: $($sourceBefore|ConvertTo-Json -Compress)"}
 $monitorBefore=MonitorBind;if($monitorBefore.sha256 -ne $MonitorHelperSha -or $monitorBefore.git_blob -ne $MonitorHelperBlob -or -not $monitorBefore.clean){throw 'Canonical resource helper identity changed before build'}
 $ggmlBefore=GitBind $Ggml;if($ggmlBefore.head -ne $GgmlPin -or -not $ggmlBefore.clean){throw "Pinned ggml identity/clean gate failed: $($ggmlBefore|ConvertTo-Json -Compress)"}
 $keyFiles=@('CMakeLists.txt','src/program/generate.cpp','src/prefill/prefill.cpp','src/core/expert_source.cpp','serve/server.py')
 $sourceFiles=@($keyFiles|ForEach-Object{$p=Join-Path $SourcePath $_;[ordered]@{path=$_;sha256=(Sha $p);bytes=(Get-Item $p).Length}})
 $ggmlFiles=@('CMakeLists.txt','ggml/include/ggml.h','ggml/include/ggml-cpu.h','ggml/src/ggml.c','ggml/src/ggml-cpu/ggml-cpu.c')|ForEach-Object{$p=Join-Path $Ggml $_;[ordered]@{path=$_;sha256=(Sha $p);bytes=(Get-Item $p).Length}}
 $toolBefore=[ordered]@{cmake=@{path=$Cmake;sha256=(Sha $Cmake);version=((& $Cmake --version)|Select-Object -First 1)};ninja=@{path=$Ninja;sha256=(Sha $Ninja);version=((& $Ninja --version)|Select-Object -First 1)};nvcc=@{path=$Nvcc;sha256=(Sha $Nvcc);version=((& $Nvcc --version|Select-String 'release 13.3').Line)};cuda_header_sha256=(Sha (Join-Path $Sdk 'include\cuda.h'));sdk_root=$Sdk;vcvars64=@{path=$Vcvars;sha256=(Sha $Vcvars)}}
 if($toolBefore.cmake.version -ne 'cmake version 4.4.3' -or $toolBefore.ninja.version -ne '1.13.2.git.kitware.jobserver-pipe-1' -or $toolBefore.nvcc.version -notmatch 'release 13\.3, V13\.3\.73'){throw 'Pinned CMake/Ninja/NVCC version gate failed'}
 $sampler=Join-Path $BuildPath 'resource_sampler.py';$samplerLines=@("import datetime,json,pathlib,sys,time","source=pathlib.Path(sys.argv[1]);out=pathlib.Path(sys.argv[2]);stop=pathlib.Path(sys.argv[3]);sys.path.insert(0,str(source))","from tools.hetero_resources import windows_memory_global","with out.open('x',encoding='utf-8',buffering=1) as f:"," while True:","  at=datetime.datetime.now(datetime.timezone.utc).isoformat(timespec='milliseconds').replace('+00:00','Z')","  try:","   m=windows_memory_global();row={'sampled_at_utc':at,'physical_total_bytes':m.get('physical_total_bytes'),'physical_available_bytes':m.get('physical_available_bytes'),'commit_limit_bytes':m.get('commit_limit_bytes'),'commit_available_bytes':m.get('commit_available_bytes'),'source':m.get('source'),'error':None}","  except Exception as e: row={'sampled_at_utc':at,'physical_available_bytes':None,'commit_available_bytes':None,'error':type(e).__name__+': '+str(e)}","  f.write(json.dumps(row,separators=(',',':'))+'\n')","  if stop.exists(): break","  time.sleep(1)")
 [IO.File]::WriteAllLines($sampler,$samplerLines,[Text.UTF8Encoding]::new($false))
 $stop=Join-Path $BuildPath 'resources.STOP';if(Test-Path $stop){throw 'Sampler stop sentinel unexpectedly exists'}
 $samOut=Join-Path $BuildPath 'resource-sampler.stdout.log';$samErr=Join-Path $BuildPath 'resource-sampler.stderr.log';$samArgs=@('-u','-B','-X','utf8',$sampler,$MonitorRoot,$script:ResourcePath,$stop)
 $script:SamplerProcess=Start-Process -FilePath $Py -ArgumentList $samArgs -WorkingDirectory $SourcePath -WindowStyle Hidden -PassThru -RedirectStandardOutput $samOut -RedirectStandardError $samErr
 try{$script:SamplerProcess.PriorityClass='Idle'}catch{}
 $sc=Get-CimInstance Win32_Process -Filter "ProcessId = $($script:SamplerProcess.Id)" -ErrorAction SilentlyContinue
 if(-not $sc -or $sc.ExecutablePath -ne $Py){throw 'Resource sampler venv launcher identity unavailable'}
 $script:SamplerChild=FindChild $script:SamplerProcess.Id 'resource_sampler.py'
 if(-not $script:SamplerChild -or $script:SamplerChild.ExecutablePath -ne 'C:\Python314\python.exe'){throw 'Resource sampler actual Python child identity unavailable'}
 try{(Get-Process -Id $script:SamplerChild.ProcessId -ErrorAction Stop).PriorityClass='Idle'}catch{}
 $samplerIdentity=[ordered]@{launcher_pid=$script:SamplerProcess.Id;launcher_exe=$sc.ExecutablePath;launcher_create_utc=$sc.CreationDate.ToUniversalTime().ToString('o');child_pid=$script:SamplerChild.ProcessId;child_parent_pid=$script:SamplerChild.ParentProcessId;child_exe=$script:SamplerChild.ExecutablePath;child_create_utc=$script:SamplerChild.CreationDate.ToUniversalTime().ToString('o');argv=$samArgs;output=$script:ResourcePath}
 WriteExclusive (Join-Path $BuildPath 'resource-sampler-process.json') $samplerIdentity
 $until=(Get-Date).AddSeconds(10);$initial=$null;do{$initial=LatestSample;if($initial -and $initial.ok){break};Start-Sleep -Milliseconds 200}while((Get-Date)-lt $until)
 if(-not $initial -or -not $initial.ok){throw 'Fresh initial physical RAM/commit gate failed or sampler telemetry unknown'}
 WriteExclusive (Join-Path $BuildPath 'resource-preflight.json') ([ordered]@{sample=$initial.raw;age_seconds=$initial.age;ram_floor_bytes=$MinRam;commit_headroom_floor_bytes=$MinCommit;sample_interval_seconds=1;pass=$true})
 $configureArgs=@('-S',$SourcePath,'-B',$BuildPath,'-G','Ninja',"-DCMAKE_MAKE_PROGRAM=$Ninja",'-DCMAKE_BUILD_TYPE=Release','-DSTRATA_ENABLE_CUDA=ON','-DCMAKE_CUDA_ARCHITECTURES=89',"-DCMAKE_CUDA_COMPILER=$Nvcc", "-DCUDAToolkit_ROOT=$Sdk",'-DSTRATA_BUILD_TESTS=OFF','-DSTRATA_NATIVE_EXPERTS=ON','-DSTRATA_PORTABLE=ON',"-DSTRATA_GGML_DIR=$Ggml",'-DSTRATA_ENABLE_HIP=OFF','-DSTRATA_HIP_GFX906=OFF','-DSTRATA_ENABLE_SYCL=OFF','-DSTRATA_MMQ_KQUANTS=OFF','-DSTRATA_Q6K_EXPERTS=OFF','-DSTRATA_PREFILL_MMQ=OFF','-DSTRATA_ORCA_Q4KS_MMQ=OFF','-DSTRATA_ISA_FLOOR=')
 $configureCmd='"'+$Cmake+'" '+(($configureArgs|ForEach-Object{QuoteArg $_}) -join ' ')
 $configureLines=@(
  ('where cl > "'+(Join-Path $BuildPath 'where-cl.log')+'" 2>&1')
  ('where nvcc > "'+(Join-Path $BuildPath 'where-nvcc.log')+'" 2>&1')
  ('cl /Bv > "'+(Join-Path $BuildPath 'cl-version.log')+'" 2>&1')
  ('"'+$Nvcc+'" --version > "'+(Join-Path $BuildPath 'nvcc-version.log')+'" 2>&1')
  ('"'+$Cmake+'" --version > "'+(Join-Path $BuildPath 'cmake-version.log')+'" 2>&1')
  ('"'+$Ninja+'" --version > "'+(Join-Path $BuildPath 'ninja-version.log')+'" 2>&1')
  ($configureCmd+' > "'+(Join-Path $BuildPath 'configure.log')+'" 2>&1')
 )
 if($configureLines.Count -ne 7 -or $configureLines[0].Contains('where nvcc') -or @($configureLines|Where-Object{$_ -match '[\r\n]'}).Count){throw 'Configure command line serialization is invalid'}
 $configureResult=RunPhase 'configure' $configureLines
 if($configureResult.exit_code -ne 0){throw "CMake configure failed: $($configureResult.exit_code)"}
 $cachePath=Join-Path $BuildPath 'CMakeCache.txt';$ninjaFile=Join-Path $BuildPath 'build.ninja';if(-not(Test-Path $cachePath) -or -not(Test-Path $ninjaFile)){throw 'Configure exit 0 without CMakeCache/build.ninja'}
 $cacheText=Get-Content -LiteralPath $cachePath -Raw;$keys=@('CMAKE_PROJECT_VERSION','CMAKE_BUILD_TYPE','CMAKE_CUDA_ARCHITECTURES','CMAKE_CUDA_COMPILER','CMAKE_CXX_COMPILER','STRATA_ENABLE_CUDA','STRATA_ENABLE_HIP','STRATA_HIP_GFX906','STRATA_ENABLE_SYCL','STRATA_BUILD_TESTS','STRATA_NATIVE_EXPERTS','STRATA_PORTABLE','STRATA_MMQ_KQUANTS','STRATA_Q6K_EXPERTS','STRATA_PREFILL_MMQ','STRATA_ORCA_Q4KS_MMQ','STRATA_ISA_FLOOR','STRATA_GGML_DIR','CUDAToolkit_ROOT','CMAKE_MAKE_PROGRAM','GGML_CUDA');$cacheValues=[ordered]@{};foreach($k in $keys){$cacheValues[$k]=CacheValue $cacheText $k}
 $compilerConfig=CMakeCompilerConfig $BuildPath
 $whereCl=(Get-Content -LiteralPath (Join-Path $BuildPath 'where-cl.log')|Select-Object -First 1).Trim()
 $expectedOff=@('STRATA_ENABLE_HIP','STRATA_HIP_GFX906','STRATA_ENABLE_SYCL','STRATA_MMQ_KQUANTS','STRATA_Q6K_EXPERTS','STRATA_PREFILL_MMQ','STRATA_ORCA_Q4KS_MMQ')
 $badOff=@($expectedOff|Where-Object{$cacheValues[$_] -ne 'OFF'})
 if($cacheValues.CMAKE_PROJECT_VERSION -ne '0.1.41' -or $cacheValues.CMAKE_BUILD_TYPE -ne 'Release' -or $cacheValues.CMAKE_CUDA_ARCHITECTURES -ne '89' -or (NormPath $cacheValues.CMAKE_CUDA_COMPILER) -ne (NormPath $Nvcc) -or (NormPath $cacheValues.CMAKE_MAKE_PROGRAM) -ne (NormPath $Ninja) -or (NormPath $cacheValues.CMAKE_CXX_COMPILER) -ne (NormPath $whereCl) -or $compilerConfig.cxx_id -ne 'MSVC' -or $compilerConfig.cxx_version -ne '19.51.36260.0' -or (NormPath $compilerConfig.cxx_path) -ne (NormPath $whereCl) -or $compilerConfig.cuda_id -ne 'NVIDIA' -or $compilerConfig.cuda_version -ne '13.3.73' -or (NormPath $compilerConfig.cuda_path) -ne (NormPath $Nvcc) -or $cacheValues.STRATA_ENABLE_CUDA -ne 'ON' -or $cacheValues.STRATA_NATIVE_EXPERTS -ne 'ON' -or $cacheValues.STRATA_PORTABLE -ne 'ON' -or $cacheValues.STRATA_BUILD_TESTS -ne 'OFF' -or (NormPath $cacheValues.STRATA_GGML_DIR) -ne (NormPath $Ggml) -or (NormPath $cacheValues.CUDAToolkit_ROOT) -ne (NormPath $Sdk) -or $cacheValues.GGML_CUDA -ne 'OFF' -or $badOff.Count -gt 0){throw 'CMakeCache/compiler configs do not match required pinned CUDA build controls'}
 $sourceAfterConfigure=GitBind $SourcePath;if(-not $sourceAfterConfigure.clean -or $sourceAfterConfigure.head -ne $sourceBefore.head){throw 'Source tree changed during configure'}
 $buildArgs=@('--build',$BuildPath,'--target','strata','--parallel','1','--verbose');$buildCmd='"'+$Cmake+'" '+(($buildArgs|ForEach-Object{QuoteArg $_}) -join ' ')+' > "'+(Join-Path $BuildPath 'build.log')+'" 2>&1'
 $buildResult=RunPhase 'build' @($buildCmd)
 if($buildResult.exit_code -ne 0){throw "CMake build failed: $($buildResult.exit_code)"}
 $exe=Join-Path $BuildPath 'strata.exe';if(-not(Test-Path $exe -PathType Leaf)){throw 'Build exit 0 without expected strata.exe'}
 $sourceAfter=GitBind $SourcePath;if(-not $sourceAfter.clean -or $sourceAfter.head -ne $sourceBefore.head){throw 'Source tree changed during build'}
 $ggmlAfter=GitBind $Ggml;if(-not $ggmlAfter.clean -or $ggmlAfter.head -ne $ggmlBefore.head){throw 'Pinned ggml source changed during build'}
 $toolAfter=[ordered]@{};foreach($k in @('cmake','ninja','nvcc','vcvars64')){$tp=$toolBefore[$k].path;$toolAfter[$k]=@{path=$tp;sha256=(Sha $tp)}};$toolAfter.cuda_header_sha256=(Sha (Join-Path $Sdk 'include\cuda.h'));if($toolAfter.cmake.sha256 -ne $toolBefore.cmake.sha256 -or $toolAfter.ninja.sha256 -ne $toolBefore.ninja.sha256 -or $toolAfter.nvcc.sha256 -ne $toolBefore.nvcc.sha256 -or $toolAfter.vcvars64.sha256 -ne $toolBefore.vcvars64.sha256 -or $toolAfter.cuda_header_sha256 -ne $toolBefore.cuda_header_sha256){throw 'SDK/toolchain hash changed during build'}
 $allSamples=@(Get-Content -LiteralPath $script:ResourcePath|ForEach-Object{$_|ConvertFrom-Json});if(-not $allSamples.Count){throw 'Resource sampler recorded no samples'};$sampleStats=[ordered]@{sample_count=$allSamples.Count;interval_seconds=1;minimum_ram_available_bytes=($allSamples|Measure-Object physical_available_bytes -Minimum).Minimum;minimum_commit_available_bytes=($allSamples|Measure-Object commit_available_bytes -Minimum).Minimum;ram_floor_bytes=$MinRam;commit_floor_bytes=$MinCommit;all_samples_pass=(@($allSamples|Where-Object{$_.error -or $null -eq $_.physical_available_bytes -or [int64]$_.physical_available_bytes -lt $MinRam -or $null -eq $_.commit_available_bytes -or [int64]$_.commit_available_bytes -lt $MinCommit}).Count -eq 0)}
 if(-not $sampleStats.all_samples_pass){throw 'At least one resource sample fell below 12 GiB RAM or 4 GiB commit'}
 $binaryItem=Get-Item $exe;$binary=[ordered]@{path=$exe;size_bytes=$binaryItem.Length;sha256=(Sha $exe)};$status='success'
}catch{$failure=$_.Exception.ToString();$status='failed'}
finally{
 if($script:SamplerProcess -and -not $script:SamplerProcess.HasExited){if(-not(Test-Path -LiteralPath $script:StopPath)){New-Item -ItemType File -Path $script:StopPath|Out-Null};try{$script:SamplerProcess.WaitForExit(15000)|Out-Null}catch{}}
 if(-not $sourceAfter){try{$sourceAfter=GitBind $SourcePath}catch{}}
 if(-not $ggmlAfter){try{$ggmlAfter=GitBind $Ggml}catch{}}
 if(-not $sampleStats -and (Test-Path -LiteralPath $script:ResourcePath)){try{$allSamples=@(Get-Content -LiteralPath $script:ResourcePath|ForEach-Object{$_|ConvertFrom-Json});$sampleStats=[ordered]@{sample_count=$allSamples.Count;interval_seconds=1;minimum_ram_available_bytes=($allSamples|Measure-Object physical_available_bytes -Minimum).Minimum;minimum_commit_available_bytes=($allSamples|Measure-Object commit_available_bytes -Minimum).Minimum;ram_floor_bytes=$MinRam;commit_floor_bytes=$MinCommit;all_samples_pass=(@($allSamples|Where-Object{$_.error -or $null -eq $_.physical_available_bytes -or [int64]$_.physical_available_bytes -lt $MinRam -or $null -eq $_.commit_available_bytes -or [int64]$_.commit_available_bytes -lt $MinCommit}).Count -eq 0)}}catch{}}
 $receipt=[ordered]@{schema_version=1;run_name=$RunName;status=$status;failure=$failure;receipt_write_failure=$receiptWriteFailure;source_before=$sourceBefore;source_after=$sourceAfter;resource_monitor_before=$monitorBefore;resource_monitor_after=(MonitorBind);ggml_before=$ggmlBefore;ggml_after=$ggmlAfter;source_key_files=$sourceFiles;ggml_key_files=$ggmlFiles;toolchain_before=$toolBefore;toolchain_after=$toolAfter;cmake_cache_controls=$cacheValues;configure=$configureResult;build=$buildResult;configure_argv=$configureArgs;build_argv=$buildArgs;compiler_config_files=$compilerConfig;binary=$binary;resource=$sampleStats;resource_sampler_process=$samplerIdentity;resource_sampler_stop_file=$script:StopPath;controller_process=[ordered]@{pid=$controllerId;exe=if($controllerCim){$controllerCim.ExecutablePath}else{$null};create_utc=if($controllerCim){$controllerCim.CreationDate.ToUniversalTime().ToString('o')}else{$null};priority='Idle'};build_directory=$BuildPath;gpu_executed=$false;model_or_engine_cli_executed=$false;tests_built=$false;tests_run=$false;log_paths=@('configure.log','build.log','resources.jsonl','resource_sampler.py','where-cl.log','where-nvcc.log','cl-version.log','nvcc-version.log','cmake-version.log','ninja-version.log')}
 try{WriteExclusive (Join-Path $BuildPath 'build-receipt.json') $receipt 12}catch{$receiptWriteFailure=$_.Exception.ToString();$status='failed';$receipt.status='failed';$receipt.receipt_write_failure=$receiptWriteFailure;$errPath=Join-Path (Split-Path -Parent $BuildPath) ($RunName+'.receipt-write-failure.txt');try{[IO.File]::WriteAllText($errPath,($failure+"`nReceipt write error: "+$receiptWriteFailure),[Text.UTF8Encoding]::new($false))}catch{}}
}
$receipt|ConvertTo-Json -Depth 7 -Compress
if($status -ne 'success'){exit 1}
