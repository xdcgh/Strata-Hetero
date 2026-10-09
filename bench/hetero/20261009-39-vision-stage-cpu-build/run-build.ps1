$ErrorActionPreference='Stop'
if($PSVersionTable.PSVersion.Major -lt 7){throw 'This reviewed runner requires pwsh 7'}
$workspace='C:\Users\DC\Documents\ChatGPT\Strata-Hetero'
$evidence=Join-Path $workspace 'bench\hetero\20261009-39-vision-stage-cpu-build'
$source='E:\Strata-Hetero-data\source\llama-3cf0325'
$fixtureBuild='E:\Strata-Hetero-data\build\vision-stage-fixture-20261009-39'
$fixtureFiles='E:\Strata-Hetero-data\vision-fixtures\stage-metadata-20261009-39'
$helperBuild='E:\Strata-Hetero-data\build\vision-stage-taps-cpu-20261009-39'
$cl='C:\Program Files (x86)\Microsoft Visual Studio\18\BuildTools\VC\Tools\MSVC\14.51.36231\bin\Hostx64\x64\cl.exe'
$vcvars='C:\Program Files (x86)\Microsoft Visual Studio\18\BuildTools\VC\Auxiliary\Build\vcvars64.bat'
$cmake='E:\Strata-Hetero-data\venv\Lib\site-packages\cmake\data\bin\cmake.exe'
$ninja='E:\Strata-Hetero-data\venv\Scripts\ninja.exe'
$python='E:\Strata-Hetero-data\venv-intel\Scripts\python.exe'
$old='E:\Strata-Hetero-data\build\vision-oracle-cpu-20261009-20\bin\strata-vision.exe'
$oldSha='7285dd08f24c3985fd6b97aaff7d9917b65e5ea8b263ae5567985d0dc04ba612'
$compilerPattern='^(cl|link|clang|clang\+\+|clang-cl|lld-link|cmake|ninja|MSBuild|nvcc|cudafe\+\+|ptxas|cicc)\.exe$'
foreach($target in @($fixtureBuild,$fixtureFiles,$helperBuild)){if(Test-Path -LiteralPath $target){throw "Exclusive target already exists: $target"}}
if((Get-FileHash -LiteralPath $old -Algorithm SHA256).Hash.ToLower() -ne $oldSha){throw 'Original run20 oracle digest changed'}
function Gate([string]$phase) {
    $raw=& $python -B -c 'import json;from tools.hetero_resources import memory_snapshot;m,e=memory_snapshot();print(json.dumps({"memory":m,"errors":e}))'
    $code=$LASTEXITCODE
    $sample=$raw|ConvertFrom-Json -AsHashtable
    $passed=$code -eq 0 -and $sample.errors.Count -eq 0 -and $sample.memory.physical_available_bytes -ge 12GB -and $sample.memory.commit_available_bytes -ge 4GB
    [ordered]@{utc=[DateTime]::UtcNow.ToString('o');phase=$phase;pass=$passed;sample=$sample}|ConvertTo-Json -Compress -Depth 8|Add-Content -LiteralPath (Join-Path $evidence 'resource.jsonl') -Encoding utf8
    if(-not $passed){throw 'Fresh known 12/4 resource gate failed'}
}
function Tree([int]$owner) {
    $all=@(Get-CimInstance Win32_Process)
    $ids=[System.Collections.Generic.HashSet[int]]::new();[void]$ids.Add($owner)
    $again=$true
    while($again){$again=$false;foreach($row in $all){if($ids.Contains([int]$row.ParentProcessId) -and $ids.Add([int]$row.ProcessId)){$again=$true}}}
    @($all|Where-Object{$ids.Contains([int]$_.ProcessId)})
}
function Phase([string]$name,[string]$command) {
    $overlap=@(Get-CimInstance Win32_Process|Where-Object{$_.Name -match $compilerPattern})
    $overlap|Select-Object ProcessId,ParentProcessId,Name,CreationDate,ExecutablePath,CommandLine|ConvertTo-Json -Depth 5|Set-Content -LiteralPath (Join-Path $evidence ($name+'.overlap.json')) -Encoding utf8
    if($overlap.Count){throw 'Another compiler is running; phase refused'}
    Gate ('before-'+$name)
    $script=Join-Path $evidence ($name+'.cmd')
    $log=Join-Path $evidence ($name+'.log')
    $exitFile=Join-Path $evidence ($name+'.exit.txt')
    if(Test-Path -LiteralPath $script){throw 'Owned phase script exists'}
    @"
@echo off
call "$vcvars"
if errorlevel 1 exit /b %errorlevel%
$command
set "PHASE_RC=%ERRORLEVEL%"
>"$exitFile" echo %PHASE_RC%
exit /b %PHASE_RC%
"@|Set-Content -LiteralPath $script -Encoding ascii
    $proc=Start-Process -FilePath "$env:WINDIR\System32\cmd.exe" -ArgumentList @('/d','/c','call',"`"$script`"") -WorkingDirectory $workspace -WindowStyle Hidden -RedirectStandardOutput $log -RedirectStandardError (Join-Path $evidence ($name+'.stderr.log')) -PassThru
    $proc.PriorityClass='Idle'
    $started=[ordered]@{phase=$name;utc=[DateTime]::UtcNow.ToString('o');pid=$proc.Id;create_utc=$proc.StartTime.ToUniversalTime().ToString('o');priority='Idle';argv=@('/d','/c','call',$script);actual_executed_script=$script;script_sha256=(Get-FileHash -LiteralPath $script -Algorithm SHA256).Hash.ToLower();command=$command}
    $started|ConvertTo-Json -Depth 5|Set-Content -LiteralPath (Join-Path $evidence ($name+'.process.json')) -Encoding utf8
    try {
        while(-not $proc.HasExited){
            $tree=@(Tree $proc.Id)
            foreach($row in $tree){try{(Get-Process -Id $row.ProcessId -ErrorAction Stop).PriorityClass='Idle'}catch{}}
            [ordered]@{utc=[DateTime]::UtcNow.ToString('o');phase=$name;tree=@($tree|Select-Object ProcessId,ParentProcessId,Name,CreationDate,ExecutablePath,CommandLine)}|ConvertTo-Json -Compress -Depth 7|Add-Content -LiteralPath (Join-Path $evidence 'process-tree.jsonl') -Encoding utf8
            Gate $name
            Start-Sleep -Seconds 2
            $proc.Refresh()
        }
        $proc.WaitForExit()
    } catch {
        foreach($bound in (@(Tree $proc.Id)|Sort-Object ProcessId -Descending)){
            $fresh=Get-CimInstance Win32_Process -Filter ('ProcessId='+$bound.ProcessId) -ErrorAction SilentlyContinue
            if($fresh -and $fresh.CreationDate -eq $bound.CreationDate -and $fresh.ExecutablePath -eq $bound.ExecutablePath -and $fresh.CommandLine -eq $bound.CommandLine){try{Stop-Process -Id $bound.ProcessId -ErrorAction Stop}catch{}}
        }
        throw
    } finally {
        $started['ended_utc']=[DateTime]::UtcNow.ToString('o');$started['exit_code']=$proc.ExitCode
        $started|ConvertTo-Json -Depth 5|Set-Content -LiteralPath (Join-Path $evidence ($name+'.terminal.json')) -Encoding utf8
    }
    if($proc.ExitCode -ne 0){throw "Phase $name failed with exit $($proc.ExitCode)"}
}
$hashPaths=@('tools\vision\strata_vision.cpp','tools\vision\vision_stage_trace.hpp','tools\vision\CMakeLists.txt','tools\hetero_native_cpu\sha256.hpp','tests\core\vision_stage_trace_test.cpp','bench\hetero\20261009-39-vision-stage-cpu-build\run-build.ps1')
$before=@($hashPaths|ForEach-Object{[ordered]@{path=$_;sha256=(Get-FileHash -LiteralPath $_ -Algorithm SHA256).Hash.ToLower()}})
$before|ConvertTo-Json -Depth 4|Set-Content -LiteralPath (Join-Path $evidence 'source-before.json') -Encoding utf8
$null=New-Item -ItemType Directory -Path $fixtureBuild -ErrorAction Stop
$fixtureExe=Join-Path $fixtureBuild 'vision-stage-trace-fixture.exe'
try {
    Phase 'fixture-compile' "`"$cl`" /nologo /std:c++17 /EHsc /O2 /MT /arch:AVX2 /I`"$source\ggml\include`" /Fo`"$fixtureBuild\fixture.obj`" /Fe`"$fixtureExe`" `"$workspace\tests\core\vision_stage_trace_test.cpp`""
    Phase 'fixture-execute' "`"$fixtureExe`" `"$fixtureFiles`""
    $config="`"$cmake`" -S `"$workspace\tools\vision`" -B `"$helperBuild`" -G Ninja -DCMAKE_MAKE_PROGRAM=`"$ninja`" -DCMAKE_BUILD_TYPE=Release -DCMAKE_EXPORT_COMPILE_COMMANDS=ON -DLLAMA_DIR=`"$source`" -DLLAMA_BUILD_COMMON=OFF -DLLAMA_BUILD_MTMD=ON -DLLAMA_BUILD_TOOLS=OFF -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF -DLLAMA_BUILD_SERVER=OFF -DLLAMA_BUILD_APP=OFF -DSTRATA_VISION_CUDA=OFF -DSTRATA_PORTABLE=ON -DGGML_CUDA=OFF -DGGML_HIP=OFF -DGGML_SYCL=OFF -DGGML_VULKAN=OFF -DGGML_OPENCL=OFF -DGGML_NATIVE=OFF -DGGML_OPENMP=OFF"
    Phase 'configure' $config
    Phase 'build' "`"$cmake`" --build `"$helperBuild`" --target strata-vision --parallel 1"
    $binary=Join-Path $helperBuild 'bin\strata-vision.exe'
    if(-not (Test-Path -LiteralPath $binary)){throw 'Build exited zero without expected helper'}
    $after=@($hashPaths|ForEach-Object{[ordered]@{path=$_;sha256=(Get-FileHash -LiteralPath $_ -Algorithm SHA256).Hash.ToLower()}})
    $after|ConvertTo-Json -Depth 4|Set-Content -LiteralPath (Join-Path $evidence 'source-after.json') -Encoding utf8
    if(($before|ConvertTo-Json -Compress) -ne ($after|ConvertTo-Json -Compress)){throw 'Source changed during compilation'}
    if((Get-FileHash -LiteralPath $old -Algorithm SHA256).Hash.ToLower() -ne $oldSha){throw 'Original oracle bytes changed'}
    [ordered]@{schema_version=1;status='fixture_pass_helper_built_not_run';created_utc=[DateTime]::UtcNow.ToString('o');powershell=$PSVersionTable.PSVersion.ToString();fixture_binary=[ordered]@{path=$fixtureExe;sha256=(Get-FileHash -LiteralPath $fixtureExe -Algorithm SHA256).Hash.ToLower()};fixture_owned_directory=$fixtureFiles;helper_binary=[ordered]@{path=$binary;bytes=(Get-Item -LiteralPath $binary).Length;sha256=(Get-FileHash -LiteralPath $binary -Algorithm SHA256).Hash.ToLower()};old_oracle_sha256_unchanged=$oldSha;source_files_unchanged=$true;pinned_source=[string](& git -C $source rev-parse HEAD);helper_started=$false;gguf_loaded=$false;openvino_core_created=$false;gpu_npu_run=$false;benchmark_run=$false}|ConvertTo-Json -Depth 6|Set-Content -LiteralPath (Join-Path $evidence 'build-receipt.json') -Encoding utf8
} catch {
    [ordered]@{schema_version=1;status='failed_preserved';utc=[DateTime]::UtcNow.ToString('o');error=$_.Exception.Message;script=$_.ScriptStackTrace;helper_started=$false;gguf_loaded=$false}|ConvertTo-Json -Depth 5|Set-Content -LiteralPath (Join-Path $evidence 'failure.json') -Encoding utf8
    throw
}
