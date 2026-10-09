param([switch]$Start)

$ErrorActionPreference = 'Stop'
$repo = 'C:\Users\DC\Documents\ChatGPT\Strata-Hetero'
$run = Join-Path $repo 'bench\hetero\20261009-41-hetero41-direct-quality'
$configPath = Join-Path $run 'config.json'
$identityPath = Join-Path $run 'identity.json'
$provenancePath = Join-Path $run 'provenance.json'
$freshIdentityPath = Join-Path $run 'fresh-identities.json'
$manifestPath = Join-Path $repo 'bench\hetero\20261009-06-quality-prompts\manifest.json'
$serverRoot = 'E:\Strata-Hetero-data\source\hetero-0-1-41'
$serverScript = Join-Path $serverRoot 'serve\server.py'
$python = 'E:\Strata-Hetero-data\venv\Scripts\python.exe'
$stdlibPython = 'C:\Python314\python.exe'
$admissionPath = Join-Path $run 'admission.json'

function Get-Sha256([string]$Path) {
    (Get-FileHash -LiteralPath $Path -Algorithm SHA256).Hash.ToLowerInvariant()
}

function Write-ExclusiveJson([string]$Path, $Value, [int]$Depth = 12) {
    $bytes = [Text.UTF8Encoding]::new($false).GetBytes(($Value | ConvertTo-Json -Depth $Depth) + "`n")
    $stream = [IO.File]::Open($Path, [IO.FileMode]::CreateNew, [IO.FileAccess]::Write, [IO.FileShare]::Read)
    try { $stream.Write($bytes, 0, $bytes.Length); $stream.Flush($true) } finally { $stream.Dispose() }
}

function Get-InputIdentityRecheck($Config, $Fresh) {
    $expectedPaths = @()
    for ($i = 0; $i -lt $Config.args.Count; $i++) {
        if ($Config.args[$i] -eq '--native-dense-gguf' -and $i + 1 -lt $Config.args.Count) { $expectedPaths += $Config.args[$i + 1] }
    }
    $pleIndex = [array]::IndexOf($Config.args, '--ple-gguf')
    if ($pleIndex -lt 0 -or $pleIndex + 1 -ge $Config.args.Count) { throw 'prepared config has no exact --ple-gguf input' }
    $plePath = [string]$Config.args[$pleIndex + 1]
    if ($plePath -notlike 'F:\Strata-data\models\*' -or $plePath -like 'E:\*') { throw 'F arm PLE path is not the original F model file' }
    if (@($expectedPaths | Where-Object { $_ -notlike 'F:\Strata-data\models\*' -or $_ -like 'E:\*' }).Count -gt 0) {
        throw 'F arm native-dense list contains a non-F model path'
    }
    $expectedPaths = @($expectedPaths | Select-Object -Unique)
    if ($expectedPaths.Count -ne 4) { throw 'F arm must retain exactly four distinct F native-dense shards' }
    $script = @'
import json, pathlib, sys
root=pathlib.Path(sys.argv[1]); fresh=json.loads(pathlib.Path(sys.argv[2]).read_text(encoding="utf-8"))
paths=json.loads(__import__("base64").b64decode(sys.argv[3])); sys.path.insert(0,str(root/"tools")); import hetero_verify_existing as verify
if fresh.get("all_current_identities_match") is not True: raise SystemExit("prepared fresh identity receipt is not passing")
records={str(x.get("path","")).casefold():x for x in fresh.get("input_files",[])}
out=[]
for p in paths:
    rec=records.get(str(p).casefold())
    if not rec: raise SystemExit("F input absent from fresh identity receipt: "+p)
    exp=rec["current_identity"]; got=verify.file_stat(pathlib.Path(p))
    ok=(got["size_bytes"]==exp["size_bytes"] and got["mtime_ns"]==exp["mtime_ns"] and
        got["ctime_ns"]==exp["ctime_ns"] and got["file_id"]["volume"]==exp["device"] and
        got["file_id"]["index"]==exp["file_index"])
    if not ok: raise SystemExit("F file stat identity changed: "+p)
    out.append({"path":p,"role":rec.get("role"),"stat":got,"prior_full_sha256":rec.get("prior_sha256",rec.get("prior_verified_sha256")),"matches_prepared_identity":True})
prior_path=root/"bench"/"hetero"/"20261008-00-admission"/"model-integrity-02.json"
prior=json.loads(prior_path.read_text(encoding="utf-8")); runtime=[]
for category,group in prior.get("runtime_identity",{}).items():
    for item in group.get("files",[]):
        path=pathlib.Path(item["path"]); exp=item["after"]; got=verify.file_stat(path)
        ok=(got["size_bytes"]==exp["size_bytes"] and got["mtime_ns"]==exp["mtime_ns"] and
            got["ctime_ns"]==exp["ctime_ns"] and got["file_id"]==exp["file_id"])
        if not ok: raise SystemExit("pack/MTP runtime stat identity changed: "+str(path))
        runtime.append({"category":category,"path":str(path),"stat":got,"prior_full_sha256":item["sha256"],"matches_prior_stat":True})
if not runtime: raise SystemExit("prior pack/MTP runtime identity inventory is empty")
print(json.dumps({"status":"pass","method":"stat/file-ID only; original payload hashes retained; no model/runtime payload read",
                  "files":out,"runtime_identity_files":runtime},separators=(",",":")))
'@
    $jsonPaths = ConvertTo-Json -InputObject $expectedPaths -Compress
    $encodedPaths = [Convert]::ToBase64String([Text.Encoding]::UTF8.GetBytes($jsonPaths))
    $encodedCode = [Convert]::ToBase64String([Text.Encoding]::UTF8.GetBytes($script))
    $wrapper = "exec(__import__('base64').b64decode('$encodedCode'))"
    $pythonArgs = @('-I','-B','-X','utf8','-c',$wrapper,$repo,$freshIdentityPath,$encodedPaths)
    $raw = & $stdlibPython @pythonArgs
    if ($LASTEXITCODE -ne 0) { throw 'F model stat/fstat identity recheck failed' }
    $check = (($raw -join "`n") | ConvertFrom-Json)
    if (@($check.runtime_identity_files).Count -ne 14) { throw 'Prior pack/MTP runtime identity receipt did not contain the expected 14 files' }
    $check | Add-Member -NotePropertyName ple_table_selection_path -NotePropertyValue $plePath -Force
    $check | Add-Member -NotePropertyName ple_table_identity_matches_shared_F_shard -NotePropertyValue $true -Force
    return $check
}

function Test-PreparedContract {
    foreach ($path in @($configPath,$identityPath,$provenancePath,$freshIdentityPath,$manifestPath,$python,$stdlibPython)) {
        if (-not (Test-Path -LiteralPath $path -PathType Leaf)) { throw "required prepared input missing: $path" }
    }
    $config = Get-Content -LiteralPath $configPath -Raw | ConvertFrom-Json
    $identity = Get-Content -LiteralPath $identityPath -Raw | ConvertFrom-Json
    $provenance = Get-Content -LiteralPath $provenancePath -Raw | ConvertFrom-Json
    if ($provenance.run_id -ne '20261009-41-hetero41-direct-quality' -or $provenance.shared_controls.ple_io -ne 'direct' -or
        $provenance.placement_contrast.path -notlike 'F:\Strata-data\models\*') { throw 'provenance is not the prepared original-F direct-ple placement arm' }
    if ($provenance.acceptance_status -ne 'prepared_not_launched' -or $identity.acceptance_status -ne 'prepared_not_launched') { throw 'prepared identity/provenance status changed' }
    if ($provenance.engine_sha256 -ne $identity.engine.sha256 -or $provenance.engine_path -ne $identity.engine.path -or
        $config.exe -ne $identity.engine.path) { throw 'flat provenance/identity/config engine identity mismatch' }
    if ($provenance.server_file_sha256 -ne '0ca2dd70485e5e03b0a6fcb5dfc1ee1a8863d2dc44947257e342ae608326b026') {
        throw 'prepared bridge identity differs from the reviewed server.py SHA-256'
    }
    if ((Get-Sha256 $configPath) -ne $identity.config.sha256 -or (Get-Sha256 $config.exe) -ne $provenance.engine_sha256 -or
        (Get-Sha256 $serverScript) -ne $provenance.server_file_sha256) { throw 'config/engine/server bridge hash mismatch' }
    if ($config.port -ne 8081 -or $config.host -ne '127.0.0.1' -or $config.model_name -ne $identity.model.id) {
        throw 'prepared server bind/model identity changed'
    }
    if ($config.env.STRATA_UNBUFFERED_LOAD -ne '0' -or $config.env.STRATA_IQ_MT_MIN -ne '1' -or
        $config.env.STRATA_PREFILL_CPU_SHARE -ne '0' -or $config.env.STRATA_STAGE_PIN -ne '0') { throw 'prepared process-local runtime environment changed' }
    $ple = [array]::IndexOf($config.args, '--ple-gguf')
    $native = [array]::IndexOf($config.args, '--native')
    $resident = [array]::IndexOf($config.args, '--resident-budget-gib')
    $nativeArgs = @(); for ($i=0; $i -lt $config.args.Count; $i++) { if ($config.args[$i] -eq '--native-dense-gguf') { $nativeArgs += $config.args[$i+1] } }
    if ($native -lt 0 -or $config.args[$native+1] -ne $identity.model.native_path -or $identity.model.native_path -notlike 'F:\Strata-data\models\*' -or
        $ple -lt 0 -or $config.args[$ple+1] -ne 'F:\Strata-data\models\unsloth-UD-Q4_K_XL\Qwen3.8-Flash-Next-UD-Q4_K_XL-00002-of-00004.gguf' -or
        $nativeArgs.Count -ne 4 -or @($nativeArgs | Where-Object { $_ -like 'E:\*' }).Count -gt 0 -or
        $resident -lt 0 -or $config.args[$resident+1] -ne '20' -or $config.args -notcontains '--ple-io' -or
        $config.args[([array]::IndexOf($config.args, '--ple-io')+1)] -ne 'direct') { throw 'F placement/shared resident controls do not match prepared run11 config' }
    if ($provenance.placement_contrast.path -ne $config.args[$ple+1] -or
        $provenance.placement_contrast.sha256 -ne '3f342f1c1580473f1ee94ddd5b28206e8c07a70fa1a366f59d1d6c922919a6c9') {
        throw 'prepared F PLE source identity differs from the original F shard/copy receipt'
    }
    if ((Get-Sha256 $manifestPath) -ne $provenance.shared_controls.quality_prompt_manifest_sha256) { throw 'quality manifest SHA differs from prepared provenance' }
    $fresh = Get-Content -LiteralPath $freshIdentityPath -Raw | ConvertFrom-Json
    $inputIdentity = Get-InputIdentityRecheck $config $fresh
    $profileIndex = [array]::IndexOf($config.args, '--expert-profile')
    if ($profileIndex -lt 0 -or -not (Test-Path -LiteralPath $config.args[$profileIndex+1] -PathType Leaf)) { throw 'configured expert profile missing' }
    $profile = Get-Item -LiteralPath $config.args[$profileIndex+1]
    $profileEvidence = [ordered]@{path=$profile.FullName;size_bytes=$profile.Length;sha256=(Get-Sha256 $profile.FullName)}
    return [pscustomobject]@{config=$config;identity=$identity;provenance=$provenance;input_identity=$inputIdentity;expert_profile=$profileEvidence;
        config_sha256=(Get-Sha256 $configPath);engine_sha256=(Get-Sha256 $config.exe);server_file_sha256=(Get-Sha256 $serverScript);
        manifest_sha256=(Get-Sha256 $manifestPath)}
}

$contract = Test-PreparedContract
$reserved = @('admission.json','identity-recheck.json','process.json','sampler-process.json','launch-status.json',
              'logs\server.stdout.log','logs\server.stderr.log','logs\sampler.stdout.log','logs\sampler.stderr.log',
              'resource\samples.jsonl','resource\STOP')
foreach ($rel in $reserved) { if (Test-Path -LiteralPath (Join-Path $run $rel)) { throw "refusing overwrite of launch evidence: $rel" } }
$response = [ordered]@{status='prepared_validated';run_id='20261009-41-hetero41-direct-quality';start_performed=$false;
    config_sha256=$contract.config_sha256;engine_sha256=$contract.engine_sha256;server_file_sha256=$contract.server_file_sha256;
    manifest_sha256=$contract.manifest_sha256;f_model_files_stat_rechecked=$contract.input_identity.files.Count;
    pack_mtp_runtime_files_stat_rechecked=$contract.input_identity.runtime_identity_files.Count;expert_profile=$contract.expert_profile;
    admission_required=@{minimum_ram_gib=12;minimum_commit_gib=4;required_vram_mib=46000;wsl_shutdown_before_observe=$true}}
if (-not $Start) { $response | ConvertTo-Json -Depth 8 -Compress; exit 0 }

$listen = @(Get-NetTCPConnection -LocalPort 8081 -State Listen -ErrorAction SilentlyContinue)
if ($listen.Count -gt 0) { throw '8081 is occupied; no process was stopped.' }
$active = @(Get-CimInstance Win32_Process | Where-Object { $_.Name -in @('strata.exe','ninja.exe','cl.exe','hetero_native_expert.exe') })
if ($active.Count -gt 0) { throw 'A Strata or compilation workload is active; refusing launch.' }
$inheritedStrataNames = @([Environment]::GetEnvironmentVariables('Process').Keys | Where-Object { $_ -like 'STRATA_*' })
foreach ($name in $inheritedStrataNames) { [Environment]::SetEnvironmentVariable([string]$name,$null,'Process') }

# This shutdown is only reached after the explicit -Start flag; it is the authorized owned pre-admission WSL shutdown.
& wsl.exe --shutdown
if ($LASTEXITCODE -ne 0) { throw 'Authorized WSL shutdown failed; server was not started.' }
& $stdlibPython -I -B -X utf8 (Join-Path $repo 'tools\hetero_admission.py') --observe --output $admissionPath `
    --required-vram-mib 46000 --minimum-ram-gib 12 --check-wsl-current
if ($LASTEXITCODE -ne 0) { throw 'Fresh stdlib admission failed; no model server was started.' }
$admission = Get-Content -LiteralPath $admissionPath -Raw | ConvertFrom-Json
$memory = $admission.raw.windows_memory
if ($admission.pass -ne $true -or $admission.status -ne 'pass' -or $null -eq $memory.available_bytes -or
    $memory.available_bytes -lt 12GB -or $null -eq $memory.available_commit_bytes -or $memory.available_commit_bytes -lt 4GB -or
    $admission.chosen_gates.vram.pass -ne $true -or $admission.chosen_gates.vram.available_mib -lt 46000) {
    throw 'Fresh 12 GiB RAM / 4 GiB commit / 46000 MiB VRAM admission failed; no model server was started.'
}
# Recheck the original F file IDs after WSL shutdown and admission, immediately before creating the server.
$afterAdmission = Test-PreparedContract
if ($afterAdmission.config_sha256 -ne $contract.config_sha256 -or $afterAdmission.engine_sha256 -ne $contract.engine_sha256 -or
    $afterAdmission.server_file_sha256 -ne $contract.server_file_sha256) { throw 'Static config/runtime identity changed during admission' }
$contract = $afterAdmission
$listeners = @(Get-NetTCPConnection -LocalPort 8081 -State Listen -ErrorAction SilentlyContinue)
if ($listeners.Count -gt 0) { throw 'Port 8081 became occupied after admission; no process was stopped.' }
$active = @(Get-CimInstance Win32_Process | Where-Object { $_.Name -in @('strata.exe','ninja.exe','cl.exe','hetero_native_expert.exe') })
if ($active.Count -gt 0) { throw 'A Strata or compilation workload became active after admission; refusing launch.' }
$identityRecheck = [ordered]@{status=$contract.input_identity.status;method=$contract.input_identity.method;
    files=$contract.input_identity.files;runtime_identity_files=$contract.input_identity.runtime_identity_files;
    ple_table_selection_path=$contract.input_identity.ple_table_selection_path;
    ple_table_identity_matches_shared_F_shard=$contract.input_identity.ple_table_identity_matches_shared_F_shard;
    engine_sha256=$contract.engine_sha256;server_file_sha256=$contract.server_file_sha256;config_sha256=$contract.config_sha256;
    manifest_sha256=$contract.manifest_sha256;expert_profile=$contract.expert_profile}
Write-ExclusiveJson (Join-Path $run 'identity-recheck.json') $identityRecheck
$serverArgs = @('-u','-X','utf8','-B',$serverScript,'--engine','strata','--config',$configPath,'--port','8081')
$server = Start-Process -FilePath $python -ArgumentList $serverArgs -WorkingDirectory $repo -WindowStyle Hidden -PassThru `
    -RedirectStandardOutput (Join-Path $run 'logs\server.stdout.log') -RedirectStandardError (Join-Path $run 'logs\server.stderr.log')
$serverCim = Get-CimInstance Win32_Process -Filter "ProcessId = $($server.Id)" -ErrorAction SilentlyContinue
if (-not $serverCim -or $serverCim.ExecutablePath -ne $python) { throw 'E venv server launcher PID/executable identity is unavailable' }
$serverStart = $serverCim.CreationDate.ToUniversalTime().ToString('o')
# Start the owner sampler as soon as the run-owned server launcher exists; its owner PID is this server launcher.
$samplerArgs = @('-u','-X','utf8','-B',(Join-Path $repo 'tools\hetero_resources.py'),'--output',(Join-Path $run 'resource\samples.jsonl'),
    '--pid',[string]$server.Id,'--interval','1','--max-seconds','10800','--minimum-commit-gib','4','--stop-file',(Join-Path $run 'resource\STOP'))
$sampler = Start-Process -FilePath $python -ArgumentList $samplerArgs -WorkingDirectory $repo -WindowStyle Hidden -PassThru `
    -RedirectStandardOutput (Join-Path $run 'logs\sampler.stdout.log') -RedirectStandardError (Join-Path $run 'logs\sampler.stderr.log')
$samplerCimLauncher = Get-CimInstance Win32_Process -Filter "ProcessId = $($sampler.Id)" -ErrorAction SilentlyContinue
if (-not $samplerCimLauncher -or $samplerCimLauncher.ExecutablePath -ne $python) { throw 'E venv sampler launcher PID/executable identity is unavailable' }
$samplerStart = if ($samplerCimLauncher) { $samplerCimLauncher.CreationDate.ToUniversalTime().ToString('o') } else { $null }
$samplerCim = $null
for ($i=0; $i -lt 50 -and -not $samplerCim; $i++) {
    $samplerCim = Get-CimInstance Win32_Process | Where-Object {
        $_.ParentProcessId -eq $sampler.Id -and $_.ExecutablePath -eq 'C:\Python314\python.exe' -and
        $_.CommandLine.Contains('hetero_resources.py') -and $_.CommandLine.Contains('samples.jsonl')
    } | Select-Object -First 1
    if (-not $samplerCim) { Start-Sleep -Milliseconds 200 }
}
if (-not $samplerCim -or $samplerCim.ExecutablePath -ne 'C:\Python314\python.exe' -or
    $samplerCim.ParentProcessId -ne $sampler.Id -or -not $samplerCim.CommandLine.Contains('hetero_resources.py')) {
    $failure=[ordered]@{schema_version=1;run_id='20261009-41-hetero41-direct-quality';status='failed_sampler_child_identity';
        server_launcher_pid=$server.Id;server_launcher_create_utc=$serverStart;sampler_launcher_pid=$sampler.Id;
        sampler_child_pid_seen=if($samplerCim){$samplerCim.ProcessId}else{$null};checked_utc=[DateTime]::UtcNow.ToString('o')}
    Write-ExclusiveJson (Join-Path $run 'launch-failure.json') $failure
    throw 'Owned resource sampler actual child identity is unavailable; server/sampler are preserved for root review.'
}
$samplerReceipt = [ordered]@{schema_version=1;run_id='20261009-41-hetero41-direct-quality';sampler_launcher_pid=$sampler.Id;
    sampler_launcher_exe=if($samplerCimLauncher){$samplerCimLauncher.ExecutablePath}else{$null};sampler_launcher_create_utc=$samplerStart;
    configured_python=$python;actual_child_pid=$samplerCim.ProcessId;actual_child_parent_pid=$samplerCim.ParentProcessId;
    actual_child_exe=$samplerCim.ExecutablePath;actual_child_create_utc=$samplerCim.CreationDate.ToUniversalTime().ToString('o');
    argv=$samplerArgs;owner_pid=$server.Id;output=(Join-Path $run 'resource\samples.jsonl')}
Write-ExclusiveJson (Join-Path $run 'sampler-process.json') $samplerReceipt
$serverChildCim = $null
for ($i=0; $i -lt 100 -and -not $serverChildCim; $i++) {
    $serverChildCim = Get-CimInstance Win32_Process | Where-Object {
        $_.ParentProcessId -eq $server.Id -and $_.Name -eq 'python.exe' -and $_.CommandLine.Contains('serve\server.py') -and $_.CommandLine.Contains($configPath)
    } | Select-Object -First 1
    if (-not $serverChildCim) { Start-Sleep -Milliseconds 200 }
}
if (-not $serverChildCim -or $serverChildCim.ExecutablePath -ne 'C:\Python314\python.exe') {
    $failure=[ordered]@{schema_version=1;run_id='20261009-41-hetero41-direct-quality';status='failed_server_child_identity';
        server_launcher_pid=$server.Id;server_launcher_create_utc=$serverStart;sampler_launcher_pid=$sampler.Id;
        sampler_actual_child_pid=$samplerCim.ProcessId;sampler_actual_child_create_utc=$samplerCim.CreationDate.ToUniversalTime().ToString('o');
        checked_utc=[DateTime]::UtcNow.ToString('o')}
    Write-ExclusiveJson (Join-Path $run 'launch-failure.json') $failure
    throw 'The actual C Python server child identity is unavailable; server/sampler are preserved for root review.'
}
$serverChildStart = $serverChildCim.CreationDate.ToUniversalTime().ToString('o')
$process = [ordered]@{schema_version=1;run_id='20261009-41-hetero41-direct-quality';launcher_pid=$server.Id;start_utc=$serverStart;
    python=$python;executable=$python;launcher_exe=$serverCim.ExecutablePath;launcher_create_utc=$serverStart;argv=$serverArgs;
    actual_server_child_pid=$serverChildCim.ProcessId;actual_server_child_parent_pid=$serverChildCim.ParentProcessId;
    actual_server_child_exe=$serverChildCim.ExecutablePath;actual_server_child_create_utc=$serverChildStart;config=$configPath;config_sha256=$contract.config_sha256;
    engine_path=$config.exe;engine_sha256=$contract.engine_sha256;engine_source_sha=$contract.provenance.engine_cpp_sha;
    server_file_sha256=$contract.server_file_sha256;identity='identity.json';provenance='provenance.json';
    expert_profile=$contract.expert_profile;admission_receipt='admission.json';admission_sha256=(Get-Sha256 $admissionPath);
    identity_recheck='identity-recheck.json';sampler_receipt='sampler-process.json';role='owned PLE F quality server';
    inherited_strata_names_cleared=$inheritedStrataNames}
Write-ExclusiveJson (Join-Path $run 'process.json') $process
$response.status='launched_not_yet_ready';$response.start_performed=$true;$response.launcher_pid=$server.Id;$response.sampler_pid=$sampler.Id
$response|ConvertTo-Json -Depth 8 -Compress
