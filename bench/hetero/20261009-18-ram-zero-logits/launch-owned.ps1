$ErrorActionPreference = 'Stop'
$repo = 'C:\Users\DC\Documents\ChatGPT\Strata-Hetero'
$run = Join-Path $repo 'bench\hetero\20261009-18-ram-zero-logits'
$config = Join-Path $run 'config.json'
$python = 'E:\Strata-Hetero-data\venv\Scripts\python.exe'
$stdlibPython = 'C:\Python314\python.exe'
foreach ($name in @('process.json', 'admission.json', 'resource\samples.jsonl', 'resource\STOP')) {
    if (Test-Path -LiteralPath (Join-Path $run $name)) { throw "Run already has launch evidence: $name" }
}
if (Get-NetTCPConnection -LocalPort 8081 -State Listen -ErrorAction SilentlyContinue) {
    throw 'Port 8081 is occupied; no process was stopped.'
}
if (Get-Process -Name strata,ninja,cl,hetero_native_expert -ErrorAction SilentlyContinue) {
    throw 'An inference or compilation workload is still active; no process was stopped.'
}
# Clear inherited experiment knobs only in this helper process, before the explicit config is applied.
$removedNames = @([Environment]::GetEnvironmentVariables('Process').Keys | Where-Object { $_ -like 'STRATA_*' })
foreach ($name in $removedNames) { [Environment]::SetEnvironmentVariable($name, $null, 'Process') }

# Verify current identities against the completed full integrity run, without re-reading 111 GB.
$recheck = @'
import hashlib,json,pathlib,sys
sys.path.insert(0,str(pathlib.Path.cwd()/"tools"))
import hetero_verify_existing as verify
root=pathlib.Path(sys.argv[1])
prior_path=root.parent/"20261008-00-admission"/"model-integrity-02.json"
prior=json.loads(prior_path.read_text())
assert prior["status"]=="pass" and prior["claims"]["model_weight_integrity"] is True
records=prior["model_shards"]+sum((v["files"] for v in prior["runtime_identity"].values()),[])
checked=[]
for item in records:
    current=verify.file_stat(pathlib.Path(item["path"]))
    if current!=item["after"]: raise RuntimeError("model/runtime file identity changed: "+item["path"])
    checked.append({"path":item["path"],"identity_matches_prior_after":True})
config=json.loads((root/"config.json").read_text())
provenance=json.loads((root/"provenance.json").read_text())
actual=hashlib.sha256(pathlib.Path(config["exe"]).read_bytes()).hexdigest()
assert actual==provenance["engine_cpp"]["binary_sha256"]
assert hashlib.sha256(pathlib.Path("serve/server.py").read_bytes()).hexdigest()==provenance["server_python"]["server_file_sha256"]
payload={"status":"pass","model_runtime_files":checked,"binary_sha256":actual,"prior_integrity_receipt_sha256":hashlib.sha256(prior_path.read_bytes()).hexdigest(),"scope":"current identity comparison and binary hash; no fresh full model hash"}
with (root/"identity-recheck.json").open("x",encoding="utf-8",newline="\n") as f: json.dump(payload,f,indent=2); f.write("\n")
'@
& $stdlibPython -B -X utf8 -c $recheck $run
if ($LASTEXITCODE -ne 0) { throw 'Identity recheck failed; no model started.' }

& wsl.exe --shutdown
if ($LASTEXITCODE -ne 0) { throw 'Authorized temporary WSL shutdown failed; no model started.' }
& $stdlibPython -B -X utf8 (Join-Path $repo 'tools\hetero_admission.py') --observe --output (Join-Path $run 'admission.json')
if ($LASTEXITCODE -ne 0) { throw 'Fresh admission failed; no model started.' }
$admission = Get-Content -LiteralPath (Join-Path $run 'admission.json') -Raw | ConvertFrom-Json
if ($admission.pass -ne $true -or $admission.raw.windows_memory.available_commit_bytes -lt 4GB) {
    throw 'Fresh memory/commit admission failed; no model started.'
}
$serverArgs = @('-u','-X','utf8','-B',(Join-Path $repo 'serve\server.py'),'--engine','strata','--config',$config,'--port','8081')
$server = Start-Process -FilePath $python -ArgumentList $serverArgs -WorkingDirectory $repo -WindowStyle Hidden -PassThru -RedirectStandardOutput (Join-Path $run 'logs\server.stdout.log') -RedirectStandardError (Join-Path $run 'logs\server.stderr.log')
$process = [ordered]@{schema_version=1; launcher_pid=$server.Id; start_utc=$server.StartTime.ToUniversalTime().ToString('o'); python=$python; argv=$serverArgs; config=$config; run=$run; role='owned zero-logits diagnostic strictRAM20'; inherited_experiment_names_cleared=$removedNames}
$encoding = [System.Text.UTF8Encoding]::new($false)
[System.IO.File]::WriteAllText((Join-Path $run 'process.json'),($process | ConvertTo-Json -Depth 5)+"`n",$encoding)
$provenancePath = Join-Path $run 'provenance.json'
$provenance = Get-Content -LiteralPath $provenancePath -Raw | ConvertFrom-Json
$binding = [ordered]@{receipt='admission.json'; receipt_sha256=(Get-FileHash -LiteralPath (Join-Path $run 'admission.json')).Hash.ToLowerInvariant(); status='pass'; launch_process_receipt='process.json'; process_receipt_sha256=(Get-FileHash -LiteralPath (Join-Path $run 'process.json')).Hash.ToLowerInvariant(); scope='launched immediately after fresh pass and commit check; exact saved JSON bytes'}
$provenance | Add-Member -NotePropertyName launch_admission -NotePropertyValue $binding -Force
[System.IO.File]::WriteAllText($provenancePath,($provenance | ConvertTo-Json -Depth 12)+"`n",$encoding)
$samplerArgs = @('-u','-X','utf8','-B',(Join-Path $repo 'tools\hetero_resources.py'),'--output',(Join-Path $run 'resource\samples.jsonl'),'--pid',[string]$server.Id,'--interval','1','--max-seconds','10800','--minimum-commit-gib','4','--stop-file',(Join-Path $run 'resource\STOP'))
$sampler = Start-Process -FilePath $python -ArgumentList $samplerArgs -WorkingDirectory $repo -WindowStyle Hidden -PassThru -RedirectStandardOutput (Join-Path $run 'logs\sampler.stdout.log') -RedirectStandardError (Join-Path $run 'logs\sampler.stderr.log')
$observer = [ordered]@{sampler_launcher_pid=$sampler.Id; start_utc=$sampler.StartTime.ToUniversalTime().ToString('o'); python=$python; argv=$samplerArgs}
[System.IO.File]::WriteAllText((Join-Path $run 'sampler-process.json'),($observer | ConvertTo-Json -Depth 5)+"`n",$encoding)
[ordered]@{run=$run; launcher_pid=$server.Id; sampler_launcher_pid=$sampler.Id; status='launched_not_yet_ready'} | ConvertTo-Json -Compress
