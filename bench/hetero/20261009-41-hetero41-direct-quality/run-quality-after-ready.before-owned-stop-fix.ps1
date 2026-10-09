#Requires -Version 7.0
param(
    [switch]$Run,
    [switch]$CheckReady,
    [int]$LauncherPid,
    [int]$SamplerPid,
    [switch]$RootReadyConfirmed
)

$ErrorActionPreference = 'Stop'
Set-StrictMode -Version 2.0
if ($Run -and $CheckReady) { throw 'Use either -CheckReady or -Run.' }
$repo = 'C:\Users\DC\Documents\ChatGPT\Strata-Hetero'
$runDir = Join-Path $repo 'bench\hetero\20261009-41-hetero41-direct-quality'
$configPath = Join-Path $runDir 'config.json'
$identityPath = Join-Path $runDir 'identity.json'
$provenancePath = Join-Path $runDir 'provenance.json'
$manifestPath = 'C:\Users\DC\Documents\ChatGPT\Strata-Hetero\bench\hetero\20261009-06-quality-prompts\manifest.json'
$serverRoot = 'E:\Strata-Hetero-data\source\hetero-0-1-41'
$serverScript = Join-Path $serverRoot 'serve\server.py'
$baselinePath = Join-Path $repo 'bench\hetero\20261009-04-hetero04-direct-quality\shutdown-final.json'
$python = 'E:\Strata-Hetero-data\venv\Scripts\python.exe'
$stdlibPython = 'C:\Python314\python.exe'
$qualityTool = Join-Path $repo 'tools\hetero_quality.py'
$benchTool = Join-Path $repo 'tools\hetero_bench.py'
$port = 8081
$expectedModel = 'qwen3.8-flash-next-unsloth-ud-q4_k_xl-hetero'
$expectedIds = @('smoke_arithmetic','smoke_unicode','smoke_json_arithmetic','smoke_python_function',
                'smoke_three_key_retrieval','long_1k','long_4k','long_16k','long_30k7')
$diagLog = Join-Path $runDir 'logs\engine.log'

function Get-Sha256([string]$Path) { (Get-FileHash -LiteralPath $Path -Algorithm SHA256).Hash.ToLowerInvariant() }

function Write-ExclusiveJson([string]$Path, $Value, [int]$Depth=10) {
    $bytes = [Text.UTF8Encoding]::new($false).GetBytes(($Value | ConvertTo-Json -Depth $Depth) + "`n")
    $out = [IO.File]::Open($Path,[IO.FileMode]::CreateNew,[IO.FileAccess]::Write,[IO.FileShare]::Read)
    try { $out.Write($bytes,0,$bytes.Length); $out.Flush($true) } finally { $out.Dispose() }
}

function Test-QualityContract {
    foreach ($p in @($configPath,$identityPath,$provenancePath,$manifestPath,$baselinePath,$python,$stdlibPython,$qualityTool,$benchTool)) {
        if (-not (Test-Path -LiteralPath $p -PathType Leaf)) { throw "required quality input missing: $p" }
    }
    $config = Get-Content -LiteralPath $configPath -Raw | ConvertFrom-Json
    $identity = Get-Content -LiteralPath $identityPath -Raw | ConvertFrom-Json
    $provenance = Get-Content -LiteralPath $provenancePath -Raw | ConvertFrom-Json
    $baseline = Get-Content -LiteralPath $baselinePath -Raw | ConvertFrom-Json
    $manifestHash = Get-Sha256 $manifestPath
    if ($provenance.run_id -ne '20261009-41-hetero41-direct-quality' -or $provenance.shared_controls.ple_io -ne 'direct' -or
        $provenance.placement_contrast.path -notlike 'F:\Strata-data\models\*' -or
        $provenance.acceptance_status -ne 'prepared_not_launched') { throw 'prepared original-F direct-ple run identity/status changed' }
    if ($manifestHash -ne $provenance.shared_controls.quality_prompt_manifest_sha256) { throw 'prompt manifest SHA differs from prepared run11 provenance' }
    if ($config.exe -ne $identity.engine.path -or $provenance.engine_path -ne $identity.engine.path -or
        $identity.engine.sha256 -ne $provenance.engine_sha256 -or
        (Get-Sha256 $config.exe) -ne $provenance.engine_sha256) { throw 'engine binary/config/provenance identity mismatch' }
    if ((Get-Sha256 $serverScript) -ne $provenance.server_file_sha256) {
        throw 'server bridge changed from the reviewed byte-identical e021... build'
    }
    $manifestCheck = @'
import hashlib,json,pathlib,sys
root=pathlib.Path(sys.argv[1]); manifest_path=pathlib.Path(sys.argv[2]); expected_ids=json.loads(__import__("base64").b64decode(sys.argv[3]))
sys.path.insert(0,str(root/"tools"))
from hetero_quality import load_manifest
from hetero_compare_quality import verify_tokenizer_manifest
m,by_id=load_manifest(manifest_path)
if list(x["id"] for x in m.get("prompts",[]))!=expected_ids: raise SystemExit("prompt IDs/order differ from fixed nine-task matrix")
caps={x:(256 if x=="smoke_python_function" else 128) for x in expected_ids}
checks=[]
for p in m["prompts"]:
    path=(manifest_path.parent/p["file"]).resolve(strict=True); data=path.read_bytes()
    if len(data)!=p["bytes"] or hashlib.sha256(data).hexdigest()!=p["sha256"]: raise SystemExit("prompt bytes/SHA mismatch: "+p["id"])
    if p.get("chat_prompt_tokens_local") is None or p.get("max_output_tokens",0)<1: raise SystemExit("prompt counts missing: "+p["id"])
    checks.append({"id":p["id"],"body_tokens":p["body_tokens_local"],"chat_tokens":p["chat_prompt_tokens_local"],"max_tokens":caps[p["id"]],"sha256":p["sha256"]})
tok=verify_tokenizer_manifest(m)
if tok.get("status")!="pass": raise SystemExit("tokenizer asset/source verification failed: "+json.dumps(tok))
print(json.dumps({"status":"pass","prompt_count":len(checks),"prompts":checks,"tokenizer":tok},separators=(",",":")))
'@
    $idsJson = ConvertTo-Json -InputObject $expectedIds -Compress
    $encodedIds = [Convert]::ToBase64String([Text.Encoding]::UTF8.GetBytes($idsJson))
    $encodedCheck = [Convert]::ToBase64String([Text.Encoding]::UTF8.GetBytes($manifestCheck))
    $wrapper = "exec(__import__('base64').b64decode('$encodedCheck'))"
    $pythonArgs = @('-I','-B','-X','utf8','-c',$wrapper,$repo,$manifestPath,$encodedIds)
    $raw = & $stdlibPython @pythonArgs
    if ($LASTEXITCODE -ne 0) { throw 'Pure nine-prompt/asset validation failed' }
    $manifestInfo = (($raw -join "`n") | ConvertFrom-Json)
    if ($manifestInfo.status -ne 'pass' -or $manifestInfo.prompt_count -ne 9) { throw 'manifest validation result was incomplete' }
    $base = $baseline.quality.prompt_results
    foreach ($p in $manifestInfo.prompts) {
        $old = $base.PSObject.Properties[$p.id].Value
        if (-not $old -or $old.local_chat_prompt_tokens -ne $p.chat_tokens -or $old.max_tokens -ne $p.max_tokens -or
            $old.warmups -ne 1 -or $old.formal_requests -ne 3 -or @($old.quality_statuses | Where-Object { $_ -ne 'pass' }).Count -gt 0) {
            throw "Fixed prompt/cap/warm/formal contract does not exactly match H4: $($p.id)"
        }
        $h4PromptDir = Join-Path (Join-Path $repo 'bench\hetero\20261009-04-hetero04-direct-quality\requests') $p.id
        for ($i=0; $i -lt 3; $i++) {
            if (-not (Test-Path -LiteralPath (Join-Path $h4PromptDir ("formal-{0:D4}.sse.txt" -f $i)) -PathType Leaf)) {
                throw "H4 actual-token-ID reference is missing for $($p.id) formal $i"
            }
        }
    }
    if ($identity.config.sha256 -ne (Get-Sha256 $configPath) -or $identity.config.prompt_cache -ne 0 -or
        $identity.config.resident_budget_gib -ne 20 -or $identity.config.capture_actual_engine_token_ids -ne $true) {
        throw 'prepared F identity/config controls changed'
    }
    if ($config.env.STRATA_PREFILL_CPU_SHARE -ne '0' -or $config.env.STRATA_STAGE_PIN -ne '0') { throw 'v0.1.41 CPU-share/stage-pin controls must remain explicitly disabled' }
    if ($config.parallel -ne 1 -or $config.sampling.temperature -ne 0 -or $config.sampling.top_p -ne 1 -or $config.sampling.seed -ne 42) {
        throw 'deterministic request controls changed'
    }
    $pleIndex = [array]::IndexOf($config.args,'--ple-gguf')
    $nativePaths = @(); for ($i=0; $i -lt $config.args.Count; $i++) { if ($config.args[$i] -eq '--native-dense-gguf') { $nativePaths += $config.args[$i+1] } }
    if ($pleIndex -lt 0 -or $config.args[$pleIndex+1] -ne $identity.config.ple_gguf_path -or
        $config.args[$pleIndex+1] -notlike 'F:\Strata-data\models\*' -or $nativePaths.Count -ne 4 -or
        @($nativePaths | Where-Object { $_ -like 'E:\*' -or $_ -notlike 'F:\Strata-data\models\*' }).Count -gt 0 -or
        $config.args[([array]::IndexOf($config.args,'--ple-io')+1)] -ne 'direct') { throw 'quality F arm does not retain the explicit original F-only placement config' }
    $fullManifest = Get-Content -LiteralPath $manifestPath -Raw | ConvertFrom-Json
    return [pscustomobject]@{manifest=$fullManifest;identity=$identity;provenance=$provenance;config=$config;
        manifest_sha256=$manifestHash;config_sha256=(Get-Sha256 $configPath);
        engine_sha256=$provenance.engine_sha256;server_file_sha256=$provenance.server_file_sha256}
}

function Get-LatestResourceGate([string]$When) {
    $path = Join-Path $runDir 'resource\samples.jsonl'
    if (-not (Test-Path -LiteralPath $path -PathType Leaf)) { throw "${When}: sampler output is absent" }
    $line = Get-Content -LiteralPath $path -Tail 1
    if (-not $line) { throw "${When}: no resource sample exists" }
    $doc = [System.Text.Json.JsonDocument]::Parse([string]$line)
    $at = [DateTimeOffset]::Parse($doc.RootElement.GetProperty('observed_at_utc').GetString())
    $sample = $line | ConvertFrom-Json
    $age = ([DateTimeOffset]::UtcNow - $at).TotalSeconds
    if ($sample.owner_pid -ne $LauncherPid -or $age -lt -2 -or $age -gt 5 -or
        $sample.ram_gate.status -ne 'pass' -or $null -eq $sample.ram_gate.available_gib -or [double]$sample.ram_gate.available_gib -lt 12 -or
        $sample.commit_gate.status -ne 'pass' -or $null -eq $sample.commit_gate.available_gib -or [double]$sample.commit_gate.available_gib -lt 4 -or
        @($sample.alerts).Count -gt 0 -or @($sample.errors).Count -gt 0) { throw "${When}: resource sample stale/unknown/blocked; no further prompt will be sent" }
    return $sample
}

function Get-DiagnosticSlice([long]$StartOffset) {
    if (-not (Test-Path -LiteralPath $diagLog)) { return [pscustomobject]@{end_offset=0;lines=@()} }
    $stream=[IO.File]::Open($diagLog,[IO.FileMode]::Open,[IO.FileAccess]::Read,[IO.FileShare]::ReadWrite)
    try {
        $end=[long]$stream.Length; [void]$stream.Seek([Math]::Min($StartOffset,$end),[IO.SeekOrigin]::Begin)
        $reader=[IO.StreamReader]::new($stream,[Text.Encoding]::UTF8,$true,4096,$true)
        $text=$reader.ReadToEnd();$reader.Dispose()
        return [pscustomobject]@{end_offset=$end;lines=@([regex]::Split($text,"\r?\n")|Where-Object{$_.Contains('strata zero-logits diag:')})}
    } finally { $stream.Dispose() }
}

function Get-ReadyBindings([string]$When, [int]$AllowedInFlight) {
    $proc = Get-CimInstance Win32_Process -Filter "ProcessId = $LauncherPid" -ErrorAction SilentlyContinue
    $procReceiptText = Get-Content -LiteralPath (Join-Path $runDir 'process.json') -Raw
    $procReceipt = $procReceiptText | ConvertFrom-Json
    $procReceiptDoc = [System.Text.Json.JsonDocument]::Parse($procReceiptText)
    $launcherCreateUtc = $procReceiptDoc.RootElement.GetProperty('launcher_create_utc').GetString()
    $serverCreateUtc = $procReceiptDoc.RootElement.GetProperty('actual_server_child_create_utc').GetString()
    if (-not $proc -or $proc.ExecutablePath -ne $procReceipt.python -or
        $proc.CreationDate.ToUniversalTime().ToString('o') -ne $launcherCreateUtc -or
        -not $proc.CommandLine.Contains('serve\server.py') -or -not $proc.CommandLine.Contains($configPath)) {
        throw "${When}: server launcher PID/executable/command is not the owned run11 process"
    }
    $server = Get-CimInstance Win32_Process | Where-Object {
        $_.ParentProcessId -eq $LauncherPid -and $_.Name -eq 'python.exe' -and $_.CommandLine.Contains($configPath)
    } | Select-Object -First 1
    if (-not $server -or $server.ProcessId -ne $procReceipt.actual_server_child_pid -or
        $server.ExecutablePath -ne $procReceipt.actual_server_child_exe -or
        $server.CreationDate.ToUniversalTime().ToString('o') -ne $serverCreateUtc) {
        throw "${When}: exact C Python server child PID/executable/CreateTime is absent"
    }
    $engine = Get-CimInstance Win32_Process | Where-Object {
        $_.ParentProcessId -eq $server.ProcessId -and $_.ExecutablePath -eq $identity.engine.path
    } | Select-Object -First 1
    if (-not $engine) { throw "${When}: exact Hetero engine child is absent" }
    $config = Get-Content -LiteralPath $configPath -Raw | ConvertFrom-Json
    $expectedEngineCommand = $identity.engine.path + ' --serve ' + ($config.args -join ' ')
    if ($engine.CommandLine -ne $expectedEngineCommand) { throw "${When}: engine child command differs from prepared config" }
    $listeners = @(Get-NetTCPConnection -LocalPort $port -State Listen -ErrorAction SilentlyContinue)
    if ($listeners.Count -ne 1 -or $listeners[0].LocalAddress -ne '127.0.0.1' -or $listeners[0].OwningProcess -ne $server.ProcessId) {
        throw "${When}: 8081 listener is not uniquely owned by the loopback server child"
    }
    $status = Invoke-RestMethod -Uri "http://127.0.0.1:$port/v1/status" -Method Get -TimeoutSec 5
    if ($status.model -ne $expectedModel -or $status.engine -ne '0.1.41' -or $status.loaded -ne $true -or
        $status.activity.in_flight -gt $AllowedInFlight) { throw "${When}: live service identity/loaded/in-flight state is invalid" }
    return [pscustomobject]@{launcher=$proc;server=$server;engine=$engine;status=$status}
}

function Stop-OwnedRunner([int]$LauncherProcessId, [string]$LauncherCreateUtc, [string]$LauncherExe,
                          [int]$ActualProcessId, [string]$ActualCreateUtc, [string]$RunId) {
    $actual = Get-CimInstance Win32_Process -Filter "ProcessId = $ActualProcessId" -ErrorAction SilentlyContinue
    if ($actual) {
        if ($actual.ExecutablePath -ne 'C:\Python314\python.exe' -or
            ($ActualProcessId -ne $LauncherProcessId -and $actual.ParentProcessId -ne $LauncherProcessId) -or
            $actual.CreationDate.ToUniversalTime().ToString('o') -ne $ActualCreateUtc -or
            -not $actual.CommandLine.Contains('tools\hetero_bench.py') -or -not $actual.CommandLine.Contains($RunId)) {
            throw 'Runner actual-child PID/executable/parent/CreateTime changed; refusing to stop an unknown process'
        }
        Stop-Process -Id $ActualProcessId -ErrorAction Stop
    }
    $launcher = Get-CimInstance Win32_Process -Filter "ProcessId = $LauncherProcessId" -ErrorAction SilentlyContinue
    if ($launcher -and $ActualProcessId -ne $LauncherProcessId) {
        if ($launcher.ExecutablePath -ne $LauncherExe -or $launcher.CreationDate.ToUniversalTime().ToString('o') -ne $LauncherCreateUtc -or
            -not $launcher.CommandLine.Contains('tools\hetero_bench.py')) { throw 'Runner launcher identity changed; refusing to stop an unknown process' }
        Stop-Process -Id $LauncherProcessId -ErrorAction Stop
    }
    if ($LauncherProcessId -ne $ActualProcessId) { Wait-Process -Id $ActualProcessId -Timeout 10 -ErrorAction SilentlyContinue }
    Wait-Process -Id $LauncherProcessId -Timeout 10 -ErrorAction SilentlyContinue
}

function Test-RunReady {
    if (-not $RootReadyConfirmed) { throw 'Only run after root explicitly confirms run11 is ready.' }
    if ($LauncherPid -le 0 -or $SamplerPid -le 0) { throw 'valid run-owned launcher/sampler PIDs are required' }
    $launch = Get-Content -LiteralPath (Join-Path $runDir 'process.json') -Raw | ConvertFrom-Json
    $sampler = Get-Content -LiteralPath (Join-Path $runDir 'sampler-process.json') -Raw | ConvertFrom-Json
    $admission = Get-Content -LiteralPath (Join-Path $runDir 'admission.json') -Raw | ConvertFrom-Json
    if ($launch.launcher_pid -ne $LauncherPid -or $launch.actual_server_child_pid -le 0 -or
        $sampler.sampler_launcher_pid -ne $SamplerPid -or $sampler.actual_child_pid -le 0 -or
        $sampler.actual_child_exe -ne 'C:\Python314\python.exe' -or $sampler.actual_child_parent_pid -ne $SamplerPid) {
        throw 'run process/sampler receipts do not bind supplied wrapper and C Python child PIDs'
    }
    $samplerReceiptText = Get-Content -LiteralPath (Join-Path $runDir 'sampler-process.json') -Raw
    $samplerDoc = [System.Text.Json.JsonDocument]::Parse($samplerReceiptText)
    $samplerCreatedUtc = $samplerDoc.RootElement.GetProperty('actual_child_create_utc').GetString()
    $samplerLive = Get-CimInstance Win32_Process -Filter "ProcessId = $($sampler.actual_child_pid)" -ErrorAction SilentlyContinue
    if (-not $samplerLive -or $samplerLive.ExecutablePath -ne 'C:\Python314\python.exe' -or
        $samplerLive.ParentProcessId -ne $SamplerPid -or $samplerLive.CreationDate.ToUniversalTime().ToString('o') -ne $samplerCreatedUtc -or
        -not $samplerLive.CommandLine.Contains('hetero_resources.py')) { throw 'sampler actual child PID/executable/parent/CreateTime is not live and owned' }
    if ($admission.pass -ne $true -or $admission.status -ne 'pass' -or $admission.chosen_gates.vram.available_mib -lt 46000 -or
        $admission.raw.windows_memory.available_commit_bytes -lt 4GB) { throw 'fresh launch admission no longer proves 12/4/46k gate' }
    if ((Get-Sha256 (Join-Path $runDir 'identity.json')) -ne (Get-Sha256 $identityPath) -or
        (Get-Sha256 (Join-Path $runDir 'provenance.json')) -ne (Get-Sha256 $provenancePath)) { throw 'prepared identity/provenance was modified' }
}

$contract = Test-QualityContract
$identity = $contract.identity
$provenance = $contract.provenance
$config = $contract.config
if (-not $Run -and -not $CheckReady) {
    [ordered]@{status='prepared_quality_validated';run_started=$false;prompt_count=9;formal_per_prompt=3;warmups_per_prompt=1;
        caps='128 for eight non-code tasks, 256 for smoke_python_function';manifest_sha256=$contract.manifest_sha256;
        expected_h4_configured_status='passed'} | ConvertTo-Json -Compress
    exit 0
}
Test-RunReady
$samplerReceiptText = Get-Content -LiteralPath (Join-Path $runDir 'sampler-process.json') -Raw
$samplerReceiptDoc = [System.Text.Json.JsonDocument]::Parse($samplerReceiptText)
$qualitySamplerCreateUtc = $samplerReceiptDoc.RootElement.GetProperty('actual_child_create_utc').GetString()
$ready = Get-ReadyBindings 'quality entry' 0
if ($CheckReady) {
    $sample = Get-LatestResourceGate 'read-only quality preflight'
    if ($ready.status.activity.requests -ne 0) { throw 'Read-only preflight requires zero model requests.' }
    [ordered]@{status='live_quality_preflight_pass';chat_requests_sent=0;prompt_count=$contract.manifest.prompts.Count;
        model_requests=$ready.status.activity.requests;engine_pid=$ready.engine.ProcessId;
        physical_gib=$sample.ram_gate.available_gib;commit_gib=$sample.commit_gate.available_gib;
        manifest_has_prompt_paths=@($contract.manifest.prompts | Where-Object { -not $_.file }).Count -eq 0;
        identity_bound=$identity.engine.path -eq $ready.engine.ExecutablePath} | ConvertTo-Json -Compress
    exit 0
}
$readyBindingPath = Join-Path $runDir 'quality\ready-binding.json'
if (Test-Path -LiteralPath $readyBindingPath) { throw 'quality ready binding exists; refusing a second run' }
    $readyBinding = [ordered]@{run_id='20261009-41-hetero41-direct-quality';checked_utc=[DateTime]::UtcNow.ToString('o');
    launcher_pid=$LauncherPid;server_pid=$ready.server.ProcessId;engine_pid=$ready.engine.ProcessId;engine_exe=$ready.engine.ExecutablePath;
    server_child_exe=$ready.server.ExecutablePath;server_child_create_utc=$ready.server.CreationDate.ToUniversalTime().ToString('o');
    engine_child_create_utc=$ready.engine.CreationDate.ToUniversalTime().ToString('o');
    sampler_actual_child_pid=(Get-Content -LiteralPath (Join-Path $runDir 'sampler-process.json') -Raw | ConvertFrom-Json).actual_child_pid;
    sampler_actual_child_create_utc=$qualitySamplerCreateUtc;
    engine_sha256=(Get-Sha256 $identity.engine.path);server_file_sha256=(Get-Sha256 $serverScript);
    model=$ready.status.model;engine_version=$ready.status.engine;loaded=$ready.status.loaded}
Write-ExclusiveJson $readyBindingPath $readyBinding
$progressPath = Join-Path $runDir 'quality\quality-progress.jsonl'
if (Test-Path -LiteralPath $progressPath) { throw 'quality progress already exists; refusing overwrite' }
$progress = [IO.File]::Open($progressPath,[IO.FileMode]::CreateNew,[IO.FileAccess]::Write,[IO.FileShare]::Read)
function Write-Progress($Event) {
    $line = (ConvertTo-Json -InputObject ([ordered]@{utc=[DateTime]::UtcNow.ToString('o');event=$Event}) -Compress) + "`n"
    $bytes=[Text.UTF8Encoding]::new($false).GetBytes($line);$progress.Write($bytes,0,$bytes.Length);$progress.Flush($true)
}
$promptResults = @(); $diagOffset = (Get-Item -LiteralPath $diagLog).Length
try {
    $null = Get-LatestResourceGate 'quality start'; Write-Progress @{phase='start';status='pass'}
    foreach ($prompt in $contract.manifest.prompts) {
        $id = $prompt.id; $maxTokens = if ($id -eq 'smoke_python_function') { 256 } else { 128 }
        $promptPath = Join-Path (Split-Path -Parent $manifestPath) $prompt.file
        $requestDir = Join-Path (Join-Path $runDir 'requests') $id
        $reportPath = Join-Path (Join-Path $runDir 'quality') "F-$id-quality.json"
        $stdoutPath = Join-Path (Join-Path $runDir 'logs') "quality-F-$id.stdout.log"
        $stderrPath = Join-Path (Join-Path $runDir 'logs') "quality-F-$id.stderr.log"
        foreach ($target in @($requestDir,$reportPath,$stdoutPath,$stderrPath)) { if (Test-Path -LiteralPath $target) { throw "${id}: output already exists: $target" } }
        $null = Get-ReadyBindings "before $id" 0; $null = Get-LatestResourceGate "before $id"
        $beforeDiag = Get-DiagnosticSlice $diagOffset; $diagOffset = $beforeDiag.end_offset
        if (@($beforeDiag.lines).Count -gt 0) { throw "${id}: zero-logits diagnostic arrived before prompt; no request sent" }
        $runnerArgs = @((Join-Path $repo 'tools\hetero_bench.py'),'--prompt-file',$promptPath,'--output',(Join-Path $runDir 'requests'),
            '--run-id',$id,'--repeats','3','--warmup','1','--max-tokens',[string]$maxTokens,'--concurrency','1','--seed','42',
            '--reasoning-effort','none','--identity-json',$identityPath,'--base-url',"http://127.0.0.1:$port")
        $runner = Start-Process -FilePath $python -ArgumentList $runnerArgs -WorkingDirectory $repo -WindowStyle Hidden -PassThru `
            -RedirectStandardOutput $stdoutPath -RedirectStandardError $stderrPath
        $runnerLauncherCim = Get-CimInstance Win32_Process -Filter "ProcessId = $($runner.Id)" -ErrorAction SilentlyContinue
        if (-not $runnerLauncherCim) { throw "${id}: benchmark launcher PID disappeared before identity capture" }
        $runnerChildCim = $null
        if ($runnerLauncherCim.ExecutablePath -eq 'C:\Python314\python.exe' -and
            $runnerLauncherCim.CommandLine.Contains('tools\hetero_bench.py')) {
            $runnerChildCim = $runnerLauncherCim
        } else {
            for ($j=0; $j -lt 50 -and -not $runnerChildCim; $j++) {
                $runnerChildCim = Get-CimInstance Win32_Process | Where-Object {
                    $_.ParentProcessId -eq $runner.Id -and $_.ExecutablePath -eq 'C:\Python314\python.exe' -and
                    $_.CommandLine.Contains('tools\hetero_bench.py') -and $_.CommandLine.Contains($id)
                } | Select-Object -First 1
                if (-not $runnerChildCim) { Start-Sleep -Milliseconds 100 }
            }
        }
        if (-not $runnerChildCim) { throw "${id}: C Python benchmark child/CreateTime was not found under the venv launcher" }
        $runnerLauncherCreateUtc = $runnerLauncherCim.CreationDate.ToUniversalTime().ToString('o')
        $runnerCreated = $runnerChildCim.CreationDate.ToUniversalTime().ToString('o')
        Write-Progress @{phase=$id;event='runner_started';launcher_pid=$runner.Id;launcher_exe=$runnerLauncherCim.ExecutablePath;
            launcher_create_utc=$runnerLauncherCreateUtc;actual_child_pid=$runnerChildCim.ProcessId;
            actual_child_parent_pid=$runnerChildCim.ParentProcessId;actual_child_exe=$runnerChildCim.ExecutablePath;
            actual_child_create_utc=$runnerCreated;max_tokens=$maxTokens}
        try {
            while ($true) {
                $runner.Refresh()
                $actualLive = Get-CimInstance Win32_Process -Filter "ProcessId = $($runnerChildCim.ProcessId)" -ErrorAction SilentlyContinue
                if ($runner.HasExited -and -not $actualLive) { break }
                if (-not $runner.HasExited -and -not $actualLive) {
                    if (-not $runner.WaitForExit(5000)) { throw "${id}: benchmark child exited but its owned launcher did not finish within 5s" }
                    break
                }
                $null = Get-LatestResourceGate "during $id"
                $null = Get-ReadyBindings "during $id" 1
                $diag = Get-DiagnosticSlice $diagOffset; $diagOffset = $diag.end_offset
                if (@($diag.lines).Count -gt 0) { Write-Progress @{phase=$id;event='zero_logits_diagnostic';lines=$diag.lines} }
                Start-Sleep -Seconds 1; $runner.Refresh()
            }
        } catch {
            Stop-OwnedRunner $runner $runnerLauncherCreateUtc $runnerLauncherCim.ExecutablePath $runnerChildCim.ProcessId $runnerCreated $id
            throw
        }
        $runnerFinalChild = Get-CimInstance Win32_Process -Filter "ProcessId = $($runnerChildCim.ProcessId)" -ErrorAction SilentlyContinue
        if ($runnerFinalChild) { throw "${id}: benchmark child PID still exists after runner exit" }
        $runnerExit = [int]$runner.ExitCode
        if (-not (Test-Path -LiteralPath $requestDir -PathType Container)) { throw "${id}: request output directory missing (runner exit $runnerExit)" }
        & $stdlibPython -I -B -X utf8 $qualityTool --responses-dir $requestDir --manifest $manifestPath --report-out $reportPath
        $qualityExit = $LASTEXITCODE
        if ($qualityExit -ne 0) { throw "${id}: offline quality report failed (exit $qualityExit)" }
        $meta = Get-Content -LiteralPath (Join-Path $requestDir 'metadata.json') -Raw | ConvertFrom-Json
        $requests = @(Get-Content -LiteralPath (Join-Path $requestDir 'requests.json') -Raw | ConvertFrom-Json)
        $warmups = @(Get-Content -LiteralPath (Join-Path $requestDir 'warmups.json') -Raw | ConvertFrom-Json)
        $report = Get-Content -LiteralPath $reportPath -Raw | ConvertFrom-Json
        $rows = @($report.results | Where-Object { $_.prompt_id -eq $id })
        $expectedPromptTokens = [int]$prompt.chat_prompt_tokens_local
        if ($runnerExit -ne 0 -or $meta.max_tokens -ne $maxTokens -or $meta.repeats -ne 3 -or $meta.warmup -ne 1 -or
            $meta.seed -ne 42 -or $meta.reasoning_effort -ne 'none' -or $requests.Count -ne 3 -or $warmups.Count -ne 1 -or
            $rows.Count -ne 3 -or @($requests | Where-Object { $_.status -ne 'completed' -or $_.finish_reason -ne 'stop' -or
                $_.usage.prompt_tokens -ne $expectedPromptTokens -or $_.timings.cache_n -ne 0 }).Count -gt 0 -or
            @($rows | Where-Object { $_.quality_status -ne 'pass' }).Count -gt 0) {
            throw "${id}: status/token-count/cache/quality gate failed; preserve outputs and stop the remaining matrix"
        }
        $rawIdScript = @'
import json,pathlib,sys
sys.path.insert(0,str(pathlib.Path(sys.argv[1])/"tools"))
from hetero_compare_quality import extract_final_sse
candidate=pathlib.Path(sys.argv[2]); h4=pathlib.Path(sys.argv[3])
rows=[extract_final_sse(candidate/f"formal-{i:04d}.sse.txt") for i in range(3)]
base=[extract_final_sse(h4/f"formal-{i:04d}.sse.txt") for i in range(3)]
valid=all(x.get("valid") and x.get("actual_generated_token_ids") and x.get("include_stop") is True for x in rows+base)
hashes=[x.get("actual_generated_token_ids_sha256") for x in rows]
base_hashes=[x.get("actual_generated_token_ids_sha256") for x in base]
matches=[rows[i].get("actual_generated_token_ids")==base[i].get("actual_generated_token_ids") for i in range(3)]
ok=valid and len(set(hashes))==1 and all(matches)
print(json.dumps({"valid":valid,"count":len(rows),"candidate_id_hashes":hashes,"h4_id_hashes":base_hashes,"matches_h4_by_repeat":matches},separators=(",",":")))
raise SystemExit(0 if ok else 1)
'@
        $baselineRequestDir = Join-Path (Join-Path $repo 'bench\hetero\20261009-04-hetero04-direct-quality\requests') $id
        $rawIdEncoded = [Convert]::ToBase64String([Text.Encoding]::UTF8.GetBytes($rawIdScript))
        $rawIdWrapper = "exec(__import__('base64').b64decode('$rawIdEncoded'))"
        $rawIdArgs = @('-I','-B','-X','utf8','-c',$rawIdWrapper,$repo,$requestDir,$baselineRequestDir)
        $idJson = & $stdlibPython @rawIdArgs
        if ($LASTEXITCODE -ne 0) { throw "${id}: actual SSE token IDs differ from H4, are invalid, or repeat inconsistently; stop matrix" }
        $ids = (($idJson -join "`n") | ConvertFrom-Json)
        $afterDiag = Get-DiagnosticSlice $diagOffset; $diagOffset = $afterDiag.end_offset
        if (@($afterDiag.lines).Count -gt 0) { throw "${id}: zero-logits diagnostic appeared during request; stop matrix" }
        $null = Get-LatestResourceGate "after $id"; $null = Get-ReadyBindings "after $id" 0
        $row = [ordered]@{prompt_id=$id;prompt_sha256=$prompt.sha256;local_chat_tokens=$expectedPromptTokens;max_tokens=$maxTokens;
            warmups=1;formal=3;runner_exit=$runnerExit;quality_statuses=@($rows|ForEach-Object{$_.quality_status});
            actual_sse_token_ids_valid=$true;actual_token_id_hashes=$ids.candidate_id_hashes;h4_token_id_hashes=$ids.h4_id_hashes;
            actual_token_ids_match_h4=$ids.matches_h4_by_repeat;formal_prompt_tokens=@($requests|ForEach-Object{$_.usage.prompt_tokens});
            cache_n=@($requests|ForEach-Object{$_.timings.cache_n});diagnostic_observed=$false}
        $promptResults += $row; Write-Progress @{phase=$id;event='prompt_pass';result=$row}
        Write-ExclusiveJson (Join-Path (Join-Path $runDir 'quality') "F-$id-result.json") $row
    }
    $final = [ordered]@{schema_version=1;run_id='20261009-41-hetero41-direct-quality';status='nine_prompt_matrix_complete';
        completed_utc=[DateTime]::UtcNow.ToString('o');prompt_count=$promptResults.Count;warmups_per_prompt=1;formal_per_prompt=3;
        caps='128 for eight tasks; 256 for smoke_python_function';engine_sha256=$identity.engine.sha256;server_file_sha256=$provenance.server_file_sha256;
        results=$promptResults;long_prompts_completed=$true}
    Write-ExclusiveJson (Join-Path (Join-Path $runDir 'quality') 'F-nine-prompt-stage.json') $final
    Write-Progress @{phase='complete';status='nine_prompt_matrix_complete';sampler_stop_requested=$false;sampler_remains_owned_for_root_cleanup=$true}
    [ordered]@{run=$runDir;status='nine_prompt_matrix_complete';prompt_count=9;formal_requests=27;warmup_requests=9;
        sampler_stop_requested=$false;sampler_remains_owned_for_root_cleanup=$true} | ConvertTo-Json -Compress
} catch {
    Write-Progress @{phase='failed';error_type=$_.Exception.GetType().Name;error=$_.Exception.Message;partial_prompts=$promptResults.Count}
    $failurePath = Join-Path (Join-Path $runDir 'quality') 'F-quality-failure.json'
    if (-not (Test-Path -LiteralPath $failurePath)) {
        Write-ExclusiveJson $failurePath ([ordered]@{schema_version=1;status='failed_stopped';phase='run-quality-after-ready';
            utc=[DateTime]::UtcNow.ToString('o');error_type=$_.Exception.GetType().Name;error=$_.Exception.Message;completed_prompts=$promptResults})
    }
    throw
} finally { $progress.Dispose() }
