$ErrorActionPreference='Stop'
$run='C:\Users\DC\Documents\ChatGPT\Strata-Hetero\bench\hetero\20261009-40-upstream41-direct-quality'
$launch=Join-Path $run 'launch-owned.ps1'
$watch=Join-Path $run 'startup-watch.ps1'
& $launch -Start
$processPath=Join-Path $run 'process.json'
if(-not(Test-Path -LiteralPath $processPath -PathType Leaf)){throw 'launch returned without the owned process receipt'}
$receipt=Get-Content -LiteralPath $processPath -Raw|ConvertFrom-Json -DateKind String
if($receipt.run_id -ne '20261009-40-upstream41-direct-quality' -or $receipt.launcher_pid -le 0 -or $receipt.actual_server_child_pid -le 0 -or $receipt.engine_path -ne (Get-Content -LiteralPath (Join-Path $run 'config.json') -Raw|ConvertFrom-Json -DateKind String).exe){throw 'run40 launch receipt does not bind exact expected launcher/server/configured engine'}
& $watch -Monitor -TimeoutSeconds 900