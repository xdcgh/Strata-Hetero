$ErrorActionPreference = 'Stop'
$root = $PSScriptRoot
$records = @()
foreach ($item in @(@('token-ids-attempt-01',11),@('nonfinite-subset-attempt-01',4))) {
    $directory = Join-Path $root ('host-tests-01\'+$item[0])
    $run = Get-Content -LiteralPath (Join-Path $directory 'run.json') -Raw | ConvertFrom-Json
    $status = Get-Content -LiteralPath (Join-Path $directory 'terminal-status.json') -Raw | ConvertFrom-Json
    if ($run.head -cne 'ddb19224de601fdd0eaee5f011e2ef1c3e7c4feb' -or
        $status.exit_code -ne 0 -or $status.terminal_status -cne 'PASS') { throw 'Host result is not the reviewed source/pass' }
    foreach ($source in $run.hashes) {
        if ((Get-FileHash -LiteralPath $source.path -Algorithm SHA256).Hash -ine $source.sha256) {
            throw ('Source/launcher hash changed: '+$source.path)
        }
    }
    $stderr = Get-Content -LiteralPath (Join-Path $directory 'stderr.raw.txt') -Raw
    if ($stderr -notmatch ('Ran '+$item[1]+' tests in ') -or $stderr -notmatch '(?m)^OK\s*$') { throw 'Raw unittest result differs' }
    $records += [pscustomobject]@{directory=$item[0];test_count=$item[1];exit_code=$status.exit_code;
        started_utc=$run.started_utc;finished_utc=$status.finished_utc;source_and_launcher_hashes_verified=$run.hashes.Count}
}
$sourceReview = 'E:\Strata-Hetero-data\source\upstream-7dd5cf18-bridge-20261025\bench\hetero\20261025-78-current-upstream-preparation\source-review.md'
$copiedReview = Join-Path $root 'source-review.md'
if (Test-Path -LiteralPath $copiedReview) { throw 'Refusing to overwrite source review' }
Copy-Item -LiteralPath $sourceReview -Destination $copiedReview
if ((Get-FileHash -LiteralPath $sourceReview).Hash -cne (Get-FileHash -LiteralPath $copiedReview).Hash) { throw 'Review copy mismatch' }
$files = @(Get-ChildItem -LiteralPath $root -Recurse -File | ForEach-Object {
    [pscustomobject]@{path=[IO.Path]::GetRelativePath($root,$_.FullName);bytes=$_.Length;
        sha256=(Get-FileHash -LiteralPath $_.FullName -Algorithm SHA256).Hash.ToLowerInvariant()}
})
$report = [pscustomobject]@{status='retained_host_checks_rehashed';upstream_engine_base='7dd5cf18e68c639a8773e096c5c79f971ea2a9f5';
    bridge_source='ddb19224de601fdd0eaee5f011e2ef1c3e7c4feb';engine_project_version='0.1.49.2';product_tag='v0.2.4.2';
    pure_host_test_count=15;records=$records;files=$files;http_fixture_run=$false;engine_model_run=$false;
    claim_limit='Fake engine and in-memory protocol tests only. Full model, native batch transport, cross-base default-byte equivalence and performance remain unverified.'}
$target = Join-Path $root 'host-review.json'
if (Test-Path -LiteralPath $target) { throw 'Refusing to overwrite host review' }
$report | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $target -Encoding utf8
$records | ConvertTo-Json -Depth 4
