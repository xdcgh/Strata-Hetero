$ErrorActionPreference = 'Stop'
Set-StrictMode -Version 2.0
$repo = 'C:\Users\DC\Documents\ChatGPT\Strata-Hetero'
. (Join-Path $repo 'tools\hetero_jsonl_tail.ps1')
$tempRoot = [IO.Path]::GetFullPath([IO.Path]::GetTempPath()).TrimEnd([IO.Path]::DirectorySeparatorChar, [IO.Path]::AltDirectorySeparatorChar)
$tempLeaf = 'strata-jsonl-tail-' + [guid]::NewGuid().ToString('N')
$temp = [IO.Path]::GetFullPath((Join-Path $tempRoot $tempLeaf))
$tempPrefix = $tempRoot + [IO.Path]::DirectorySeparatorChar
if (-not $temp.StartsWith($tempPrefix, [StringComparison]::OrdinalIgnoreCase) -or
    [IO.Path]::GetFileName($temp) -notmatch '^strata-jsonl-tail-[0-9a-f]{32}$') {
    throw 'Refusing to create fixture files outside its exact GUID-named temp directory.'
}
[IO.Directory]::CreateDirectory($temp) | Out-Null
$evidence = Join-Path $temp 'partials'
[IO.Directory]::CreateDirectory($evidence) | Out-Null
$results = [Collections.Generic.List[string]]::new()
function Assert-Throws([scriptblock]$Action, [string]$Label) {
    $threw = $false
    try { & $Action } catch { $threw = $true }
    if (-not $threw) { throw "Expected rejection: $Label" }
    $results.Add($Label)
}
function Write-Utf8([string]$Path, [string]$Text) {
    [IO.File]::WriteAllText($Path, $Text, [Text.UTF8Encoding]::new($false))
}
try {
    $now = [DateTimeOffset]::UtcNow.ToString('o')
    $row1 = (@{record_type='resource_sample';observed_at_utc=$now;owner_pid=321;sequence=1} | ConvertTo-Json -Compress)
    $row2 = (@{record_type='resource_sample';observed_at_utc=$now;owner_pid=321;sequence=2} | ConvertTo-Json -Compress)

    # Multiple records + CRLF + incomplete trailing bytes: return only the newest complete row,
    # preserve the exact incomplete tail in a new content-addressed evidence file.
    $path = Join-Path $temp 'multiple-crlf.jsonl'
    $partial = '{"record_type":"resource_sample","observed_at_utc":"partial'
    Write-Utf8 $path ($row1 + "`r`n" + $row2 + "`r`n" + $partial)
    $got = Read-LastCompleteJsonlRecord -Path $path -MaxRecordBytes 4096 -MaxTailBytes 8192 `
        -TimestampField observed_at_utc -MaxAgeSeconds 10 -OwnerField owner_pid -ExpectedOwnerPid 321 `
        -RecordTypeField record_type -ExpectedRecordType resource_sample -PartialEvidenceDirectory $evidence
    if ($got.record.sequence -ne 2 -or $got.trailing_partial_bytes -ne [Text.Encoding]::UTF8.GetByteCount($partial) -or
        -not (Test-Path -LiteralPath $got.trailing_partial_path) -or
        [IO.File]::ReadAllText($got.trailing_partial_path, [Text.Encoding]::UTF8) -cne $partial) {
        throw 'multiple/CRLF/partial-tail selection or preservation failed'
    }
    $results.Add('multiple records with CRLF select newest complete line and preserve trailing partial bytes')

    # A complete single LF record is accepted.
    $path = Join-Path $temp 'single-lf.jsonl'
    Write-Utf8 $path ($row1 + "`n")
    $got = Read-LastCompleteJsonlRecord -Path $path -MaxRecordBytes 4096 -MaxTailBytes 8192 `
        -TimestampField observed_at_utc -MaxAgeSeconds 10 -OwnerField owner_pid -ExpectedOwnerPid 321 `
        -RecordTypeField record_type -ExpectedRecordType resource_sample
    if ($got.record.sequence -ne 1 -or $got.trailing_partial_bytes -ne 0) { throw 'single LF record failed' }
    $results.Add('single newline-terminated JSON record accepted')

    # Generic bounded-reader mode leaves schema-specific fields unbound under StrictMode.
    $path = Join-Path $temp 'generic-default-fields.jsonl'
    Write-Utf8 $path ('{"value":7}' + "`n")
    $generic = Read-LastCompleteJsonlRecord -Path $path -MaxRecordBytes 4096 -MaxTailBytes 8192 `
        -TimestampField '' -OwnerField '' -RecordTypeField ''
    if ($generic.record.value -ne 7 -or $null -ne $generic.observed_at_utc -or
        $null -ne $generic.owner_pid -or $null -ne $generic.record_type -or $null -ne $generic.age_seconds) {
        throw 'generic-reader empty typed fields were not initialized safely'
    }
    $results.Add('generic JSON object accepted with timestamp, owner, and record-type fields unbound')

    # Strictly reject a bad latest complete line; do not silently fall back to the older valid row.
    $path = Join-Path $temp 'bad-latest.jsonl'
    Write-Utf8 $path ($row1 + "`n" + '{"owner_pid":321, broken}' + "`n")
    Assert-Throws { Read-LastCompleteJsonlRecord -Path $path -MaxRecordBytes 4096 -MaxTailBytes 8192 `
        -TimestampField observed_at_utc -OwnerField owner_pid -ExpectedOwnerPid 321 } 'malformed latest complete row is not skipped'

    # One physical line containing two JSON values is not one valid record.
    $path = Join-Path $temp 'two-values-one-line.jsonl'
    Write-Utf8 $path ($row1 + $row2 + "`n")
    Assert-Throws { Read-LastCompleteJsonlRecord -Path $path -MaxRecordBytes 8192 -MaxTailBytes 16384 `
        -TimestampField observed_at_utc -OwnerField owner_pid -ExpectedOwnerPid 321 } 'concatenated JSON values rejected'

    # Hard record bound, stale timestamp, and owner mismatch.
    $path = Join-Path $temp 'oversized.jsonl'
    Write-Utf8 $path ((@{record_type='resource_sample';observed_at_utc=$now;owner_pid=321;data=('x' * 500)} | ConvertTo-Json -Compress) + "`n")
    Assert-Throws { Read-LastCompleteJsonlRecord -Path $path -MaxRecordBytes 128 -MaxTailBytes 2048 `
        -TimestampField observed_at_utc -OwnerField owner_pid -ExpectedOwnerPid 321 } 'oversized record rejected'

    $stale = [DateTimeOffset]::UtcNow.AddDays(-1).ToString('o')
    $path = Join-Path $temp 'stale.jsonl'
    Write-Utf8 $path ((@{record_type='resource_sample';observed_at_utc=$stale;owner_pid=321} | ConvertTo-Json -Compress) + "`n")
    Assert-Throws { Read-LastCompleteJsonlRecord -Path $path -MaxRecordBytes 4096 -MaxTailBytes 8192 `
        -TimestampField observed_at_utc -MaxAgeSeconds 5 -OwnerField owner_pid -ExpectedOwnerPid 321 } 'stale record rejected'

    $path = Join-Path $temp 'wrong-owner.jsonl'
    Write-Utf8 $path ((@{record_type='resource_sample';observed_at_utc=$now;owner_pid=999} | ConvertTo-Json -Compress) + "`n")
    Assert-Throws { Read-LastCompleteJsonlRecord -Path $path -MaxRecordBytes 4096 -MaxTailBytes 8192 `
        -TimestampField observed_at_utc -MaxAgeSeconds 5 -OwnerField owner_pid -ExpectedOwnerPid 321 } 'owner mismatch rejected'

    # No complete newline row: preserve the current fragment, but never use a previous run/artifact.
    $path = Join-Path $temp 'partial-only.jsonl'
    Write-Utf8 $path $partial
    Assert-Throws { Read-LastCompleteJsonlRecord -Path $path -MaxRecordBytes 4096 -MaxTailBytes 8192 `
        -TimestampField observed_at_utc -PartialEvidenceDirectory $evidence } 'partial-only file cannot fall back to an old record'
    $partialHash = [Convert]::ToHexString([Security.Cryptography.SHA256]::HashData([Text.Encoding]::UTF8.GetBytes($partial))).ToLowerInvariant()
    $partialEvidence = Join-Path $evidence ('partial-only.jsonl.partial.' + $partialHash + '.bin')
    if (-not (Test-Path -LiteralPath $partialEvidence) -or
        [IO.File]::ReadAllText($partialEvidence, [Text.Encoding]::UTF8) -cne $partial) {
        throw 'partial-only fragment was not preserved byte-for-byte'
    }
    $results.Add('partial-only fragment preserved while stale prior records remain ineligible')

    [ordered]@{status='passed';fixture_count=$results.Count;cases=@($results);hardware_or_model_calls=0;network_calls=0} |
        ConvertTo-Json -Depth 8 -Compress
} finally {
    $cleanupTarget = [IO.Path]::GetFullPath($temp)
    $cleanupRoot = [IO.Path]::GetFullPath($tempRoot).TrimEnd([IO.Path]::DirectorySeparatorChar, [IO.Path]::AltDirectorySeparatorChar)
    $cleanupPrefix = $cleanupRoot + [IO.Path]::DirectorySeparatorChar
    $cleanupItem = Get-Item -LiteralPath $cleanupTarget -Force -ErrorAction SilentlyContinue
    if (-not $cleanupTarget.StartsWith($cleanupPrefix, [StringComparison]::OrdinalIgnoreCase) -or
        [IO.Path]::GetFileName($cleanupTarget) -notmatch '^strata-jsonl-tail-[0-9a-f]{32}$' -or
        -not $cleanupItem -or -not $cleanupItem.PSIsContainer -or
        ($cleanupItem.Attributes -band [IO.FileAttributes]::ReparsePoint)) {
        throw 'Refusing recursive cleanup because the owned GUID temp directory failed path checks.'
    }
    Remove-Item -LiteralPath $cleanupTarget -Recurse -Force
}
