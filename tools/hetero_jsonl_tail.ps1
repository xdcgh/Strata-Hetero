function Read-LastCompleteJsonlRecord {
    [CmdletBinding()]
    param(
        [Parameter(Mandatory)][string]$Path,
        [int]$MaxRecordBytes = 65536,
        [int]$MaxTailBytes = 262144,
        [string]$TimestampField = 'observed_at_utc',
        [double]$MaxAgeSeconds = -1,
        [double]$FutureToleranceSeconds = 2,
        [string]$OwnerField = '',
        [long]$ExpectedOwnerPid = -1,
        [string]$RecordTypeField = '',
        [string]$ExpectedRecordType = '',
        [string]$PartialEvidenceDirectory = ''
    )

    if ($MaxRecordBytes -lt 1 -or $MaxTailBytes -lt $MaxRecordBytes) {
        throw 'JSONL tail bounds are invalid.'
    }
    $stream = [IO.File]::Open($Path, [IO.FileMode]::Open, [IO.FileAccess]::Read,
        [IO.FileShare]::ReadWrite -bor [IO.FileShare]::Delete)
    try {
        $fileLength = [long]$stream.Length
        if ($fileLength -eq 0) { throw "JSONL file is empty: $Path" }
        $readLength = [int][Math]::Min($fileLength, [long]$MaxTailBytes + 1)
        $tailOffset = $fileLength - $readLength
        [void]$stream.Seek($tailOffset, [IO.SeekOrigin]::Begin)
        $buffer = [byte[]]::new($readLength)
        $read = 0
        while ($read -lt $readLength) {
            $n = $stream.Read($buffer, $read, $readLength - $read)
            if ($n -le 0) { break }
            $read += $n
        }
        if ($read -ne $readLength) { throw "Short read while reading bounded JSONL tail: $Path" }
    } finally {
        $stream.Dispose()
    }

    $dataStart = 0
    if ($tailOffset -gt 0) {
        $boundary = [Array]::IndexOf($buffer, [byte]10)
        if ($boundary -lt 0) { throw "No newline boundary within $MaxTailBytes bytes of JSONL tail: $Path" }
        $dataStart = $boundary + 1
    }
    $lastLf = [Array]::LastIndexOf($buffer, [byte]10)
    if ($lastLf -lt $dataStart) {
        $fragmentLength = $readLength - $dataStart
        if ($fragmentLength -gt 0 -and $PartialEvidenceDirectory) {
            if (-not (Test-Path -LiteralPath $PartialEvidenceDirectory -PathType Container)) {
                throw "Partial evidence directory does not exist: $PartialEvidenceDirectory"
            }
            $fragment = [byte[]]::new($fragmentLength)
            [Array]::Copy($buffer, $dataStart, $fragment, 0, $fragmentLength)
            $fragmentSha = [Convert]::ToHexString([Security.Cryptography.SHA256]::HashData($fragment)).ToLowerInvariant()
            $fragmentPath = Join-Path $PartialEvidenceDirectory ([IO.Path]::GetFileName($Path) + '.partial.' + $fragmentSha + '.bin')
            if (-not (Test-Path -LiteralPath $fragmentPath -PathType Leaf)) {
                try {
                    $fragmentStream = [IO.File]::Open($fragmentPath, [IO.FileMode]::CreateNew, [IO.FileAccess]::Write, [IO.FileShare]::Read)
                    try { $fragmentStream.Write($fragment, 0, $fragment.Length); $fragmentStream.Flush($true) }
                    finally { $fragmentStream.Dispose() }
                } catch [IO.IOException] {
                    if (-not (Test-Path -LiteralPath $fragmentPath -PathType Leaf) -or
                        (Get-FileHash -LiteralPath $fragmentPath -Algorithm SHA256).Hash.ToLowerInvariant() -ne $fragmentSha) { throw }
                }
            }
        }
        throw "No complete newline-terminated JSONL record in bounded tail: $Path"
    }

    $hasTrailingPartial = ($lastLf -lt ($readLength - 1))
    $partialBytes = [byte[]]::new(0)
    $recordEnd = $lastLf
    if ($hasTrailingPartial) {
        $partialLength = $readLength - ($lastLf + 1)
        $partialBytes = [byte[]]::new($partialLength)
        [Array]::Copy($buffer, $lastLf + 1, $partialBytes, 0, $partialLength)
    }

    $partialSha = if ($partialBytes.Length) {
        [Convert]::ToHexString([Security.Cryptography.SHA256]::HashData($partialBytes)).ToLowerInvariant()
    } else { $null }
    $partialPath = $null
    if ($partialBytes.Length -and $PartialEvidenceDirectory) {
        if (-not (Test-Path -LiteralPath $PartialEvidenceDirectory -PathType Container)) {
            throw "Partial evidence directory does not exist: $PartialEvidenceDirectory"
        }
        $partialName = ([IO.Path]::GetFileName($Path) + '.partial.' + $partialSha + '.bin')
        $partialPath = Join-Path $PartialEvidenceDirectory $partialName
        if (Test-Path -LiteralPath $partialPath -PathType Leaf) {
            if ((Get-FileHash -LiteralPath $partialPath -Algorithm SHA256).Hash.ToLowerInvariant() -ne $partialSha) {
                throw "Existing partial evidence does not match its content hash: $partialPath"
            }
        } else {
            try {
                $partialStream = [IO.File]::Open($partialPath, [IO.FileMode]::CreateNew,
                    [IO.FileAccess]::Write, [IO.FileShare]::Read)
                try { $partialStream.Write($partialBytes, 0, $partialBytes.Length); $partialStream.Flush($true) }
                finally { $partialStream.Dispose() }
            } catch [IO.IOException] {
                if (-not (Test-Path -LiteralPath $partialPath -PathType Leaf) -or
                    (Get-FileHash -LiteralPath $partialPath -Algorithm SHA256).Hash.ToLowerInvariant() -ne $partialSha) { throw }
            }
        }
    }

    $previousLf = -1
    for ($i = $recordEnd - 1; $i -ge $dataStart; $i--) {
        if ($buffer[$i] -eq 10) { $previousLf = $i; break }
    }
    $recordStart = if ($previousLf -ge $dataStart) { $previousLf + 1 } else { $dataStart }
    $recordLength = $recordEnd - $recordStart
    if ($recordLength -gt 0 -and $buffer[$recordStart + $recordLength - 1] -eq 13) { $recordLength-- }
    if ($recordLength -le 0) { throw "Last complete JSONL record is empty: $Path" }
    if ($recordLength -gt $MaxRecordBytes) { throw "Last complete JSONL record exceeds $MaxRecordBytes bytes: $Path" }

    $recordBytes = [byte[]]::new($recordLength)
    [Array]::Copy($buffer, $recordStart, $recordBytes, 0, $recordLength)
    $utf8 = [Text.UTF8Encoding]::new($false, $true)
    try { $line = $utf8.GetString($recordBytes) }
    catch { throw "Last complete JSONL record is not valid UTF-8: $Path" }
    try { $document = [System.Text.Json.JsonDocument]::Parse($line) }
    catch { throw "Last complete JSONL record is not exactly one valid JSON value: $Path; $($_.Exception.Message)" }
    try {
        $root = $document.RootElement
        if ($root.ValueKind -ne [System.Text.Json.JsonValueKind]::Object) {
            throw "Last complete JSONL record must be a JSON object: $Path"
        }
        $observed = $null
        if ($TimestampField) {
            $timestampElement = [System.Text.Json.JsonElement]::new()
            if (-not $root.TryGetProperty($TimestampField, [ref]$timestampElement) -or
                $timestampElement.ValueKind -ne [System.Text.Json.JsonValueKind]::String) {
                throw "JSONL timestamp '$TimestampField' is missing or not a string: $Path"
            }
            $observed = $timestampElement.GetString()
        }
        if ($OwnerField) {
            $ownerElement = [System.Text.Json.JsonElement]::new()
            if (-not $root.TryGetProperty($OwnerField, [ref]$ownerElement) -or
                $ownerElement.ValueKind -ne [System.Text.Json.JsonValueKind]::Number) {
                throw "JSONL owner '$OwnerField' is missing or not numeric: $Path"
            }
            $owner = $ownerElement.GetInt64()
            if ($ExpectedOwnerPid -ge 0 -and $owner -ne $ExpectedOwnerPid) {
                throw "JSONL owner mismatch: expected $ExpectedOwnerPid, observed ${owner}: $Path"
            }
        } else { $owner = $null }
        $recordType = $null
        if ($RecordTypeField) {
            $typeElement = [System.Text.Json.JsonElement]::new()
            if (-not $root.TryGetProperty($RecordTypeField, [ref]$typeElement) -or
                $typeElement.ValueKind -ne [System.Text.Json.JsonValueKind]::String) {
                throw "JSONL record type '$RecordTypeField' is missing or not a string: $Path"
            }
            $recordType = $typeElement.GetString()
            if ($ExpectedRecordType -and $recordType -cne $ExpectedRecordType) {
                throw "JSONL record type mismatch: expected '$ExpectedRecordType', observed '$recordType': $Path"
            }
        } else { $recordType = $null }
        if ($observed) {
            try { $observedAt = [DateTimeOffset]::Parse($observed, [Globalization.CultureInfo]::InvariantCulture,
                [Globalization.DateTimeStyles]::RoundtripKind) }
            catch { throw "JSONL timestamp '$TimestampField' is not valid round-trip time: $Path" }
            $ageSeconds = ([DateTimeOffset]::UtcNow - $observedAt).TotalSeconds
            if ($ageSeconds -lt (-1 * $FutureToleranceSeconds)) { throw "JSONL timestamp is in the future: $Path" }
            if ($MaxAgeSeconds -ge 0 -and $ageSeconds -gt $MaxAgeSeconds) {
                throw "JSONL record is stale ($([Math]::Round($ageSeconds, 3)) seconds): $Path"
            }
        } else { $ageSeconds = $null }
    } finally {
        $document.Dispose()
    }

    $record = ConvertFrom-Json -InputObject $line -DateKind String -ErrorAction Stop
    [pscustomobject]@{
        record = $record
        raw_record = $line
        record_sha256 = [Convert]::ToHexString([Security.Cryptography.SHA256]::HashData($recordBytes)).ToLowerInvariant()
        record_start_offset = [long]($tailOffset + $recordStart)
        record_end_offset = [long]($tailOffset + $recordEnd)
        file_length_bytes = [long]$fileLength
        age_seconds = $ageSeconds
        observed_at_utc = $observed
        owner_pid = $owner
        record_type = $recordType
        trailing_partial_bytes = $partialBytes.Length
        trailing_partial_sha256 = $partialSha
        trailing_partial_path = $partialPath
        trailing_partial_base64 = if ($partialBytes.Length) { [Convert]::ToBase64String($partialBytes) } else { $null }
    }
}
