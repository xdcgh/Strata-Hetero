param([string]$PathsJson)
$ErrorActionPreference='Stop'
$paths=ConvertFrom-Json -InputObject $PathsJson -AsHashtable
foreach($path in $paths){$tokens=$null;$errors=$null;[void][System.Management.Automation.Language.Parser]::ParseFile($path,[ref]$tokens,[ref]$errors);if($errors.Count){$errors|ForEach-Object{[Console]::Error.WriteLine($_.ToString())};exit 1};[Console]::Out.WriteLine((Split-Path -Leaf $path)+' parse-only pass')}
