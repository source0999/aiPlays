$ErrorActionPreference = 'Stop'
$repo = Split-Path $PSScriptRoot -Parent
$python = if (Get-Command py -ErrorAction SilentlyContinue) { 'py' } else { 'python' }
if ($python -eq 'py') { & py -3.11 -m venv "$repo\.venv" } else { & python -m venv "$repo\.venv" }
$venvPython = Join-Path $repo '.venv\Scripts\python.exe'
& $venvPython -m pip install --upgrade pip setuptools wheel
& $venvPython -m pip install -e "$repo[dev]"
& $venvPython -m pip freeze | Set-Content -Encoding utf8 "$repo\requirements-lock.txt"
& $venvPython -m aiplays doctor
& $venvPython -m pytest -q
Write-Host "Next: copy your legal ROM to $repo\roms\pokemon_red.gb, then run .\scripts\doctor.ps1 -RequireRom"
