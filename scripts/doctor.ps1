param([switch]$RequireRom)
$ErrorActionPreference = 'Stop'; $repo = Split-Path $PSScriptRoot -Parent; $python = Join-Path $repo '.venv\Scripts\python.exe'
& $python -m aiplays doctor --config "$repo\configs\pokemon_red.yaml" $(if ($RequireRom) { '--require-rom' }); exit $LASTEXITCODE
