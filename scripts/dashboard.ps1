$ErrorActionPreference = 'Stop'; $repo = Split-Path $PSScriptRoot -Parent; & "$repo\.venv\Scripts\python.exe" -m aiplays dashboard --config "$repo\configs\pokemon_red.yaml"; exit $LASTEXITCODE
