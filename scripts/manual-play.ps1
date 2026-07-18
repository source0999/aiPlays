$ErrorActionPreference = 'Stop'; $repo = Split-Path $PSScriptRoot -Parent; & "$repo\.venv\Scripts\python.exe" -m aiplays manual --config "$repo\configs\pokemon_red.yaml" @args; exit $LASTEXITCODE
