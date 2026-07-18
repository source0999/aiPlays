param([int]$Timesteps = 50000)
$ErrorActionPreference = 'Stop'; $repo = Split-Path $PSScriptRoot -Parent; & "$repo\.venv\Scripts\python.exe" -m aiplays train --config "$repo\configs\pokemon_red.yaml" --timesteps $Timesteps --num-envs 1 --render; exit $LASTEXITCODE
