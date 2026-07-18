param([int]$Timesteps = 1000000, [int]$NumEnvs = 1)
$ErrorActionPreference = 'Stop'; $repo = Split-Path $PSScriptRoot -Parent; & "$repo\.venv\Scripts\python.exe" -m aiplays train --config "$repo\configs\pokemon_red.yaml" --timesteps $Timesteps --num-envs $NumEnvs; exit $LASTEXITCODE
