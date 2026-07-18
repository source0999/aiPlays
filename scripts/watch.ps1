param([string]$Model = '')
$ErrorActionPreference = 'Stop'; $repo = Split-Path $PSScriptRoot -Parent; $params = @('-m','aiplays','watch','--config',"$repo\configs\pokemon_red.yaml"); if ($Model) { $params += @('--model',$Model) }; & "$repo\.venv\Scripts\python.exe" @params; exit $LASTEXITCODE
