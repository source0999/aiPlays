$ErrorActionPreference = 'Stop'; $repo = Split-Path $PSScriptRoot -Parent; $python = Join-Path $repo '.venv\Scripts\python.exe'
Push-Location $repo; try { & $python -m ruff check .; & $python -m mypy src; & $python -m pytest -q; if ($LASTEXITCODE) { exit $LASTEXITCODE } } finally { Pop-Location }
