$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $root

Write-Host 'CircuitMind: http://127.0.0.1:8000/student'
& conda run -n llm python -m uvicorn backend.app.main:app --host 127.0.0.1 --port 8000

