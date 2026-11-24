# One-command setup + launch (Windows PowerShell). Re-running is safe.
# If scripts are blocked:  Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot
if (-not (Test-Path ".venv")) {
    Write-Host "Creating virtual environment..."
    python -m venv .venv
}
$py = ".\.venv\Scripts\python.exe"
& $py -c "import torch, cv2, fastapi, multipart" 2>$null
if ($LASTEXITCODE -ne 0) {
    Write-Host "Installing dependencies (first run only, ~250 MB)..."
    & $py -m pip install --upgrade pip
    & $py -m pip install -r requirements.txt
}
$port = if ($env:PORT) { $env:PORT } else { "8000" }
Write-Host "Open http://localhost:$port in your browser (Ctrl+C to stop)."
& $py -m uvicorn app.main:app --host 127.0.0.1 --port $port
