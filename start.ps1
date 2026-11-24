# One-command setup + launch (Windows PowerShell). Re-running is safe.
# If scripts are blocked:  Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot
if (-not (Test-Path ".venv")) {
    Write-Host "Creating virtual environment..."
    python -m venv .venv
    if ($LASTEXITCODE -ne 0) { throw "Could not create .venv. Is Python 3.10+ installed and on PATH?" }
}
$py = ".\.venv\Scripts\python.exe"
# Windows PowerShell 5.1 turns a native command's redirected stderr into a terminating error when
# ErrorActionPreference is Stop, so relax it just for this "are packages installed?" probe.
$ErrorActionPreference = "Continue"
& $py -c "import torch, cv2, fastapi, multipart" 2>$null
$installed = ($LASTEXITCODE -eq 0)
$ErrorActionPreference = "Stop"
if (-not $installed) {
    Write-Host "Installing dependencies (first run only, ~250 MB)..."
    & $py -m pip install --upgrade pip
    & $py -m pip install -r requirements.txt
    if ($LASTEXITCODE -ne 0) { throw "pip install failed - check your internet connection and try again." }
}
$port = if ($env:PORT) { $env:PORT } else { "8000" }
Write-Host "Open http://localhost:$port in your browser (Ctrl+C to stop)."
& $py -m uvicorn app.main:app --host 127.0.0.1 --port $port
