param(
    [string]$EnvironmentPath = ".venv"
)

$ErrorActionPreference = "Stop"
$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$environmentRoot = Join-Path $projectRoot $EnvironmentPath

# Create the application environment with the system Python selected by the user
python -m venv $environmentRoot
$pythonPath = Join-Path $environmentRoot "Scripts\python.exe"

# Install PanelTone and its local Web service dependencies without downloading model weights
& $pythonPath -m pip install --upgrade pip
& $pythonPath -m pip install -e $projectRoot

Write-Host "PanelTone is installed"
Write-Host "Start it with: powershell -ExecutionPolicy Bypass -File scripts/start_app.ps1"
