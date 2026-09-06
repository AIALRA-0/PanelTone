param(
    [string]$EnvironmentPath = "",
    [string]$CobraRepoPath = "",
    [string]$ModelCachePath = "",
    [string]$HfCachePath = "",
    [int]$Port = 8783
)

$ErrorActionPreference = "Stop"
$candidateRoot = Join-Path $env:LOCALAPPDATA "PanelTone\candidates"
if (-not $EnvironmentPath) { $EnvironmentPath = Join-Path $candidateRoot "cobra-runtime-311" }
if (-not $CobraRepoPath) { $CobraRepoPath = Join-Path $candidateRoot "Cobra" }
if (-not $ModelCachePath) { $ModelCachePath = Join-Path $candidateRoot "cobra-hf" }
if (-not $HfCachePath) { $HfCachePath = Join-Path $candidateRoot "hf-base" }
$pythonPath = if (Test-Path -LiteralPath (Join-Path $EnvironmentPath "python.exe")) {
    Join-Path $EnvironmentPath "python.exe"
} else {
    Join-Path $EnvironmentPath "Scripts\python.exe"
}
if (-not (Test-Path -LiteralPath $pythonPath)) {
    throw "Cobra candidate environment is missing: $pythonPath"
}
if (-not (Test-Path -LiteralPath (Join-Path $CobraRepoPath "app.py"))) {
    throw "Cobra source checkout is missing: $CobraRepoPath"
}
$env:PANELTONE_COBRA_REPO = (Resolve-Path $CobraRepoPath).Path
$env:PANELTONE_COBRA_MODEL_CACHE = $ModelCachePath
$env:HF_HOME = $HfCachePath
$env:PANELTONE_COBRA_HEADLESS = "1"
$env:PANELTONE_COBRA_TMP = Join-Path $env:LOCALAPPDATA "PanelTone\cobra-candidate"
New-Item -ItemType Directory -Force -Path $env:PANELTONE_COBRA_TMP | Out-Null
& $pythonPath -m uvicorn scripts.cobra_http_service:app --app-dir (Resolve-Path (Join-Path $PSScriptRoot "..")) --host 127.0.0.1 --port $Port
