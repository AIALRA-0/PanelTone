param(
    [string]$EnvironmentPath = ".venv-flux",
    [int]$Port = 8781,
    [string]$ModelCachePath = "",
    [double]$IdleReleaseSeconds = 60
)

$ErrorActionPreference = "Stop"
$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$pythonPath = Join-Path (Join-Path $projectRoot $EnvironmentPath) "Scripts\python.exe"

if (-not (Test-Path -LiteralPath $pythonPath)) {
    throw "Model environment is missing; run scripts/install_flux2_klein.ps1 first"
}

# Keep the model service local and allow one long-lived process to reuse the loaded model
$env:PANELTONE_MODEL_ID = "black-forest-labs/FLUX.2-klein-4B"
$env:PANELTONE_MODEL_CPU_OFFLOAD = "1"
$env:PANELTONE_MODEL_IDLE_RELEASE_SECONDS = [string]$IdleReleaseSeconds
# Keep the previous names for the alpha compatibility window
$env:MANGA_REPAINT_MODEL_ID = $env:PANELTONE_MODEL_ID
$env:MANGA_REPAINT_MODEL_CPU_OFFLOAD = $env:PANELTONE_MODEL_CPU_OFFLOAD
if ($ModelCachePath) {
    $env:HF_HOME = $ModelCachePath
}
& $pythonPath -m uvicorn manga_repaint.model_server:app --host 127.0.0.1 --port $Port
