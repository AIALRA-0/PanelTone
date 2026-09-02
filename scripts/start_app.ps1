param(
    [int]$Port = 8765,
    [int]$ModelPort = 8781,
    [int]$SemanticPort = 8782,
    [string]$ModelCachePath = "",
    [string]$SemanticModelRootPath = "",
    [switch]$SkipModelService
)

$ErrorActionPreference = "Stop"
$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$pythonPath = Join-Path $projectRoot ".venv\Scripts\python.exe"
$engineConfig = Join-Path $projectRoot "configs\engines.example.json"
$semanticPythonPath = Join-Path $projectRoot ".venv-semantic\Scripts\python.exe"
if (-not $SemanticModelRootPath) {
    $SemanticModelRootPath = Join-Path $env:LOCALAPPDATA "PanelTone\models\semantic\koharu-yolo26s"
}

if (-not (Test-Path -LiteralPath $pythonPath)) {
    throw "Application environment is missing; install the project before starting"
}

# Leave Hugging Face's default cache unchanged unless the user selected another directory
# This lets an existing verified download be reused instead of silently starting from an empty cache
if (-not $ModelCachePath) {
    $defaultModelCache = Join-Path $env:USERPROFILE ".cache\huggingface\hub\models--black-forest-labs--FLUX.2-klein-4B"
    $appModelCache = Join-Path $env:LOCALAPPDATA "PanelTone\models"
    $appModelHub = Join-Path $appModelCache "hub\models--black-forest-labs--FLUX.2-klein-4B"
    if (-not (Test-Path -LiteralPath $defaultModelCache) -and (Test-Path -LiteralPath $appModelHub)) {
        $ModelCachePath = $appModelCache
    }
}

# Start the optional local model helper when its isolated environment is installed
$modelHealthy = $false
try {
    $null = Invoke-RestMethod -Uri "http://127.0.0.1:$ModelPort/health" -TimeoutSec 2
    $modelHealthy = $true
} catch {
    $modelHealthy = $false
}
if (-not $modelHealthy -and -not $SkipModelService -and (Test-Path -LiteralPath (Join-Path $projectRoot ".venv-flux\Scripts\python.exe"))) {
    $logDir = Join-Path $env:LOCALAPPDATA "PanelTone\logs"
    New-Item -ItemType Directory -Path $logDir -Force | Out-Null
    $modelScript = Join-Path $PSScriptRoot "start_flux2_klein.ps1"
    $arguments = @(
        "-NoProfile",
        "-ExecutionPolicy", "Bypass",
        "-File", "`"$modelScript`"",
        "-Port", $ModelPort
    )
    if ($ModelCachePath) {
        $arguments += @("-ModelCachePath", "`"$ModelCachePath`"")
    }
    Start-Process -FilePath "powershell.exe" -ArgumentList $arguments -WindowStyle Hidden `
        -RedirectStandardOutput (Join-Path $logDir "flux2.stdout.log") `
        -RedirectStandardError (Join-Path $logDir "flux2.stderr.log")
    for ($attempt = 0; $attempt -lt 20; $attempt++) {
        Start-Sleep -Milliseconds 500
        try {
            $null = Invoke-RestMethod -Uri "http://127.0.0.1:$ModelPort/health" -TimeoutSec 2
            $modelHealthy = $true
            break
        } catch {
            $modelHealthy = $false
        }
    }
}
if (-not $modelHealthy) {
    Write-Warning "The model service is not ready; PanelTone will open so the model can be installed or reconnected"
}

# Keep the optional segmentation stack out of the main application environment
# and bind it to loopback just like the diffusion service.
$semanticHealthy = $false
try {
    $null = Invoke-RestMethod -Uri "http://127.0.0.1:$SemanticPort/health" -TimeoutSec 2
    $semanticHealthy = $true
} catch {
    $semanticHealthy = $false
}
if (-not $semanticHealthy -and
    (Test-Path -LiteralPath $semanticPythonPath) -and
    (Test-Path -LiteralPath (Join-Path $SemanticModelRootPath "model.safetensors"))) {
    $logDir = Join-Path $env:LOCALAPPDATA "PanelTone\logs"
    New-Item -ItemType Directory -Path $logDir -Force | Out-Null
    $semanticScript = Join-Path $PSScriptRoot "start_semantic_koharu.ps1"
    $semanticArguments = @(
        "-NoProfile",
        "-ExecutionPolicy", "Bypass",
        "-File", "`"$semanticScript`"",
        "-Port", $SemanticPort,
        "-ModelRootPath", "`"$SemanticModelRootPath`""
    )
    Start-Process -FilePath "powershell.exe" -ArgumentList $semanticArguments -WindowStyle Hidden `
        -RedirectStandardOutput (Join-Path $logDir "semantic.stdout.log") `
        -RedirectStandardError (Join-Path $logDir "semantic.stderr.log")
    for ($attempt = 0; $attempt -lt 20; $attempt++) {
        Start-Sleep -Milliseconds 500
        try {
            $null = Invoke-RestMethod -Uri "http://127.0.0.1:$SemanticPort/health" -TimeoutSec 2
            $semanticHealthy = $true
            break
        } catch {
            $semanticHealthy = $false
        }
    }
}
if ($semanticHealthy) {
    $env:PANELTONE_SEMANTIC_URL = "http://127.0.0.1:$SemanticPort"
} else {
    Remove-Item Env:PANELTONE_SEMANTIC_URL -ErrorAction SilentlyContinue
    Write-Warning "语义保护服务未连接；PanelTone 将使用确定性文字、气泡、墨线和边框保护"
}

# Bind to localhost so book pages and task metadata are not exposed to the network
& $pythonPath -m manga_repaint.cli --engines $engineConfig serve `
    --host 127.0.0.1 --port $Port
