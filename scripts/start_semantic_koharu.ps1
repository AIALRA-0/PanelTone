param(
    [string]$EnvironmentPath = ".venv-semantic",
    [string]$ModelRootPath = "",
    [int]$Port = 8782
)

$ErrorActionPreference = "Stop"
$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$pythonPath = Join-Path (Join-Path $projectRoot $EnvironmentPath) "Scripts\python.exe"
if (-not $ModelRootPath) {
    $ModelRootPath = Join-Path $env:LOCALAPPDATA "PanelTone\models\semantic\koharu-yolo26s"
}
if (-not (Test-Path -LiteralPath $pythonPath)) {
    throw "语义模型环境不存在，请先运行 scripts/install_semantic_koharu.ps1"
}
if (-not (Test-Path -LiteralPath (Join-Path $ModelRootPath "model.safetensors"))) {
    throw "语义模型权重不存在，请先运行 scripts/install_semantic_koharu.ps1"
}
$env:PANELTONE_SEMANTIC_MODEL_DIR = $ModelRootPath
& $pythonPath -m uvicorn manga_repaint.semantic_service:app --host 127.0.0.1 --port $Port
