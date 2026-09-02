param(
    [string]$EnvironmentPath = ".venv-semantic",
    [string]$RuntimeEnvironmentPath = ".venv-flux",
    [string]$ModelRootPath = "",
    [string]$Revision = "",
    [switch]$SkipEnvironmentInstall
)

$ErrorActionPreference = "Stop"
$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$environmentRoot = Join-Path $projectRoot $EnvironmentPath
$pythonPath = Join-Path $environmentRoot "Scripts\python.exe"
if (-not $ModelRootPath) {
    $ModelRootPath = Join-Path $env:LOCALAPPDATA "PanelTone\models\semantic\koharu-yolo26s"
}
$modelRoot = [System.IO.DirectoryInfo](New-Item -ItemType Directory -Path $ModelRootPath -Force)

if (-not (Test-Path -LiteralPath $pythonPath)) {
    if ($SkipEnvironmentInstall) {
        throw "Semantic model environment does not exist: $pythonPath"
    }
    python -m venv $environmentRoot
}

$runtimeSitePackages = Join-Path (Join-Path $projectRoot $RuntimeEnvironmentPath) "Lib\site-packages"
$sharedRuntimePath = Join-Path $environmentRoot "Lib\site-packages\paneltone_shared_runtime.pth"
$sharedRuntimeReady = Test-Path -LiteralPath (Join-Path $runtimeSitePackages "torch")
if ($sharedRuntimeReady) {
    # Reuse the already verified CUDA runtime in .venv-flux instead of
    # downloading a second multi-gigabyte Torch wheel.  The semantic service
    # remains a separate venv and owns its Ultralytics dependency and model.
    $runtimeSitePackages | Set-Content -LiteralPath $sharedRuntimePath -Encoding ASCII
}

if (-not $SkipEnvironmentInstall) {
    & $pythonPath -m pip install --upgrade pip
    if ($LASTEXITCODE -ne 0) { throw "Semantic environment pip upgrade failed" }
    if (-not $sharedRuntimeReady) {
        & $pythonPath -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
        if ($LASTEXITCODE -ne 0) { throw "Torch CUDA dependency installation failed" }
    } else {
        & $pythonPath -c "import torch; print(torch.__version__)"
        if ($LASTEXITCODE -ne 0) { throw "Unable to reuse the installed CUDA Torch runtime" }
    }
    $editableSpec = $projectRoot + '[models]'
    & $pythonPath -m pip install -e $editableSpec
    if ($LASTEXITCODE -ne 0) { throw "PanelTone semantic dependency installation failed" }
    & $pythonPath -m pip install "ultralytics==8.4.43"
    if ($LASTEXITCODE -ne 0) { throw "Ultralytics dependency installation failed" }
}

$repoId = "mayocream/koharu-yolo26s"
if (-not $Revision) {
    $Revision = (& $pythonPath -c "from huggingface_hub import model_info; print(model_info('$repoId').sha)").Trim()
}
if (-not $Revision) {
    throw "Unable to resolve the pinned semantic model revision"
}

$downloadCode = "from huggingface_hub import snapshot_download; snapshot_download(repo_id='$repoId', revision='$Revision', local_dir=r'$($modelRoot.FullName)', allow_patterns=['*.safetensors', '*.yaml', '*.json'])"
& $pythonPath -c $downloadCode
if ($LASTEXITCODE -ne 0) { throw "Semantic model download failed" }

$weightPath = Join-Path $modelRoot.FullName "model.safetensors"
$architecturePath = Join-Path $modelRoot.FullName "yolo26s-seg.yaml"
$configPath = Join-Path $modelRoot.FullName "config.json"
if (-not (Test-Path -LiteralPath $weightPath) -or
    -not (Test-Path -LiteralPath $architecturePath) -or
    -not (Test-Path -LiteralPath $configPath)) {
    throw "Semantic model download is incomplete; SafeTensors, architecture, or config is missing"
}

$sha256 = (Get-FileHash -Algorithm SHA256 -LiteralPath $weightPath).Hash.ToLowerInvariant()
$metadata = [ordered]@{
    id = "semantic-manga-v1-compatible"
    provider = "semantic-manga-v1-compatible/koharu-yolo26s"
    repository = $repoId
    revision = $Revision
    weight_sha256 = $sha256
    model_root = $modelRoot.FullName
    installed_at = (Get-Date).ToUniversalTime().ToString("o")
}
$markerPath = Join-Path $modelRoot.FullName "semantic-model.json"
$metadata | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath $markerPath -Encoding UTF8
$installedDirectory = Join-Path $modelRoot.Parent.Parent.FullName "installed"
New-Item -ItemType Directory -Path $installedDirectory -Force | Out-Null
$metadata | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath (Join-Path $installedDirectory "semantic-manga-v1.json") -Encoding UTF8
Write-Output ($metadata | ConvertTo-Json -Depth 5)
