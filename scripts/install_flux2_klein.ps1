param(
    [string]$EnvironmentPath = ".venv-flux",
    [string]$ModelCachePath = "",
    [switch]$DownloadModel
)

$ErrorActionPreference = "Stop"
$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$environmentRoot = Join-Path $projectRoot $EnvironmentPath

# Create an isolated model environment so CUDA and diffusion packages cannot destabilize the app
python -m venv $environmentRoot
$pythonPath = Join-Path $environmentRoot "Scripts\python.exe"

# Install the CUDA 12.8 PyTorch build used by the local RTX GPU
& $pythonPath -m pip install --upgrade pip
& $pythonPath -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128

# Install the application and model-service dependencies from this checkout
& $pythonPath -m pip install -e "$projectRoot[models]"

if ($DownloadModel) {
    # Download the pinned Apache-2.0 FLUX.2 Klein 4B revision into the selected cache
    if ($ModelCachePath) {
        $env:HF_HOME = $ModelCachePath
    }
    & $pythonPath -c "from huggingface_hub import snapshot_download; snapshot_download('black-forest-labs/FLUX.2-klein-4B', revision='e7b7dc27f91deacad38e78976d1f2b499d76a294')"
}

Write-Host "Model environment ready at $environmentRoot"
