param(
    [string]$ProjectRoot = "",
    [string]$CandidateRoot = "",
    [string]$SemanticModelRoot = "",
    [string]$DataRoot = "",
    [string]$ModelRoot = "",
    [ValidateRange(1, 300)]
    [int]$CheckIntervalSeconds = 10,
    [ValidateRange(1, 300)]
    [int]$RestartDelaySeconds = 5,
    [ValidateRange(10, 900)]
    [int]$UnhealthyRestartSeconds = 45,
    [string]$LogDirectory = "",
    [switch]$ValidateOnly
)

$ErrorActionPreference = "Stop"

if (-not $ProjectRoot) {
    $ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
}
$resolvedProjectRoot = (Resolve-Path -LiteralPath $ProjectRoot).Path

if (-not $CandidateRoot) {
    $CandidateRoot = Join-Path ([Environment]::GetFolderPath("LocalApplicationData")) `
        "PanelTone\candidates"
}
$resolvedCandidateRoot = [IO.Path]::GetFullPath($CandidateRoot)

if (-not $DataRoot) {
    $DataRoot = Join-Path ([Environment]::GetFolderPath("LocalApplicationData")) `
        "PanelTone\jobs"
}
$resolvedDataRoot = [IO.Path]::GetFullPath($DataRoot)

if (-not $ModelRoot) {
    $ModelRoot = Join-Path ([Environment]::GetFolderPath("LocalApplicationData")) `
        "PanelTone\models"
}
$resolvedModelRoot = [IO.Path]::GetFullPath($ModelRoot)

if (-not $LogDirectory) {
    $LogDirectory = Join-Path ([Environment]::GetFolderPath("LocalApplicationData")) `
        "PanelTone\logs\services"
}
$resolvedLogDirectory = [IO.Path]::GetFullPath($LogDirectory)
New-Item -ItemType Directory -Path $resolvedLogDirectory -Force | Out-Null
$supervisorLog = Join-Path $resolvedLogDirectory "supervisor.log"

$appPython = Join-Path $resolvedProjectRoot ".venv\Scripts\python.exe"
$fluxPython = Join-Path $resolvedProjectRoot ".venv-flux\Scripts\python.exe"
$semanticPython = Join-Path $resolvedProjectRoot ".venv-semantic\Scripts\python.exe"
$engineConfig = Join-Path $resolvedProjectRoot "configs\engines.example.json"
if (-not $SemanticModelRoot) {
    $SemanticModelRoot = Join-Path ([Environment]::GetFolderPath("LocalApplicationData")) `
        "PanelTone\models\semantic\koharu-yolo26s"
}
$semanticModelRoot = [IO.Path]::GetFullPath($SemanticModelRoot)
$cobraPython = Join-Path $resolvedCandidateRoot "cobra-runtime-311\python.exe"
$cobraRepo = Join-Path $resolvedCandidateRoot "Cobra"
$cobraCache = Join-Path $resolvedCandidateRoot "cobra-hf"
$cobraHfCache = Join-Path $resolvedCandidateRoot "hf-base"
$cobraTemp = Join-Path $resolvedCandidateRoot "cobra-tmp"

$requiredPaths = @(
    $appPython,
    $fluxPython,
    $semanticPython,
    $engineConfig,
    $resolvedDataRoot,
    $resolvedModelRoot,
    (Join-Path $semanticModelRoot "model.safetensors"),
    $cobraPython,
    (Join-Path $cobraRepo "app.py")
)
foreach ($path in $requiredPaths) {
    if (-not (Test-Path -LiteralPath $path)) {
        throw "Required PanelTone runtime path is missing: $path"
    }
}
New-Item -ItemType Directory -Path $cobraTemp -Force | Out-Null

function Write-SupervisorLog([string]$Message) {
    Add-Content -LiteralPath $supervisorLog -Encoding utf8 -Value (
        "{0:o} {1}" -f (Get-Date), $Message
    )
}

function Test-ServiceHealth([int]$Port, [string]$Path) {
    try {
        $response = Invoke-WebRequest `
            -Uri ("http://127.0.0.1:{0}{1}" -f $Port, $Path) `
            -UseBasicParsing `
            -TimeoutSec 3
        return $response.StatusCode -eq 200
    } catch {
        return $false
    }
}

function Get-ListeningProcessId([int]$Port) {
    $listener = Get-NetTCPConnection `
        -State Listen `
        -LocalPort $Port `
        -ErrorAction SilentlyContinue |
        Where-Object { $_.LocalAddress -in @("127.0.0.1", "::1") } |
        Select-Object -First 1
    if ($listener) {
        return [int]$listener.OwningProcess
    }
    return $null
}

function Test-OwnedServiceProcess(
    [int]$ProcessId,
    [string]$Executable,
    [string]$ProcessMarker
) {
    if ($ProcessId -le 0) {
        return $false
    }
    $process = Get-CimInstance Win32_Process `
        -Filter "ProcessId=$ProcessId" `
        -ErrorAction SilentlyContinue
    if (-not $process) {
        return $false
    }
    # A Windows virtual-environment launcher reports the base interpreter in
    # Win32_Process.ExecutablePath.  Match the executable name plus the unique
    # service command marker instead of rejecting a healthy venv child solely
    # because its resolved base path differs from the launcher path.
    $expectedExecutableName = [IO.Path]::GetFileName($Executable)
    $actualExecutableName = [IO.Path]::GetFileName([string]$process.ExecutablePath)
    return (
        $actualExecutableName.Equals(
            $expectedExecutableName,
            [StringComparison]::OrdinalIgnoreCase
        ) -and
        [string]$process.CommandLine -like "*$ProcessMarker*"
    )
}

function Start-ManagedProcess(
    [string]$Name,
    [string]$Executable,
    [string[]]$Arguments,
    [hashtable]$EnvironmentValues
) {
    $savedValues = @{}
    foreach ($key in $EnvironmentValues.Keys) {
        $savedValues[$key] = [Environment]::GetEnvironmentVariable($key, "Process")
        [Environment]::SetEnvironmentVariable(
            $key,
            [string]$EnvironmentValues[$key],
            "Process"
        )
    }
    try {
        $stdout = Join-Path $resolvedLogDirectory "$Name.stdout.log"
        $stderr = Join-Path $resolvedLogDirectory "$Name.stderr.log"
        $process = Start-Process `
            -FilePath $Executable `
            -ArgumentList $Arguments `
            -WorkingDirectory $resolvedProjectRoot `
            -WindowStyle Hidden `
            -RedirectStandardOutput $stdout `
            -RedirectStandardError $stderr `
            -PassThru
        Write-SupervisorLog "started service=$Name pid=$($process.Id)"
        return $process
    } finally {
        foreach ($key in $EnvironmentValues.Keys) {
            [Environment]::SetEnvironmentVariable($key, $savedValues[$key], "Process")
        }
    }
}

$services = @(
    [pscustomobject]@{
        Name = "semantic"
        Port = 8782
        HealthPath = "/health"
        Executable = $semanticPython
        Arguments = @(
            "-m", "uvicorn", "manga_repaint.semantic_service:app",
            "--host", "127.0.0.1", "--port", "8782"
        )
        Environment = @{
            PYTHONPATH = (Join-Path $resolvedProjectRoot "src")
            PANELTONE_SEMANTIC_MODEL_DIR = $semanticModelRoot
        }
        ProcessMarker = "manga_repaint.semantic_service:app"
    },
    [pscustomobject]@{
        Name = "flux"
        Port = 8781
        HealthPath = "/health"
        Executable = $fluxPython
        Arguments = @(
            "-m", "uvicorn", "manga_repaint.model_server:app",
            "--host", "127.0.0.1", "--port", "8781"
        )
        Environment = @{
            PYTHONPATH = (Join-Path $resolvedProjectRoot "src")
            PANELTONE_MODEL_ID = "black-forest-labs/FLUX.2-klein-4B"
            PANELTONE_MODEL_CPU_OFFLOAD = "1"
            PANELTONE_MODEL_IDLE_RELEASE_SECONDS = "60"
            MANGA_REPAINT_MODEL_ID = "black-forest-labs/FLUX.2-klein-4B"
            MANGA_REPAINT_MODEL_CPU_OFFLOAD = "1"
        }
        ProcessMarker = "manga_repaint.model_server:app"
    },
    [pscustomobject]@{
        Name = "cobra"
        Port = 8783
        HealthPath = "/health"
        Executable = $cobraPython
        Arguments = @(
            "-m", "uvicorn", "scripts.cobra_http_service:app",
            "--app-dir", ('"' + $resolvedProjectRoot + '"'),
            "--host", "127.0.0.1", "--port", "8783"
        )
        Environment = @{
            PYTHONPATH = "$resolvedProjectRoot;$(Join-Path $resolvedProjectRoot 'src')"
            PANELTONE_COBRA_REPO = $cobraRepo
            PANELTONE_COBRA_MODEL_CACHE = $cobraCache
            HF_HOME = $cobraHfCache
            PANELTONE_COBRA_HEADLESS = "1"
            PANELTONE_COBRA_TMP = $cobraTemp
        }
        ProcessMarker = "scripts.cobra_http_service:app"
    },
    [pscustomobject]@{
        Name = "app"
        Port = 8765
        HealthPath = "/api/health"
        Executable = $appPython
        Arguments = @(
            "-m", "manga_repaint.cli",
            "--engines", ('"' + $engineConfig + '"'),
            "serve", "--host", "127.0.0.1", "--port", "8765"
        )
        Environment = @{
            PYTHONPATH = (Join-Path $resolvedProjectRoot "src")
            PANELTONE_SEMANTIC_URL = "http://127.0.0.1:8782"
            PANELTONE_DATA_ROOT = $resolvedDataRoot
            PANELTONE_MODEL_ROOT = $resolvedModelRoot
        }
        ProcessMarker = "manga_repaint.cli"
    }
)

if ($ValidateOnly) {
    [pscustomobject]@{
        status = "valid"
        project_root = $resolvedProjectRoot
        candidate_root = $resolvedCandidateRoot
        data_root = $resolvedDataRoot
        model_root = $resolvedModelRoot
        semantic_model_root = $semanticModelRoot
        services = @($services | ForEach-Object {
            [pscustomobject]@{
                name = $_.Name
                endpoint = "http://127.0.0.1:$($_.Port)$($_.HealthPath)"
                executable = $_.Executable
            }
        })
        log_directory = $resolvedLogDirectory
    }
    exit 0
}

Write-SupervisorLog "supervisor_started"
$tracked = @{}
$lastStart = @{}
$unhealthySince = @{}
while ($true) {
    foreach ($service in $services) {
        if (Test-ServiceHealth $service.Port $service.HealthPath) {
            $unhealthySince.Remove($service.Name)
            continue
        }

        $trackedProcess = $tracked[$service.Name]
        if ($trackedProcess -and -not $trackedProcess.HasExited) {
            $now = Get-Date
            if (-not $unhealthySince[$service.Name]) {
                $unhealthySince[$service.Name] = $now
                Write-SupervisorLog "unhealthy service=$($service.Name) pid=$($trackedProcess.Id)"
                continue
            }
            if (($now - $unhealthySince[$service.Name]).TotalSeconds -lt $UnhealthyRestartSeconds) {
                continue
            }
            if (Test-OwnedServiceProcess `
                $trackedProcess.Id `
                $service.Executable `
                $service.ProcessMarker
            ) {
                Stop-Process -Id $trackedProcess.Id -Force -ErrorAction Stop
                Write-SupervisorLog "unhealthy_process_stopped service=$($service.Name) pid=$($trackedProcess.Id)"
            } else {
                Write-SupervisorLog "unhealthy_process_not_owned service=$($service.Name) pid=$($trackedProcess.Id)"
            }
            $tracked.Remove($service.Name)
            $unhealthySince.Remove($service.Name)
            continue
        }

        $now = Get-Date
        $listenerProcessId = Get-ListeningProcessId $service.Port
        if ($listenerProcessId) {
            if (-not $unhealthySince[$service.Name]) {
                $unhealthySince[$service.Name] = $now
                Write-SupervisorLog "unhealthy_listener service=$($service.Name) pid=$listenerProcessId"
                continue
            }
            if (($now - $unhealthySince[$service.Name]).TotalSeconds -lt $UnhealthyRestartSeconds) {
                continue
            }
            if (Test-OwnedServiceProcess `
                $listenerProcessId `
                $service.Executable `
                $service.ProcessMarker
            ) {
                Stop-Process -Id $listenerProcessId -Force -ErrorAction Stop
                Write-SupervisorLog "unhealthy_listener_stopped service=$($service.Name) pid=$listenerProcessId"
            } else {
                Write-SupervisorLog "unhealthy_listener_not_owned service=$($service.Name) pid=$listenerProcessId"
            }
            $unhealthySince.Remove($service.Name)
            continue
        }
        $unhealthySince.Remove($service.Name)
        $previousStart = $lastStart[$service.Name]
        if ($previousStart -and ($now - $previousStart).TotalSeconds -lt $RestartDelaySeconds) {
            continue
        }

        try {
            $tracked[$service.Name] = Start-ManagedProcess `
                $service.Name `
                $service.Executable `
                $service.Arguments `
                $service.Environment
            $lastStart[$service.Name] = $now
        } catch {
            Write-SupervisorLog (
                "start_error service=$($service.Name) message=$($_.Exception.Message)"
            )
        }
    }
    Start-Sleep -Seconds $CheckIntervalSeconds
}
