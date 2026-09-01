param(
    [Parameter(Mandatory = $true)]
    [string]$ProtectedJobId,
    [string]$AppBaseUrl = "http://127.0.0.1:8765",
    [string]$ModelBaseUrl = "http://127.0.0.1:8781",
    [string]$EvidenceDirectory = "",
    [string]$ModelCachePath = "",
    [int]$PollSeconds = 5,
    [int]$ReadinessTimeoutSeconds = 300,
    [int]$MaxWaitSeconds = 0,
    [switch]$SkipWait,
    [switch]$SkipControls
)

$ErrorActionPreference = "Stop"
$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$terminalStatuses = @("completed", "needs_attention", "failed", "cancelled")
$startedAt = Get-Date

if (-not $EvidenceDirectory) {
    $EvidenceDirectory = Join-Path $env:LOCALAPPDATA "PanelTone\maintenance"
}
New-Item -ItemType Directory -Path $EvidenceDirectory -Force | Out-Null

# Reuse the verified local cache when the application has one, otherwise keep
# Hugging Face's user cache unchanged.  This makes the restart deterministic
# without copying or redownloading model weights.
if (-not $ModelCachePath) {
    $defaultModelCache = Join-Path $env:USERPROFILE ".cache\huggingface\hub\models--black-forest-labs--FLUX.2-klein-4B"
    $appModelCache = Join-Path $env:LOCALAPPDATA "PanelTone\models"
    $appModelHub = Join-Path $appModelCache "hub\models--black-forest-labs--FLUX.2-klein-4B"
    if (-not (Test-Path -LiteralPath $defaultModelCache) -and (Test-Path -LiteralPath $appModelHub)) {
        $ModelCachePath = $appModelCache
    }
}

function Get-Json([string]$Uri) {
    try {
        return Invoke-RestMethod -Uri $Uri -TimeoutSec 10 -Headers @{ "Cache-Control" = "no-cache" }
    } catch {
        return [pscustomobject]@{ available = $false; error = $_.Exception.Message; uri = $Uri }
    }
}

function Write-Evidence([string]$Name, $Value) {
    $path = Join-Path $EvidenceDirectory $Name
    $Value | ConvertTo-Json -Depth 20 | Set-Content -LiteralPath $path -Encoding UTF8
    return $path
}

function Get-ModelProcesses {
    Get-CimInstance Win32_Process | Where-Object {
        $_.CommandLine -and
        $_.CommandLine -match "manga_repaint\.model_server:app" -and
        $_.CommandLine -match "--port\s+8781"
    }
}

function Get-DescendantIds([int[]]$RootIds) {
    $all = Get-CimInstance Win32_Process | Select-Object ProcessId, ParentProcessId
    $result = [System.Collections.Generic.List[int]]::new()
    $pending = [System.Collections.Generic.Queue[int]]::new()
    foreach ($id in $RootIds) { $pending.Enqueue($id) }
    while ($pending.Count -gt 0) {
        $parent = $pending.Dequeue()
        foreach ($child in $all | Where-Object { $_.ParentProcessId -eq $parent }) {
            $childId = [int]$child.ProcessId
            if (-not $result.Contains($childId)) {
                $result.Add($childId)
                $pending.Enqueue($childId)
            }
        }
    }
    return $result.ToArray()
}

function Stop-ExactModelProcesses {
    $processes = @(Get-ModelProcesses)
    if (-not $processes.Count) { return @() }
    $rootIds = @($processes | ForEach-Object { [int]$_.ProcessId })
    $ids = @($rootIds + (Get-DescendantIds $rootIds)) | Sort-Object -Unique
    # Ask the exact process tree to exit first.  A forceful stop is only the
    # bounded fallback when the model server does not honor graceful exit.
    foreach ($id in ($ids | Sort-Object -Descending)) {
        try {
            $process = Get-Process -Id $id -ErrorAction Stop
            if ($process.MainWindowHandle -ne 0) {
                $process.CloseMainWindow() | Out-Null
            } else {
                Stop-Process -Id $id -ErrorAction Stop
            }
        } catch { }
    }
    $graceDeadline = (Get-Date).AddSeconds(10)
    do {
        $remaining = @($ids | Where-Object { Get-Process -Id $_ -ErrorAction SilentlyContinue })
        if (-not $remaining.Count) { break }
        Start-Sleep -Milliseconds 250
    } while ((Get-Date) -lt $graceDeadline)
    foreach ($id in ($ids | Sort-Object -Descending)) {
        if (Get-Process -Id $id -ErrorAction SilentlyContinue) {
            try { Stop-Process -Id $id -Force -ErrorAction Stop } catch { }
        }
    }
    $deadline = (Get-Date).AddSeconds(20)
    do {
        $listeners = @(Get-NetTCPConnection -LocalPort 8781 -State Listen -ErrorAction SilentlyContinue)
        if (-not $listeners.Count) { break }
        Start-Sleep -Milliseconds 250
    } while ((Get-Date) -lt $deadline)
    if (@(Get-NetTCPConnection -LocalPort 8781 -State Listen -ErrorAction SilentlyContinue).Count) {
        throw "8781 端口在 20 秒内未释放"
    }
    return $ids
}

function Start-ModelProcess([string]$EnvironmentPath = ".venv-flux", [double]$IdleSeconds = 60) {
    $scriptPath = Join-Path $PSScriptRoot "start_flux2_klein.ps1"
    $stdout = Join-Path $EvidenceDirectory "flux2-maintenance.stdout.log"
    $stderr = Join-Path $EvidenceDirectory "flux2-maintenance.stderr.log"
    $arguments = @(
        "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", "`"$scriptPath`"",
        "-Port", "8781", "-EnvironmentPath", $EnvironmentPath,
        "-IdleReleaseSeconds", [string]$IdleSeconds
    )
    if ($ModelCachePath) { $arguments += @("-ModelCachePath", "`"$ModelCachePath`"") }
    return Start-Process -FilePath "powershell.exe" -ArgumentList $arguments `
        -WorkingDirectory $projectRoot -WindowStyle Hidden -PassThru `
        -RedirectStandardOutput $stdout -RedirectStandardError $stderr
}

function Wait-ModelHealth {
    $deadline = (Get-Date).AddSeconds($ReadinessTimeoutSeconds)
    do {
        $health = Get-Json "$ModelBaseUrl/health"
        $names = @($health.PSObject.Properties.Name)
        if ($health.state -in @("idle", "ready") -and
            $null -ne $health.loaded -and
            $null -ne $health.active_requests -and
            $null -ne $health.last_activity -and
            $null -ne $health.idle_release_seconds -and
            $names -contains "supports_interrupt" -and
            $names -contains "supports_release" -and
            $health.idle_release_seconds -eq 60) {
            return $health
        }
        Start-Sleep -Seconds 2
    } while ((Get-Date) -lt $deadline)
    throw "8781 在 $ReadinessTimeoutSeconds 秒内未返回新的控制能力"
}

function Get-GpuSnapshot {
    try {
        $output = & nvidia-smi --query-gpu=name,utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw,driver_version --format=csv,noheader,nounits 2>&1
        if ($LASTEXITCODE -ne 0) {
            return [pscustomobject]@{ captured_at = (Get-Date).ToUniversalTime().ToString("o"); available = $false; reason = (@($output) -join " ") }
        }
        return [pscustomobject]@{ captured_at = (Get-Date).ToUniversalTime().ToString("o"); available = $true; output = @($output) }
    } catch {
        return [pscustomobject]@{ captured_at = (Get-Date).ToUniversalTime().ToString("o"); available = $false; reason = $_.Exception.Message }
    }
}

while (-not $SkipWait) {
    $job = Get-Json "$AppBaseUrl/api/jobs/$ProtectedJobId"
    if ($job.status -in $terminalStatuses) { break }
    if ($MaxWaitSeconds -gt 0 -and ((Get-Date) - $startedAt).TotalSeconds -ge $MaxWaitSeconds) {
        throw "受保护任务仍未进入最终状态，未执行 8781 重启"
    }
    Start-Sleep -Seconds ([Math]::Max(1, $PollSeconds))
}

$jobs = Get-Json "$AppBaseUrl/api/jobs?include_archived=true"
$appHealth = Get-Json "$AppBaseUrl/api/health"
$appGpu = Get-Json "$AppBaseUrl/api/gpu"
$modelHealthBefore = Get-Json "$ModelBaseUrl/health"
$listenersBefore = @(Get-NetTCPConnection -LocalPort 8781 -State Listen -ErrorAction SilentlyContinue | Select-Object LocalAddress, LocalPort, OwningProcess)
$processesBefore = @(Get-ModelProcesses | Select-Object ProcessId, ParentProcessId, Name, ExecutablePath, CommandLine)
$gpuBefore = Get-GpuSnapshot
Write-Evidence "before.json" ([pscustomobject]@{
    captured_at = (Get-Date).ToUniversalTime().ToString("o")
    protected_job = $job
    jobs = $jobs
    app_health = $appHealth
    app_gpu = $appGpu
    model_health = $modelHealthBefore
    model_cache_path = $ModelCachePath
    listeners = $listenersBefore
    processes = $processesBefore
    gpu = $gpuBefore
}) | Out-Null

if ($job.status -notin $terminalStatuses) { throw "受保护任务未进入允许维护的最终状态" }
$protectedQueuePosition = $job.queue_position
$protectedCurrentPage = if ($job.progress) { $job.progress.current_page } else { $null }
if ($null -ne $protectedQueuePosition) { throw "受保护任务仍在 GPU 队列中，未执行 8781 重启" }
if ($null -ne $protectedCurrentPage) { throw "受保护任务仍有当前页面，未执行 8781 重启" }
$running = @($jobs | Where-Object { $_.status -eq "running" })
if ($running.Count) { throw "仍有 $($running.Count) 个任务运行，未执行 8781 重启" }
$healthNames = @($modelHealthBefore.PSObject.Properties.Name)
$requiredHealthFields = @("active_requests", "last_activity", "idle_release_seconds", "supports_interrupt", "supports_release", "state", "loaded")
$missingHealthFields = @($requiredHealthFields | Where-Object { $healthNames -notcontains $_ })
if ($missingHealthFields.Count) {
    throw "8781 健康接口缺少控制字段: $($missingHealthFields -join ', ')，未执行维护重启"
}
if ([int]$modelHealthBefore.active_requests -ne 0) {
    throw "8781 仍有活动请求，未执行维护重启"
}

$oldProcesses = $processesBefore
try {
    Stop-ExactModelProcesses | Out-Null
    Start-ModelProcess | Out-Null
    $modelHealthAfter = Wait-ModelHealth
    $readyListeners = @(Get-NetTCPConnection -LocalPort 8781 -State Listen -ErrorAction SilentlyContinue)
    if ($readyListeners.Count -ne 1) { throw "8781 未保持单监听器" }
    if (-not $modelHealthAfter.supports_interrupt -or -not $modelHealthAfter.supports_release) {
        throw "8781 控制能力验收失败"
    }
    if (-not $SkipControls) {
        $verifyScript = Join-Path $PSScriptRoot "verify_flux2_klein_controls.py"
        $verifyPython = Join-Path (Join-Path $projectRoot ".venv-flux") "Scripts\python.exe"
        if (-not (Test-Path -LiteralPath $verifyScript)) { throw "缺少 8781 控制验收脚本" }
        if (-not (Test-Path -LiteralPath $verifyPython)) { throw "缺少 .venv-flux 控制验收环境" }
        $verifyOutput = & $verifyPython $verifyScript `
            --model-url $ModelBaseUrl `
            --app-url $AppBaseUrl `
            --evidence-directory $EvidenceDirectory `
            --idle-seconds 60 `
            --idle-wait-seconds 70 2>&1
        $verifyExitCode = $LASTEXITCODE
        $verifyOutput | Set-Content -LiteralPath (Join-Path $EvidenceDirectory "controls.stdout.log") -Encoding UTF8
        if ($verifyExitCode -ne 0) { throw "8781 控制验收失败，详见 controls.stdout.log" }
    }
} catch {
    # 失败时只回滚模型服务，绝不修改任务数据库和页面结果
    Stop-ExactModelProcesses | Out-Null
    $python = $null
    if ($oldProcesses.Count -and $oldProcesses[0].ExecutablePath -and (Test-Path -LiteralPath $oldProcesses[0].ExecutablePath)) {
        $python = $oldProcesses[0].ExecutablePath
    } elseif ($oldProcesses.Count -and $oldProcesses[0].CommandLine -match '(?i)("[^"]+python\.exe"|[^\s"]+python\.exe)') {
        $python = $Matches[1].Trim('"')
    }
    if ($python -and (Test-Path -LiteralPath $python)) {
        Start-Process -FilePath $python -ArgumentList @("-m", "uvicorn", "manga_repaint.model_server:app", "--host", "127.0.0.1", "--port", "8781") -WorkingDirectory $projectRoot -WindowStyle Hidden | Out-Null
    }
    Write-Evidence "failure.json" ([pscustomobject]@{ captured_at = (Get-Date).ToUniversalTime().ToString("o"); error = $_.Exception.Message }) | Out-Null
    throw
}

$listenersAfter = @(Get-NetTCPConnection -LocalPort 8781 -State Listen -ErrorAction SilentlyContinue | Select-Object LocalAddress, LocalPort, OwningProcess)
$processesAfter = @(Get-ModelProcesses | Select-Object ProcessId, ParentProcessId, Name, ExecutablePath, CommandLine)
$gpuAfter = Get-GpuSnapshot
$appGpuAfter = Get-Json "$AppBaseUrl/api/gpu"
Write-Evidence "after-start.json" ([pscustomobject]@{
    captured_at = (Get-Date).ToUniversalTime().ToString("o")
    model_health = $modelHealthAfter
    model_cache_path = $ModelCachePath
    listeners = $listenersAfter
    processes = $processesAfter
    gpu = $gpuAfter
    app_gpu = $appGpuAfter
}) | Out-Null

Write-Output ($modelHealthAfter | ConvertTo-Json -Depth 10)
