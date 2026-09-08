param(
    [string]$TaskName = "PanelTone Local Services",
    [string]$TunnelTaskName = "PanelTone Private Tunnel",
    [ValidateRange(10, 600)]
    [int]$HealthTimeoutSeconds = 120,
    [switch]$Force,
    [switch]$WaitForInput
)

$ErrorActionPreference = "Stop"

$services = @(
    [pscustomobject]@{ Name = "PanelTone"; Port = 8765; Path = "/api/health"; Marker = "manga_repaint.cli" },
    [pscustomobject]@{ Name = "FLUX"; Port = 8781; Path = "/health"; Marker = "manga_repaint.model_server:app" },
    [pscustomobject]@{ Name = "语义保护"; Port = 8782; Path = "/health"; Marker = "manga_repaint.semantic_service:app" },
    [pscustomobject]@{ Name = "Cobra"; Port = 8783; Path = "/health"; Marker = "scripts.cobra_http_service:app" }
)

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

function Stop-VerifiedPanelToneListener([pscustomobject]$Service) {
    $listeners = Get-NetTCPConnection `
        -State Listen `
        -LocalPort $Service.Port `
        -ErrorAction SilentlyContinue |
        Where-Object { $_.LocalAddress -in @("127.0.0.1", "::1") }
    foreach ($listener in $listeners) {
        $process = Get-CimInstance Win32_Process `
            -Filter "ProcessId=$($listener.OwningProcess)" `
            -ErrorAction SilentlyContinue
        if (-not $process -or [string]$process.CommandLine -notlike "*$($Service.Marker)*") {
            throw "端口 $($Service.Port) 被非 PanelTone 进程占用，已停止强制启动。"
        }
        Stop-Process -Id $listener.OwningProcess -Force -ErrorAction Stop
        Write-Host "已停止旧的 $($Service.Name) 进程 $($listener.OwningProcess)"
    }
}

try {
    $task = Get-ScheduledTask -TaskName $TaskName -ErrorAction Stop
    if ($Force) {
        if ($task.State -eq "Running") {
            Stop-ScheduledTask -TaskName $TaskName
            Start-Sleep -Milliseconds 750
        }
        foreach ($service in $services) {
            Stop-VerifiedPanelToneListener $service
        }
    }

    $task = Get-ScheduledTask -TaskName $TaskName -ErrorAction Stop
    if ($task.State -ne "Running") {
        Start-ScheduledTask -TaskName $TaskName
    }
    $tunnel = Get-ScheduledTask -TaskName $TunnelTaskName -ErrorAction SilentlyContinue
    if ($tunnel -and $tunnel.State -ne "Running") {
        Start-ScheduledTask -TaskName $TunnelTaskName
    }

    $deadline = (Get-Date).AddSeconds($HealthTimeoutSeconds)
    do {
        $ready = @($services | Where-Object { Test-ServiceHealth $_.Port $_.Path })
        Write-Progress `
            -Activity "正在启动 PanelTone" `
            -Status "$($ready.Count) / $($services.Count) 个本机服务已就绪" `
            -PercentComplete ([int](100 * $ready.Count / $services.Count))
        if ($ready.Count -eq $services.Count) {
            break
        }
        Start-Sleep -Seconds 2
    } while ((Get-Date) -lt $deadline)
    Write-Progress -Activity "正在启动 PanelTone" -Completed

    $failed = @($services | Where-Object { -not (Test-ServiceHealth $_.Port $_.Path) })
    if ($failed.Count -gt 0) {
        throw "以下服务未在时限内启动：$($failed.Name -join '、')"
    }
    Write-Host "PanelTone 已就绪：http://127.0.0.1:8765/" -ForegroundColor Green
} catch {
    Write-Host "PanelTone 启动失败：$($_.Exception.Message)" -ForegroundColor Red
    if (-not $WaitForInput) {
        throw
    }
} finally {
    if ($WaitForInput) {
        Read-Host "按 Enter 关闭窗口"
    }
}
