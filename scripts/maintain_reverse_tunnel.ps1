param(
    [Parameter(Mandatory = $true)]
    [ValidatePattern("^[A-Za-z0-9.-]+$")]
    [string]$GatewayHost,
    [Parameter(Mandatory = $true)]
    [ValidatePattern("^[A-Za-z0-9._-]+$")]
    [string]$GatewayUser,
    [Parameter(Mandatory = $true)]
    [string]$IdentityFile,
    [ValidateRange(1, 65535)]
    [int]$GatewaySshPort = 22,
    [ValidateRange(1024, 65535)]
    [int]$RemotePort = 18766,
    [ValidateRange(1, 65535)]
    [int]$LocalPort = 8765,
    [ValidateRange(1, 300)]
    [int]$ReconnectDelaySeconds = 5,
    [ValidateRange(1, 100)]
    [int]$MaxLogSizeMiB = 5,
    [ValidateRange(1, 10)]
    [int]$LogBackups = 4,
    [string]$LogDirectory = "",
    [string]$SshExecutable = "ssh.exe",
    [switch]$ValidateOnly
)

$ErrorActionPreference = "Stop"

if (-not $LogDirectory) {
    $localAppData = [Environment]::GetFolderPath("LocalApplicationData")
    if (-not $localAppData -and $env:USERPROFILE) {
        $localAppData = Join-Path $env:USERPROFILE "AppData\Local"
    }
    if (-not $localAppData) {
        throw "Unable to resolve the current user's local application-data directory"
    }
    $LogDirectory = Join-Path $localAppData "PanelTone\logs\tunnel"
}
$resolvedLogDirectory = [IO.Path]::GetFullPath($LogDirectory)
New-Item -ItemType Directory -Path $resolvedLogDirectory -Force | Out-Null
$logPath = Join-Path $resolvedLogDirectory "reverse-tunnel.log"
$maxLogBytes = $MaxLogSizeMiB * 1MB

try {
    $resolvedIdentity = (Resolve-Path -LiteralPath $IdentityFile).Path
    $systemSsh = Join-Path $env:SystemRoot "System32\OpenSSH\ssh.exe"
    $resolvedSsh = if ($SshExecutable -eq "ssh.exe" -and (Test-Path $systemSsh)) {
        $systemSsh
    } else {
        (Get-Command $SshExecutable -ErrorAction Stop).Source
    }
} catch {
    Add-Content -LiteralPath $logPath -Encoding utf8 -Value (
        "{0:o} startup_error {1}" -f (Get-Date), $_.Exception.Message
    )
    throw
}

function Rotate-TunnelLog {
    if (-not (Test-Path -LiteralPath $logPath)) {
        return
    }
    if ((Get-Item -LiteralPath $logPath).Length -lt $maxLogBytes) {
        return
    }
    $oldest = "$logPath.$LogBackups"
    if (Test-Path -LiteralPath $oldest) {
        Remove-Item -LiteralPath $oldest -Force
    }
    for ($index = $LogBackups - 1; $index -ge 1; $index--) {
        $source = "$logPath.$index"
        if (Test-Path -LiteralPath $source) {
            Move-Item -LiteralPath $source -Destination "$logPath.$($index + 1)" -Force
        }
    }
    Move-Item -LiteralPath $logPath -Destination "$logPath.1" -Force
}

$forward = "127.0.0.1:$RemotePort`:127.0.0.1:$LocalPort"
$sshArguments = @(
    "-N",
    "-T",
    "-p", [string]$GatewaySshPort,
    "-i", $resolvedIdentity,
    "-o", "BatchMode=yes",
    "-o", "IdentitiesOnly=yes",
    "-o", "ExitOnForwardFailure=yes",
    "-o", "ServerAliveInterval=30",
    "-o", "ServerAliveCountMax=3",
    "-o", "ConnectTimeout=15",
    "-o", "ForwardAgent=no",
    "-o", "ForwardX11=no",
    "-o", "RequestTTY=no",
    "-R", $forward,
    "$GatewayUser@$GatewayHost"
)

if ($ValidateOnly) {
    [pscustomobject]@{
        status = "valid"
        ssh = $resolvedSsh
        remote_bind = "127.0.0.1:$RemotePort"
        local_target = "127.0.0.1:$LocalPort"
        log_directory = $resolvedLogDirectory
    }
    exit 0
}

while ($true) {
    Rotate-TunnelLog
    Add-Content -LiteralPath $logPath -Encoding utf8 -Value (
        "{0:o} connecting remote=127.0.0.1:{1} local=127.0.0.1:{2}" -f (
            Get-Date
        ), $RemotePort, $LocalPort
    )
    & $resolvedSsh @sshArguments 2>&1 | ForEach-Object {
        Add-Content -LiteralPath $logPath -Encoding utf8 -Value (
            "{0:o} ssh {1}" -f (Get-Date), $_
        )
    }
    $exitCode = $LASTEXITCODE
    Add-Content -LiteralPath $logPath -Encoding utf8 -Value (
        "{0:o} disconnected exit_code={1}; reconnecting in {2}s" -f (
            Get-Date
        ), $exitCode, $ReconnectDelaySeconds
    )
    Start-Sleep -Seconds $ReconnectDelaySeconds
}
