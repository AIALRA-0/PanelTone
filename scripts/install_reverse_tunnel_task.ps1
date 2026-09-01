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
    [string]$TaskName = "PanelTone Private Tunnel"
)

$ErrorActionPreference = "Stop"
$scriptPath = (Resolve-Path (Join-Path $PSScriptRoot "maintain_reverse_tunnel.ps1")).Path
$resolvedIdentity = (Resolve-Path -LiteralPath $IdentityFile).Path

function Quote-TaskArgument([string]$Value) {
    return '"' + $Value.Replace('"', '`"') + '"'
}

$arguments = @(
    "-NoProfile",
    "-NonInteractive",
    "-WindowStyle", "Hidden",
    "-ExecutionPolicy", "Bypass",
    "-File", (Quote-TaskArgument $scriptPath),
    "-GatewayHost", (Quote-TaskArgument $GatewayHost),
    "-GatewayUser", (Quote-TaskArgument $GatewayUser),
    "-IdentityFile", (Quote-TaskArgument $resolvedIdentity),
    "-GatewaySshPort", [string]$GatewaySshPort,
    "-RemotePort", [string]$RemotePort,
    "-LocalPort", [string]$LocalPort
) -join " "

$action = New-ScheduledTaskAction -Execute "powershell.exe" -Argument $arguments
$currentUser = [System.Security.Principal.WindowsIdentity]::GetCurrent().Name
$trigger = New-ScheduledTaskTrigger -AtLogOn -User $currentUser
$settings = New-ScheduledTaskSettingsSet `
    -AllowStartIfOnBatteries `
    -DontStopIfGoingOnBatteries `
    -ExecutionTimeLimit ([TimeSpan]::Zero) `
    -MultipleInstances IgnoreNew `
    -RestartCount 999 `
    -RestartInterval (New-TimeSpan -Minutes 1) `
    -StartWhenAvailable
$principal = New-ScheduledTaskPrincipal `
    -UserId $currentUser `
    -LogonType Interactive `
    -RunLevel Limited

Register-ScheduledTask `
    -TaskName $TaskName `
    -Action $action `
    -Trigger $trigger `
    -Settings $settings `
    -Principal $principal `
    -Description "Maintains the private PanelTone reverse SSH tunnel" `
    -Force | Out-Null
Start-ScheduledTask -TaskName $TaskName
Get-ScheduledTask -TaskName $TaskName | Select-Object TaskName, State
