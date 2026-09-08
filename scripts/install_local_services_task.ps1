param(
    [string]$ProjectRoot = "",
    [string]$CandidateRoot = "",
    [string]$SemanticModelRoot = "",
    [string]$DataRoot = "",
    [string]$ModelRoot = "",
    [string]$TaskName = "PanelTone Local Services"
)

$ErrorActionPreference = "Stop"
$scriptPath = (Resolve-Path (Join-Path $PSScriptRoot "maintain_local_services.ps1")).Path
if (-not $ProjectRoot) {
    $ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
}
if (-not $CandidateRoot) {
    $CandidateRoot = Join-Path ([Environment]::GetFolderPath("LocalApplicationData")) `
        "PanelTone\candidates"
}
if (-not $SemanticModelRoot) {
    $SemanticModelRoot = Join-Path ([Environment]::GetFolderPath("LocalApplicationData")) `
        "PanelTone\models\semantic\koharu-yolo26s"
}
if (-not $DataRoot) {
    $DataRoot = Join-Path ([Environment]::GetFolderPath("LocalApplicationData")) `
        "PanelTone\jobs"
}
if (-not $ModelRoot) {
    $ModelRoot = Join-Path ([Environment]::GetFolderPath("LocalApplicationData")) `
        "PanelTone\models"
}

function Quote-TaskArgument([string]$Value) {
    return '"' + $Value.Replace('"', '`"') + '"'
}

$arguments = @(
    "-NoProfile",
    "-NonInteractive",
    "-WindowStyle", "Hidden",
    "-ExecutionPolicy", "Bypass",
    "-File", (Quote-TaskArgument $scriptPath),
    "-ProjectRoot", (Quote-TaskArgument ([IO.Path]::GetFullPath($ProjectRoot))),
    "-CandidateRoot", (Quote-TaskArgument ([IO.Path]::GetFullPath($CandidateRoot))),
    "-SemanticModelRoot", (Quote-TaskArgument ([IO.Path]::GetFullPath($SemanticModelRoot))),
    "-DataRoot", (Quote-TaskArgument ([IO.Path]::GetFullPath($DataRoot))),
    "-ModelRoot", (Quote-TaskArgument ([IO.Path]::GetFullPath($ModelRoot)))
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
    -Description "Maintains the local loopback-only PanelTone services" `
    -Force | Out-Null
Start-ScheduledTask -TaskName $TaskName
Get-ScheduledTask -TaskName $TaskName | Select-Object TaskName, State
