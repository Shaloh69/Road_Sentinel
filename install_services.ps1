# Road Sentinel - register the three services as Windows Scheduled Tasks.
#
# Why scheduled tasks rather than `Start-Process`: a process started from an
# SSH session dies when that session ends, because it belongs to the session's
# job object. Scheduled tasks run detached from any login session and can also
# start automatically at boot, which is what a host machine needs.
#
# Run once, elevated:
#   powershell -ExecutionPolicy Bypass -File install_services.ps1
#
# Afterwards:
#   schtasks /run    /tn RoadSentinel-AI
#   schtasks /end    /tn RoadSentinel-AI
#   schtasks /query  /tn RoadSentinel-AI
#   schtasks /delete /tn RoadSentinel-AI /f

$ErrorActionPreference = "Stop"

$Repo = "D:\RoadSentinel"

# Use COMPUTERNAME, not USERDOMAIN. On a machine that isn't domain-joined,
# USERDOMAIN reports "WORKGROUP", and "WORKGROUP\user" is not a resolvable
# principal — Register-ScheduledTask fails with "No mapping between account
# names and security IDs was done." The local-account form is COMPUTERNAME\user,
# which is what `whoami` reports.
$User = "$env:COMPUTERNAME\$env:USERNAME"

function Register-RSTask {
    param(
        [string]$Name,
        [string]$Exe,
        [string]$Arguments,
        [string]$WorkDir
    )

    $action   = New-ScheduledTaskAction -Execute $Exe -Argument $Arguments -WorkingDirectory $WorkDir
    $trigger  = New-ScheduledTaskTrigger -AtStartup
    # RunOnlyIfNetworkAvailable is deliberately off: the AI service and MySQL
    # are local, and the Node service degrades gracefully without a network.
    $settings = New-ScheduledTaskSettingsSet `
        -AllowStartIfOnBatteries `
        -DontStopIfGoingOnBatteries `
        -StartWhenAvailable `
        -RestartCount 3 `
        -RestartInterval (New-TimeSpan -Minutes 1) `
        -ExecutionTimeLimit ([TimeSpan]::Zero)

    if (Get-ScheduledTask -TaskName $Name -ErrorAction SilentlyContinue) {
        Unregister-ScheduledTask -TaskName $Name -Confirm:$false
    }

    Register-ScheduledTask -TaskName $Name `
        -Action $action -Trigger $trigger -Settings $settings `
        -RunLevel Highest -User $User -Force | Out-Null

    Write-Output "  registered: $Name"
}

Write-Output "Registering Road Sentinel scheduled tasks..."

Register-RSTask -Name "RoadSentinel-AI" `
    -Exe "$Repo\server\ai-service\venv\Scripts\python.exe" `
    -Arguments "-m uvicorn app.main:app --host 0.0.0.0 --port 8000" `
    -WorkDir "$Repo\server\ai-service"

Register-RSTask -Name "RoadSentinel-Node" `
    -Exe "C:\Program Files\nodejs\npm.cmd" `
    -Arguments "run dev" `
    -WorkDir "$Repo\server\node-service"

Register-RSTask -Name "RoadSentinel-Client" `
    -Exe "C:\Program Files\nodejs\npm.cmd" `
    -Arguments "run dev" `
    -WorkDir "$Repo\client\web"

Write-Output ""
Write-Output "Starting them now..."
foreach ($t in @("RoadSentinel-AI", "RoadSentinel-Node", "RoadSentinel-Client")) {
    Start-ScheduledTask -TaskName $t
    Write-Output "  started: $t"
}

Write-Output ""
Write-Output "Done. They will also start automatically at boot."
