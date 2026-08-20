# Road Sentinel - re-read the current Cloudflare quick-tunnel URLs and rewire
# the services to match.
#
# Quick-tunnel URLs are regenerated every time cloudflared restarts, so after
# any tunnel restart (including a reboot) the client is pointing at a dead API
# URL and Node's CORS allowlist names a client origin that no longer exists.
# The visible symptom is "cannot reach the API" in the browser, while curl
# against the API still returns 200 — because the request only fails in a
# browser, on the missing Access-Control-Allow-Origin header.
#
# Run after tunnels restart:
#   powershell -ExecutionPolicy Bypass -File rewire_tunnels.ps1

$ErrorActionPreference = "Continue"

$Repo = "D:\RoadSentinel"
$Dir  = "$Repo\.tunnels"

function Get-TunnelUrl([string]$name) {
    $log = "$Dir\$name.log"
    if (-not (Test-Path $log)) { return $null }
    $m = Select-String -Path $log -Pattern "https://[a-z0-9-]+\.trycloudflare\.com" | Select-Object -First 1
    if ($m) { return $m.Matches[0].Value }
    return $null
}

$client = Get-TunnelUrl "client"
$node   = Get-TunnelUrl "node"
$ai     = Get-TunnelUrl "ai"

Write-Output "=== current tunnel URLs ==="
Write-Output "  client = $client"
Write-Output "  node   = $node"
Write-Output "  ai     = $ai"

if (-not $client -or -not $node) {
    Write-Output ""
    Write-Output "ERROR: client and/or node tunnel URL not found."
    Write-Output "Are the tunnels running?  schtasks /query /tn RoadSentinel-Tunnel-node"
    exit 1
}

Write-Output ""
Write-Output "=== rewiring ==="

# Client calls the Node API through its tunnel.
"NEXT_PUBLIC_API_URL=$node" | Set-Content -Path "$Repo\client\web\.env.local" -Encoding ASCII
Write-Output "  client/.env.local -> $node"

# Node must allow the client's tunnel origin, or the browser blocks every call.
# AI_SERVICE_URL deliberately stays on localhost: Node and the AI service are on
# the same machine, so routing that hop out through Cloudflare and back would
# add latency for nothing.
$envPath = "$Repo\server\node-service\.env"
$c = Get-Content $envPath
$c = $c -replace '^CORS_ORIGIN=.*', "CORS_ORIGIN=$client,http://localhost:3000"
Set-Content -Path $envPath -Value $c -Encoding ASCII
Write-Output "  node CORS_ORIGIN  -> $client,http://localhost:3000"

Write-Output ""
Write-Output "=== restarting node + client ==="
# Stop-ScheduledTask alone is NOT enough: the task launches npm, which spawns
# nodemon, which spawns node. Stopping the task leaves that child chain running
# with the OLD environment, so the new CORS_ORIGIN never takes effect and the
# browser keeps failing. Kill the node processes explicitly.
Stop-ScheduledTask -TaskName RoadSentinel-Node   -ErrorAction SilentlyContinue
Stop-ScheduledTask -TaskName RoadSentinel-Client -ErrorAction SilentlyContinue
Start-Sleep -Seconds 3
Get-Process node -ErrorAction SilentlyContinue | Stop-Process -Force -ErrorAction SilentlyContinue
Start-Sleep -Seconds 3

Start-ScheduledTask -TaskName RoadSentinel-Node
Start-ScheduledTask -TaskName RoadSentinel-Client
Write-Output "  restarted, waiting for them to come up..."
Start-Sleep -Seconds 45

foreach ($pt in @(3000, 3001)) {
    $ok = Test-NetConnection 127.0.0.1 -Port $pt -InformationLevel Quiet -WarningAction SilentlyContinue
    Write-Output "  port ${pt}: $ok"
}

Write-Output ""
Write-Output "=== verifying CORS header ==="
try {
    $r = Invoke-WebRequest -Uri "$node/api/analytics/summary" -Headers @{ "Origin" = $client } -UseBasicParsing -TimeoutSec 30
    $acao = $r.Headers["Access-Control-Allow-Origin"]
    if ($acao) { Write-Output "  OK - Access-Control-Allow-Origin: $acao" }
    else { Write-Output "  WARNING - no Access-Control-Allow-Origin header; the browser will block calls" }
} catch {
    Write-Output ("  could not verify: " + $_.Exception.Message)
}

Write-Output ""
Write-Output "=== YOUR URLS ==="
Write-Output "  Dashboard      $client"
Write-Output "  Admin Terminal $client/admin"
Write-Output "  Public Status  $client/status"
Write-Output "  Node API       $node"
Write-Output "  AI Service     $ai"
