@echo off
setlocal enabledelayedexpansion
cd /d "%~dp0"

REM Road Sentinel - Cloudflare Quick Tunnels
REM
REM Exposes the Node API and AI service to the public internet via Cloudflare
REM quick tunnels (no account, no domain, no port forwarding).
REM
REM IMPORTANT: quick-tunnel URLs are RANDOM and change every restart. After
REM running this, copy the printed URLs into:
REM   client/web/.env.local            NEXT_PUBLIC_API_URL=<node url>
REM   server/node-service/.env         AI_SERVICE_URL=<ai url>
REM                                     CORS_ORIGIN=<client url, if tunneling it too>
REM ...then restart those services so they pick up the change.
REM
REM MySQL is deliberately NOT tunneled - it stays bound to localhost only.

echo ============================================================
echo  Road Sentinel - Cloudflare Quick Tunnels
echo ============================================================
echo.

REM Find cloudflared: PATH first, then the standard winget install locations
REM (winget adds it to PATH, but existing shells do not pick that up until
REM they are restarted - so check the known paths too rather than failing).
set "CFD="
where cloudflared >nul 2>&1
if not errorlevel 1 set "CFD=cloudflared"
if not defined CFD if exist "%ProgramFiles(x86)%\cloudflared\cloudflared.exe" set "CFD=%ProgramFiles(x86)%\cloudflared\cloudflared.exe"
if not defined CFD if exist "%ProgramFiles%\cloudflared\cloudflared.exe" set "CFD=%ProgramFiles%\cloudflared\cloudflared.exe"

if not defined CFD (
    echo ERROR: cloudflared is not installed or could not be found.
    echo   Install it with:  winget install Cloudflare.cloudflared
    echo   Then re-run this script.
    pause
    exit /b 1
)

echo Using cloudflared: %CFD%
echo.

if not exist ".tunnels" mkdir ".tunnels"

echo Starting tunnels in separate windows...
echo.

start "RoadSentinel Tunnel - Node API (:3001)" cmd /k ""%CFD%" tunnel --url http://localhost:3001 --logfile "%~dp0.tunnels\node.log""
start "RoadSentinel Tunnel - AI Service (:8000)" cmd /k ""%CFD%" tunnel --url http://localhost:8000 --logfile "%~dp0.tunnels\ai.log""
start "RoadSentinel Tunnel - Client (:3000)" cmd /k ""%CFD%" tunnel --url http://localhost:3000 --logfile "%~dp0.tunnels\client.log""

echo Waiting for Cloudflare to assign URLs (about 20s)...
ping -n 21 127.0.0.1 >nul

echo.
echo ============================================================
echo  Your public URLs
echo ============================================================
echo.

for %%s in (node ai client) do (
    set "FOUND="
    for /f "tokens=*" %%u in ('findstr /c:"trycloudflare.com" ".tunnels\%%s.log" 2^>nul') do (
        if not defined FOUND (
            echo   %%s : %%u
            set "FOUND=1"
        )
    )
    if not defined FOUND echo   %%s : still starting - check the %%s tunnel window
)

echo.
echo ============================================================
echo  Next steps
echo ============================================================
echo.
echo  1. Copy the Node API URL into client\web\.env.local:
echo        NEXT_PUBLIC_API_URL=https://^<node-url^>.trycloudflare.com
echo.
echo  2. Copy the Client URL into server\node-service\.env:
echo        CORS_ORIGIN=https://^<client-url^>.trycloudflare.com
echo.
echo  3. Copy the AI URL into server\node-service\.env:
echo        AI_SERVICE_URL=https://^<ai-url^>.trycloudflare.com
echo.
echo  4. Restart the Node service and client so they pick up the changes.
echo.
echo  5. For the Raspberry Pis, re-run their setup with the tunnel URLs:
echo        PI_AGENT_TOKEN=^<token^> bash setup_pi4.sh https://^<node-url^>.trycloudflare.com "" https://^<ai-url^>.trycloudflare.com
echo     ...or keep using the Tailscale addresses, which do not change:
echo        http://100.120.27.110:3001  /  http://100.120.27.110:8000
echo.
echo  NOTE: these URLs are regenerated every time you restart the tunnels.
echo  Tailscale is the stable option for Pi-to-server traffic; the tunnels
echo  are for public/browser access from outside the tailnet.
echo.
echo  Full URLs are also in .tunnels\*.log
echo ============================================================
echo.
pause
