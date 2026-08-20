@echo off
setlocal enabledelayedexpansion
cd /d "%~dp0"

echo ============================================================
echo  Road Sentinel - local dev stack
echo ============================================================
echo.

REM ---- 1. Docker Desktop ------------------------------------------------
echo [1/6] Checking Docker Desktop...
docker info >nul 2>&1
if not errorlevel 1 goto dockerready

echo       Not running yet - attempting to start Docker Desktop...
if exist "%ProgramFiles%\Docker\Docker\Docker Desktop.exe" goto launchdocker
echo       Could not find Docker Desktop.exe automatically.
echo       Start Docker Desktop yourself, then re-run this script.
pause
exit /b 1

:launchdocker
start "" "%ProgramFiles%\Docker\Docker\Docker Desktop.exe"
echo       Waiting for the Docker engine to come up (up to ~90s)...
set /a tries=0

:waitdocker
set /a tries+=1
ping -n 4 127.0.0.1 >nul
docker info >nul 2>&1
if not errorlevel 1 goto dockerready
if !tries! GEQ 30 (
    echo       Docker still isn't responding. Start Docker Desktop
    echo       manually and re-run this script.
    pause
    exit /b 1
)
goto waitdocker

:dockerready
echo       Docker is up.
echo.

REM ---- 2. MySQL + Adminer via docker compose -----------------------------
echo [2/6] Starting local MySQL (port 3307) + Adminer (port 8080)...
docker compose up -d
if errorlevel 1 (
    echo       docker compose failed - see output above.
    pause
    exit /b 1
)

echo       Waiting for MySQL to become healthy...
set /a tries=0

:waitmysql
set /a tries+=1
set MYSQL_HEALTH=
for /f "tokens=*" %%h in ('docker inspect --format "{{.State.Health.Status}}" roadsentinel-mysql 2^>nul') do set MYSQL_HEALTH=%%h
if "!MYSQL_HEALTH!"=="healthy" goto mysqlready
if !tries! GEQ 40 (
    echo       MySQL did not become healthy in time. Check: docker logs roadsentinel-mysql
    pause
    exit /b 1
)
ping -n 3 127.0.0.1 >nul
goto waitmysql

:mysqlready
echo       MySQL is healthy.
echo.

REM ---- 3. Env files -------------------------------------------------------
echo [3/6] Checking .env files...
if not exist "server\node-service\.env" (
    copy "server\node-service\.env.example" "server\node-service\.env" >nul
    echo       Created server\node-service\.env from .env.example - edit it
    echo       to set your own JWT_SECRET / ADMIN_PASSWORD / PI_AGENT_TOKEN.
)
if not exist "server\ai-service\.env" (
    copy "server\ai-service\.env.example" "server\ai-service\.env" >nul
    echo       Created server\ai-service\.env from .env.example - set
    echo       TRAFFIC_MODEL_PATH to your trained weight before relying on it.
)
if not exist "client\web\.env.local" (
    echo NEXT_PUBLIC_API_URL=http://localhost:3001> "client\web\.env.local"
    echo       Created client\web\.env.local
)
echo       Done.
echo.

REM ---- 4. Node dependencies -------------------------------------------------
echo [4/6] Checking Node dependencies (skips if already installed)...
if not exist "server\node-service\node_modules" (
    echo       Installing server\node-service dependencies...
    pushd server\node-service
    call npm install
    popd
)
if not exist "client\web\node_modules" (
    echo       Installing client\web dependencies...
    pushd client\web
    call npm install
    popd
)
echo       Done.
echo.

REM ---- 5. AI service venv check ---------------------------------------------
echo [5/6] Checking AI service Python environment...
if exist "server\ai-service\venv\Scripts\python.exe" (
    echo       Found.
) else (
    echo       No venv found at server\ai-service\venv - the AI service window
    echo       below will likely fail to start. Set it up first:
    echo         cd server\ai-service
    echo         python -m venv venv
    echo         venv\Scripts\pip install -r requirements.txt   ^(or requirements-cpu.txt^)
)
echo.

REM ---- 5b. Port conflict warnings (informational only) -----------------------
for %%p in (3000 3001 8000) do (
    netstat -ano | findstr "LISTENING" | findstr ":%%p " >nul 2>&1
    if not errorlevel 1 (
        echo       WARNING: port %%p is already in use by another process.
        echo       The corresponding service window below may fail to start.
        echo       ^(Check with: netstat -ano ^| findstr :%%p^)
    )
)
echo.

REM ---- 6. Launch everything ---------------------------------------------------
echo [6/6] Launching services in separate windows...
start "RoadSentinel - AI Service (:8000)" cmd /k "cd /d %~dp0server\ai-service && venv\Scripts\python.exe -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload"
start "RoadSentinel - Node Service (:3001)" cmd /k "cd /d %~dp0server\node-service && npm run dev"
start "RoadSentinel - Client (:3000)" cmd /k "cd /d %~dp0client\web && npm run dev"

echo.
echo ============================================================
echo  Everything is starting up. Give it 10-15s, then:
echo ============================================================
echo.
echo   Dashboard            http://localhost:3000
echo   Public status page   http://localhost:3000/status
echo   Node API             http://localhost:3001
echo   AI service           http://localhost:8000
echo   Adminer (DB GUI)     http://localhost:8080
echo.
echo   Adminer login:
echo     System    MySQL
echo     Server    mysql
echo     Username  roadsentinel
echo     Password  roadsentinel_dev
echo     Database  roadsentinel
echo.
echo   Admin Terminal login (sidebar -^> Admin Terminal):
echo     password = ADMIN_PASSWORD in server\node-service\.env
echo.
echo   These MySQL/Adminer credentials are local-dev-only defaults set in
echo   docker-compose.yml - never used for the real irm-pc production DB.
echo ============================================================
echo.

ping -n 11 127.0.0.1 >nul
start http://localhost:3000
start http://localhost:8080

echo Press any key to close this launcher window (the 3 service windows
echo and the Docker containers keep running independently).
pause >nul
