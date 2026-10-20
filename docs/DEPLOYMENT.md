# Deployment — hosting Road Sentinel on `irm-pc`

`irm-pc` is the deployment target: it runs the AI service, the Node service, the Next.js client, MySQL, and the Cloudflare tunnels. The Raspberry Pis connect back to it over Tailscale; the public reaches it over Cloudflare quick tunnels.

| Node | Tailscale IP | Role |
|---|---|---|
| `irm-pc` | `100.120.27.110` | **Host** — all services, MySQL, tunnels, trained models |
| `road-sentinel-pi4` | `100.98.53.95` | Camera A + LED matrix |
| `road-sentinel-pi5` | `100.94.18.9` | Camera B + LED matrix |
| `minniedumpor` | `100.111.57.42` | Development laptop |

## Why this split

- **Pi → server traffic uses Tailscale**, not the Cloudflare tunnel. Quick-tunnel URLs are regenerated on every restart, which would break the Pis' hardcoded service URLs each time. Tailscale IPs are stable.
- **Browser/public access uses the Cloudflare tunnel.** No port forwarding, no static IP, no domain needed.
- **MySQL is never exposed** — bound to `127.0.0.1` only, reachable neither through Tailscale nor the tunnel. The Node service talks to it over loopback on the same machine.

## `irm-pc` machine facts

Verified directly over SSH, not assumed:

| | |
|---|---|
| GPU | **NVIDIA RTX 3060 Ti, 8 GB** — the same GPU the vehicle model was trained on, so CUDA inference is available |
| Disks | **C: ~11 GB free** (nearly full — avoid installing anything large here), **D: ~292 GB free**, **E: ~465 GB free** |
| Repo location | `D:\RoadSentinel` — on D: deliberately, so `.data\`, `media\recordings\`, `node_modules\`, and the AI venv all land on the drive with space |
| Python | 3.14 is the default `python`, which **PyTorch and ultralytics do not support**. Use `py -3.12` — Python 3.12.10 is installed and is what the AI service venv is built from |
| Docker | **Not installed.** MySQL runs natively instead (see below) rather than via `docker-compose.yml`, because Docker Desktop plus its WSL2 image would not fit comfortably in C:'s remaining space |
| Node / Git / cloudflared | All installed and on PATH |
| Oracle downloads | `dev.mysql.com` returns **403 Forbidden** from this machine's IP (works fine elsewhere), so the MySQL ZIP has to be fetched on another machine and copied over |

## Prerequisites

- Node.js 18+ ✅
- Python 3.12 (via `py -3.12`) ✅
- Tailscale, connected as `100.120.27.110` ✅
- `cloudflared` ✅
- Git ✅
- MySQL — installed natively from the ZIP distribution, see below

## First-time setup

```bat
git clone https://github.com/Shaloh69/Road_Sentinel.git
cd Road_Sentinel
```

**Place the trained model weights.** `*.pt` files are gitignored, so they don't come with the clone. Copy the trained vehicle model to:

```
models\runs\vehicle\vehicle_yolo26n_20260203_032528\weights\best.pt
```

Then point `server\ai-service\.env` at it:

```env
TRAFFIC_MODEL_PATH=../../models/runs/vehicle/vehicle_yolo26n_20260203_032528/weights/best.pt
DEVICE=cuda          # or cpu if this machine has no NVIDIA GPU
```

**Set the secrets** in `server\node-service\.env` (generate fresh ones — do not reuse the development laptop's):

```env
JWT_SECRET=<node -e "console.log(require('crypto').randomBytes(48).toString('hex'))">
ADMIN_PASSWORD=<a real password>
PI_AGENT_TOKEN=<a random shared secret — must match both Pis>
```

**Create the AI service venv:**

```bat
cd server\ai-service
python -m venv venv
venv\Scripts\pip install -r requirements.txt
cd ..\..
```

## MySQL on `irm-pc` (native, not Docker)

Docker isn't installed and C: is nearly full, so MySQL runs as a native Windows service from the ZIP distribution, with everything on D:

| | |
|---|---|
| Install root | `D:\RoadSentinel-mysql` |
| Data directory | `D:\RoadSentinel-mysql\data` |
| Config | `D:\RoadSentinel-mysql\my.ini` |
| Port | **3307** (matching the dev machine's Docker setup, so `.env` files are portable) |
| Bind address | **127.0.0.1 only** — never reachable over Tailscale or the tunnel |
| Windows service | `RoadSentinelMySQL` |
| Database / user | `roadsentinel` / `roadsentinel` |

Service control:

```bat
net start RoadSentinelMySQL
net stop RoadSentinelMySQL
```

Because `dev.mysql.com` returns 403 from this machine, the ZIP must be downloaded elsewhere and copied across (e.g. `scp mysql-winx64.zip "user@100.120.27.110:D:/"`).

> `docker-compose.yml` at the repo root is the **development-machine** path — MySQL 8.0 + Adminer in containers, data bind-mounted to `.\.data\mysql`. It is not used on `irm-pc`. Both setups deliberately use port 3307 and the same credentials so `.env` files work unchanged on either.

## Running

```bat
start.bat
```

Starts the AI service, Node service, and client, each in its own window, after scaffolding any missing `.env` files, installing absent Node dependencies, and warning about port conflicts.

> **On `irm-pc`, start MySQL first** (`net start RoadSentinelMySQL`) — `start.bat`'s Docker/compose steps are for the development machine and will report Docker as missing here. The three service launches still work.

Then, for public access:

```bat
tunnel.bat
```

Opens a Cloudflare quick tunnel for each of the three services and prints the assigned URLs.

## Wiring up the tunnel URLs

Quick-tunnel URLs are regenerated **every time cloudflared restarts** — including on reboot. After any tunnel restart, just run:

```bat
powershell -ExecutionPolicy Bypass -File rewire_tunnels.ps1
```

It reads the current URLs out of `.tunnels\*.log`, updates the client's `NEXT_PUBLIC_API_URL` and Node's `CORS_ORIGIN`, restarts both services, verifies the CORS header actually comes back, and prints your URLs.

### Two traps this script exists to avoid

**1. "Cannot reach the API" in the browser while `curl` returns 200.**
If Node's `CORS_ORIGIN` doesn't name the client's *current* tunnel origin, responses come back without an `Access-Control-Allow-Origin` header. `curl` doesn't care and shows a healthy `200`; a browser blocks the request entirely. So the API looks fine from the command line while the dashboard appears completely broken. To check for it directly:

```bat
curl -s -i -H "Origin: <client-url>" <node-url>/api/analytics/summary
```

If there's no `Access-Control-Allow-Origin` line in the response, that's the problem.

**3. The script used to pick the *oldest* tunnel URL.** (Fixed 2026-09-04.)

`cloudflared` **appends** to the same `.tunnels\*.log` file on every restart, so
a log accumulates every URL that tunnel has ever had. `Get-TunnelUrl` took
`Select-Object -First 1`, which is the oldest — a hostname from a tunnel that no
longer exists.

The failure is quiet and looks like success: both ports come up, the script
prints a full set of URLs, and the only hint is one line reading
`could not verify: The remote name could not be resolved` while the dashboard
is unreachable. It now takes `-Last 1`.

If you ever see the script report URLs that don't match the ones the tunnel
tasks just printed, that is this bug returning — check the logs have not been
hand-edited, or clear `.tunnels\*.log` before restarting the tunnels.

**2. `Stop-ScheduledTask` doesn't actually stop Node.**
The task launches `npm`, which spawns `nodemon`, which spawns `node`. Stopping the task leaves that child chain alive holding the **old** environment — so an edited `.env` silently has no effect and the symptom above persists through what looks like a restart. The node processes have to be killed explicitly:

```powershell
Get-Process node | Stop-Process -Force
```

> `AI_SERVICE_URL` deliberately stays on `http://localhost:8000`. Node and the AI service run on the same machine, so routing that hop out through Cloudflare and back would add latency for nothing. The AI tunnel exists only for external testing.

> If re-wiring after every restart gets tedious, a **named** Cloudflare tunnel with a real domain gives a fixed URL. Quick tunnels were chosen deliberately here (no account or domain needed), with this tradeoff accepted.

## Setting up the Raspberry Pis

Run on each Pi (they default to `irm-pc`'s Tailscale address, so no arguments needed):

```bash
PI_AGENT_TOKEN=<same token as node-service/.env> bash setup_pi4.sh   # on the Pi 4
PI_AGENT_TOKEN=<same token as node-service/.env> bash setup_pi5.sh   # on the Pi 5
```

Each installs three systemd services: `roadsentinel-camera`, `roadsentinel-display`, `roadsentinel-agent`.

To override the server address (e.g. to use a tunnel URL instead of Tailscale):

```bash
PI_AGENT_TOKEN=<token> bash setup_pi4.sh https://<node-url>.trycloudflare.com "" https://<ai-url>.trycloudflare.com
```

## Verifying

| Check | How |
|---|---|
| MySQL up | On `irm-pc`: `sc query RoadSentinelMySQL` shows `RUNNING` and port 3307 is listening. (Dev machine: `docker compose ps` — `roadsentinel-mysql` shows `healthy`) |
| Database contents | Dev machine only: Adminer at `http://localhost:8080` (server `mysql`, credentials per `docker-compose.yml`) |
| Cameras streaming | `curl <node-url>/api/public/status` — `cameras_online` should equal `cameras_total`. On a Pi, `tail ~/roadsentinel/logs/camera.log`: `sent=` must keep rising; a rising `dropped=` with a flat `sent=` means the camera works but the Pi's uplink to the server does not |
| AI service + model | `curl http://localhost:8000/api/stats` — `traffic_model.loaded` should be `true` after the first detection; the startup log must say `custom_model=True` (`False` means it silently fell back to the untrained stock model) |
| Node service | `curl http://localhost:3001/health`; startup log should show `Database: Connected` and `AI Service: Connected` |
| Client | `http://localhost:3000` |
| Public status page | `http://localhost:3000/status` |
| Pis online | Admin Terminal in the dashboard — both should show ONLINE |

## Known blockers

- **The services do not come back after a reboot.** (Found 2026-10-20.) The scheduled tasks are registered with `LogonType: Interactive` — "run only when user is logged on". The `AtStartup` trigger fires at boot, *before* anyone has logged on, so it is skipped — and logging on afterwards does not re-fire it. `irm-pc` rebooted at 18:59 on 2026-10-18, someone logged on at 19:00, and every service (AI, Node, client, tunnels) still stayed down for two days; only MySQL (a real Windows service) came back. (An earlier version of this note blamed nobody logging on — the `query user` output disproved that.)

  **Fix:** `install_services.ps1` now registers with `LogonType S4U` ("run whether the user is logged on or not", no stored password). Re-run it elevated on `irm-pc`, and switch the three `RoadSentinel-Tunnel-*` tasks the same way (Task Scheduler → task → *Run whether user is logged on or not*). As of 2026-10-20 the AI/Node/client tasks are S4U (verified with `Get-ScheduledTask`); the three tunnel tasks are still `Interactive`. Even with it applied, a reboot rotates the quick-tunnel URLs, so `rewire_tunnels.ps1` still has to run afterwards.

  Until then, after any reboot: start the three service tasks, bring the tunnels up, then run `rewire_tunnels.ps1`.

## Remote access

SSH works from the development laptop (key-based, no prompt):

| Host | Command |
|---|---|
| `irm-pc` | `ssh "malubay ivan ray@100.120.27.110"` — note the spaces in the username; the shell is `cmd`, so run PowerShell scripts as `powershell -NoProfile -ExecutionPolicy Bypass -File <script>` |
| Pi 4 | `ssh roadsentinel@100.98.53.95` (sudo needs a password) |
| Pi 5 | `ssh raspi5@100.94.18.9` (passwordless sudo; Tailscale SSH may ask for a one-time browser re-auth) |

## Disk space

MySQL data is a bind mount at `.\.data\mysql`, and recorded video segments land in `server\ai-service\media\recordings\` — both on whatever drive the repo is cloned to. Detections and recordings only grow, so clone the repo to the drive with the most free space and check it periodically.
