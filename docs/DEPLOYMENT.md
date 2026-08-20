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

## Prerequisites on `irm-pc`

- Docker Desktop
- Node.js 18+
- Python 3.10–3.12
- Tailscale (already connected — `100.120.27.110`)
- `cloudflared` — `winget install Cloudflare.cloudflared`
- Git

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

## Running

```bat
start.bat
```

Brings up MySQL + Adminer (Docker), then the AI service, Node service, and client, each in its own window. It creates any missing `.env` files from the examples, installs Node dependencies if absent, and warns about port conflicts before launching. Database files land in `.\.data\mysql` on whatever drive the repo lives on.

Then, for public access:

```bat
tunnel.bat
```

Opens a Cloudflare quick tunnel for each of the three services and prints the assigned URLs.

## Wiring up the tunnel URLs

Quick-tunnel URLs change on every restart, so after each `tunnel.bat` run:

1. `client\web\.env.local` → `NEXT_PUBLIC_API_URL=https://<node-url>.trycloudflare.com`
2. `server\node-service\.env` → `CORS_ORIGIN=https://<client-url>.trycloudflare.com`
3. `server\node-service\.env` → `AI_SERVICE_URL=https://<ai-url>.trycloudflare.com`

Restart the Node service and client afterward so they pick up the changes.

> If you get tired of re-wiring these, a **named** Cloudflare tunnel with a real domain gives you a fixed URL. Quick tunnels were chosen deliberately here (no account or domain required) with this tradeoff accepted.

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
| MySQL up | `docker compose ps` — `roadsentinel-mysql` shows `healthy` |
| Database contents | Adminer at `http://localhost:8080` (server `mysql`, credentials per `docker-compose.yml`) |
| AI service + model | `curl http://localhost:8000/api/stats` — `traffic_model.loaded` should be `true` after the first detection; the startup log must say `custom_model=True` (`False` means it silently fell back to the untrained stock model) |
| Node service | `curl http://localhost:3001/health`; startup log should show `Database: Connected` and `AI Service: Connected` |
| Client | `http://localhost:3000` |
| Public status page | `http://localhost:3000/status` |
| Pis online | Admin Terminal in the dashboard — both should show ONLINE |

## Known blockers

- **Tailscale SSH is not permitted by the tailnet ACL.** `ssh <user>@100.98.53.95` returns `tailnet policy does not permit you to SSH as user "..."` for every username tried. Network reachability is fine (ping works, services are reachable by IP:port) — this only blocks interactive shell access. Fix by adding an `ssh` section to the Access Controls in the Tailscale admin console.
- **SSH into `irm-pc` requires an authorized key.** The `ivanraybagnol` account exists but rejects key and password auth from outside. To allow remote administration from the development laptop, add its public key to `irm-pc`'s `C:\ProgramData\ssh\administrators_authorized_keys` (with the correct restrictive ACLs) or the user's `~\.ssh\authorized_keys`:

  ```
  ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIAT2XcxaYhbAB17iHfpUEUIY5Nsim7W4Rgk7kjulm8XS shaloh@MinnieDumpor
  ```

  Until then, everything on `irm-pc` has to be run at the machine itself.

## Disk space

MySQL data is a bind mount at `.\.data\mysql`, and recorded video segments land in `server\ai-service\media\recordings\` — both on whatever drive the repo is cloned to. Detections and recordings only grow, so clone the repo to the drive with the most free space and check it periodically.
