# Cloudflare Self-Hosting Guide — AI Service (Free Tier)

This guide explains how to expose the **Road Sentinel AI Service** running on your local RTX 3060 Ti machine to the internet for **free** using **Cloudflare Tunnel** (`cloudflared`).

> **Why Cloudflare Tunnel?**
> The AI service needs a real GPU. Render/cloud services either don't have GPUs or are very expensive. Cloudflare Tunnel creates a secure, encrypted tunnel from your local machine to Cloudflare's global edge network — completely free, no port-forwarding, no static IP needed.

---

## Architecture Overview

```
[Your PC — RTX 3060 Ti]          [Cloudflare Edge]         [Render]
  FastAPI :8000  ──cloudflared──► *.trycloudflare.com  ◄──  Node Service
                                        or
                                 ai.yourdomain.com
                                 (free with Cloudflare DNS)
```

---

## Prerequisites

- Cloudflare account (free) — https://dash.cloudflare.com/sign-up
- `cloudflared` installed on the machine running the AI service
- AI service working locally (`bash start.sh` succeeds)

---

## Step 1 — Install cloudflared

### Windows
```powershell
# Option A: winget
winget install Cloudflare.cloudflared

# Option B: direct download
# https://github.com/cloudflare/cloudflared/releases/latest
# Download cloudflared-windows-amd64.exe, rename to cloudflared.exe
# Add to PATH or run from the folder
```

### Linux / WSL2
```bash
# Debian / Ubuntu
curl -L https://pkg.cloudflare.com/cloudflare-main.gpg | sudo tee /usr/share/keyrings/cloudflare-main.gpg >/dev/null
echo 'deb [signed-by=/usr/share/keyrings/cloudflare-main.gpg] https://pkg.cloudflare.com/cloudflared any main' | sudo tee /etc/apt/sources.list.d/cloudflared.list
sudo apt update && sudo apt install cloudflared

# Or via direct binary
wget -q https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64 -O cloudflared
chmod +x cloudflared && sudo mv cloudflared /usr/local/bin/
```

---

## Option A — Quick Tunnel (No Account Setup, Temporary URL)

Best for: **testing and development**. URL changes every time you restart.

```bash
# 1. Start the AI service
cd server/ai-service
bash start.sh

# 2. In a new terminal, open the tunnel
cloudflared tunnel --url http://localhost:8000
```

Output will show something like:
```
+--------------------------------------------------------------------------------------------+
|  Your quick Tunnel has been created! Visit it at (it may take some time to be reachable):  |
|  https://whatever-random-name.trycloudflare.com                                            |
+--------------------------------------------------------------------------------------------+
```

Copy the URL and set it in the **Node Service** (Render):
```
AI_SERVICE_URL=https://whatever-random-name.trycloudflare.com
```

---

## Option B — Named Tunnel with Custom Domain (Permanent URL)

Best for: **production**. URL never changes. Requires a domain on Cloudflare DNS (free).

### 1. Authenticate cloudflared
```bash
cloudflared tunnel login
# Opens browser — log in and pick your domain
```

### 2. Create the tunnel
```bash
cloudflared tunnel create road-sentinel-ai
# Saves credentials to ~/.cloudflared/<UUID>.json
# Note the Tunnel ID shown in the output
```

### 3. Create the config file

Create `~/.cloudflared/config.yml` (or `C:\Users\<you>\.cloudflared\config.yml` on Windows):

```yaml
tunnel: <YOUR_TUNNEL_ID>
credentials-file: /home/<you>/.cloudflared/<UUID>.json   # adjust path

ingress:
  - hostname: ai.yourdomain.com
    service: http://localhost:8000
  - service: http_status:404
```

### 4. Add DNS record (one-time)
```bash
cloudflared tunnel route dns road-sentinel-ai ai.yourdomain.com
# Adds a CNAME in Cloudflare DNS pointing to the tunnel — no cost
```

### 5. Run the tunnel
```bash
cloudflared tunnel run road-sentinel-ai
```

Your AI service is now permanently available at:
```
https://ai.yourdomain.com
```

Set this in the **Node Service** (Render):
```
AI_SERVICE_URL=https://ai.yourdomain.com
```

---

## Option C — Run cloudflared as a Windows Service (Auto-start on Boot)

So the tunnel starts automatically whenever your PC boots, without you doing anything:

```powershell
# Run once as Administrator
cloudflared service install
# Now cloudflared starts automatically with Windows
```

To uninstall:
```powershell
cloudflared service uninstall
```

---

## Step 2 — Configure the Node Service on Render

In your Render **node-service** environment variables, set:

| Variable | Value |
|---|---|
| `AI_SERVICE_URL` | `https://ai.yourdomain.com` (or the trycloudflare URL) |
| `AI_SERVICE_TIMEOUT` | `30000` |

The node service will now call your local RTX 3060 Ti through Cloudflare for every detection request.

---

## Step 3 — Verify End-to-End

```bash
# From anywhere on the internet:
curl https://ai.yourdomain.com/health
# Expected: {"status": "healthy", "timestamp": 1234567890.0}

curl https://ai.yourdomain.com/api/stats
# Expected: model paths, device, confidence threshold
```

Or from the Render dashboard, check the node-service logs for:
```
AI Service: Connected
```

---

## Security — Restrict Access to Cloudflare Only (Recommended)

By default the tunnel URL is public. To restrict it so **only your Render node service** can call it:

### Option 1: Cloudflare Access (Zero Trust) — Free up to 50 users

1. Go to **Cloudflare Zero Trust → Access → Applications**
2. Add an application for `ai.yourdomain.com`
3. Set a **Service Token** policy
4. Generate a Service Token (`Client ID` + `Client Secret`)
5. Add headers to the node service AI calls:

In [server/node-service/src/services/ai.service.ts](src/services/ai.service.ts), update the axios client:

```typescript
this.client = axios.create({
  baseURL: this.baseURL,
  timeout: this.timeout,
  headers: {
    'CF-Access-Client-Id': process.env.CF_ACCESS_CLIENT_ID || '',
    'CF-Access-Client-Secret': process.env.CF_ACCESS_CLIENT_SECRET || '',
  },
});
```

Add to Render env vars:
```
CF_ACCESS_CLIENT_ID=your-client-id.access
CF_ACCESS_CLIENT_SECRET=your-client-secret
```

### Option 2: Simple API Key Header (Quick)

In [server/ai-service/app/main.py](app/main.py), add a middleware:

```python
from fastapi import Request, HTTPException

@app.middleware("http")
async def verify_api_key(request: Request, call_next):
    # Allow health check without key
    if request.url.path == "/health":
        return await call_next(request)
    key = request.headers.get("X-API-Key")
    if key != os.getenv("API_KEY"):
        return JSONResponse(status_code=403, content={"error": "Forbidden"})
    return await call_next(request)
```

Add `API_KEY=your-secret-key` to `server/ai-service/.env`, and add the header in the node service.

---

## Deployment Summary

| Component | Where | Cost |
|---|---|---|
| AI Service (FastAPI + YOLO) | Your PC (RTX 3060 Ti) | Free (electricity) |
| Cloudflare Tunnel | Cloudflare Edge | Free |
| Node Service (Express + Socket.IO) | Render | Free tier |
| Web Dashboard (Next.js) | Render Static Sites | Free tier |
| Database | Aiven MySQL | Free tier (up to 5GB) |
| File Storage | Supabase Storage | Free tier (1GB) |

**Total cost: $0/month** for a fully operational production system.

---

## Troubleshooting

### Tunnel connects but AI service returns 502
- Make sure the AI service is actually running: `curl http://localhost:8000/health`
- Check that `PORT=8000` matches what cloudflared is tunneling

### Models not loading (traffic.pt / incident.pt not found)
- Train the models first: see [training/README.md](../../training/README.md)
- Or set `TRAFFIC_MODEL_PATH` to point to an existing `yolov8n.pt` for testing (will use COCO fallback)

### Cloudflare tunnel drops after PC sleeps
- Disable sleep in Windows Power Settings → or install as a service (Option C)

### Node service on Render can't reach tunnel (timeout)
- Quick tunnels time out after ~30 min of inactivity — use a Named Tunnel (Option B) for production
- Increase `AI_SERVICE_TIMEOUT` in Render env vars if requests are slow
