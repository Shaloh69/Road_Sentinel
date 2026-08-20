# Road Sentinel Revamp — Running Summary

Live record of the full revamp per `ROAD_SENTINEL_REVAMP_MASTER (1).md`. Updated continuously instead of pausing for go-ahead between phases, per instruction. Nothing is committed/pushed until Phase 5, and only if Phase 4's audit passes.

Legend: ✅ done & verified · 🟡 done, unverified (no hardware/infra access) · ⛔ blocked (needs your action) · ⏳ in progress

---

## Phase 0 — Critical security/safety fixes — ✅ COMPLETE

All 9 items done. Static verification (tsc/eslint/py_compile all clean) plus a **live smoke-test** of the auth system against a real running instance (isolated test port, not the pre-existing dev instance on 3001).

1. **Unauthenticated RCE — ✅ fixed, live-verified.**
   - `server/node-service/src/middleware/auth.ts` (new): JWT sign/verify, `requireAuth` HTTP middleware, `adminNamespaceAuth` Socket.IO middleware.
   - `src/routes/auth.ts` (new): `POST /api/auth/login` (password → JWT), 5-attempts/15min rate limiter.
   - `src/server.ts`: default `/` namespace now only carries public camera/incident streaming. All admin-terminal + Pi-agent-relay events (`terminal_command`, `terminal_kill`, `pi_register`, `pi_output`, `pi_command`, `pi_kill`) moved to a new `/admin` namespace gated by `adminNamespaceAuth` (accepts admin JWT or static `PI_AGENT_TOKEN`; rejects the connection outright otherwise).
   - `raspi_scripts/pi_agent.py`: connects to `/admin` with `auth={"token": PI_AGENT_TOKEN}`; refuses to start without one.
   - `client/web/lib/adminSocket.ts` (new) + `app/admin/page.tsx`: real login gate (password → JWT → sessionStorage → authenticated socket).
   - `raspi_scripts/setup_pi4.sh` / `setup_pi5.sh`: require `PI_AGENT_TOKEN`, wire into the `roadsentinel-agent` systemd unit.
   - **Live-verified**: wrong password → 401; correct → real JWT; `/admin` connect with no/garbage token → rejected; valid `PI_AGENT_TOKEN` → connects; valid admin JWT → connects and runs a real command end-to-end; old vulnerable path on the default namespace confirmed closed (emitting `terminal_command` there is now a no-op).
   - Real secrets generated into local `server/node-service/.env` (untracked): `JWT_SECRET`, `PI_AGENT_TOKEN`, `ADMIN_PASSWORD=bD03Y90AT2RoucP` (change anytime).

2. **CORS wildcard — ✅ fixed, live-verified.** Both Express and Socket.IO now check `Origin` against `CORS_ORIGIN` (comma-separated allowlist, default `http://localhost:3000`). Verified: disallowed origin gets no ACAO header; allowed one does.

3. **Plaintext secrets — ✅ redacted per your instruction.** `render.env.txt`'s DB password and Supabase service-role key replaced with `<rotate-and-paste-here>`. **⛔ Still needs you**: rotate both on Aiven/Supabase dashboards (see Phase 0.5 note below — Aiven may already be gone, see there).

4. **Hardcoded model path — ✅ fixed, verified.** `server/ai-service/app/main.py` resolves relative `TRAFFIC_MODEL_PATH`/`INCIDENT_MODEL_PATH` against `server/ai-service/` regardless of CWD. `.env` now uses `../../models/runs/vehicle/vehicle_yolo26n_20260203_032528/weights/best.pt` — confirmed resolves to the real weight file. Startup now logs which model actually loaded and warns if incident detection is running its heuristic fallback.

5. **LED matrix — 🟡 code-level fixes applied, hardware-unverified (no Tailscale-to-Pi access yet).**
   - Pi 4: `--led-slowdown-gpio` default raised 4→6 in `display_manager.py`. New `raspi_scripts/lcd_pi4/fix_gpio_timing.sh` (report-only by default, `--fix` applies) checks/fixes `snd_bcm2835` blacklist, 1-Wire overlay conflict, `isolcpus`; documents the 74HCT245-vs-74HC245 logic-chip check (hardware-only, can't be automated).
   - Pi 5: audited every render/update path — confirmed no code draws directly onto a live canvas; `RGBMatrixBackend` (disabled) already implements correct offscreen-canvas + `SwapOnVSync`. Its actual disabling reason is a *different*, unresolved bug (SetImage mirrors chained panels). The backend that's actually active, `LedImageViewerBackend`, restarts a subprocess per update instead of double-buffering — added a settle delay before each restart to reduce the transition window, and exposed `--pi5-backend {viewer,rgbmatrix}` so `rgbmatrix` can be tested for real once reachable. Did **not** flip the default — that's an unverified change to a currently-working safety display.

6. **`train.py` `DATASETS_DIR` bug — ✅ fixed, verified.** No longer assumes a sibling folder literally named `Road_Sentinel`; resolves from the script's own location. Confirmed against real `datasets/processed/*/data.yaml` and `models/runs/` on this checkout.

7. **Schema drift — ✅ resolved.** `server/database/mysql_schema.sql` regenerated to match `migrate.ts` exactly (`hourly_analytics` naming, aligned types/defaults). No `recordings` table added to either — that's Phase 2.

8. **Camera B IP — ✅ stopgap + real fix wired in**, per your correction to lean on auto-discovery rather than hardcoding an IP.
   - `seed.ts`'s fallback default aligned to `.108` (was `.102`, inconsistent with `setup_pi5.sh`/the legacy autostart script).
   - `camera_sender.py`'s existing auto-discovery now **persists** a newly-found IP back to Node via `PUT /api/cameras/:id`, so the DB — not a static default — becomes the source of truth. 🟡 Not hardware-tested (needs a real camera).
   - Recommended to you (outside what I can configure): a DHCP reservation for Camera B's MAC on your router eliminates this class of problem permanently.

9. **Confidence threshold — ✅ aligned to 0.5 everywhere** (`ai-service/.env.example`, all 6 fallback defaults in `main.py`, `seed.ts`, `mysql_schema.sql`; `.env`/`seed.ts` were already 0.5).

---

## Phase 0.5 — Hosting migration (Render → self-hosted `irm-pc`, Cloudflare Tunnel, Tailscale) — ⏳ IN PROGRESS, largely blocked on access

**What I found when I actually checked (not assumed):**

| Item | Status | Detail |
|---|---|---|
| Tailscale on laptop (`minniedumpor`, this machine) | ✅ already working | `BackendState: Running`, key valid until 2027 — contrary to the master doc's note that it needed re-auth, it didn't. |
| Tailscale on `irm-pc` | ✅ already working | Online, pingable via DERP relay (73ms). |
| OpenSSH Server on `irm-pc` | ✅ already running | Port 22 responds with `SSH-2.0-OpenSSH_for_Windows_9.5`. |
| SSH **login** to `irm-pc` from this machine | ⛔ **blocked — needs you** | My local key isn't authorized there and I don't know the right Windows username. This machine's public key (add to `irm-pc`'s `authorized_keys` for whichever account should own this): <br>`ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIAT2XcxaYhbAB17iHfpUEUIY5Nsim7W4Rgk7kjulm8XS shaloh@MinnieDumpor` <br>Once that's added, I can reach `irm-pc` directly and do the rest of this phase for real. |
| Tailscale on Pi 4 / Pi 5 | ⛔ **blocked — needs you** | Neither Pi is in the tailnet at all yet (only 2 devices total: this laptop + `irm-pc`). I have no network path to either Pi from here — you'd need to run `curl -fsSL https://tailscale.com/install.sh \| sh && sudo tailscale up --ssh` on each, physically or over your LAN. |
| Aiven MySQL reachability | ⛔ **needs your attention — possible data-loss risk, not just a blocker** | `roadsentinel-1e7c7c14-vandrepaul01-030a.l.aivencloud.com` fails to resolve via **two independent DNS resolvers** (local + 8.8.8.8): "non-existent domain." This isn't a network hiccup on my end — that hostname appears to no longer exist, which usually means the instance was deleted/expired. **Please check the Aiven dashboard directly** to confirm whether the instance (and its data) still exists under a different address or is genuinely gone, before this migration plan can proceed as written. I have not attempted anything further here — didn't want to guess at a production data question. |
| MySQL locally | Found, but likely irrelevant | MySQL 8.0 Server is already installed and *running* on **this laptop** (`minniedumpor`, not `irm-pc`), listening on `0.0.0.0:3306` (all interfaces — not local-only). This doesn't match the plan (self-hosted DB was meant to live on `irm-pc`) and I don't know if it's related to this project at all, so I have not touched it or assumed it's "the" new database. Flagging its existence in case it's relevant to you. |
| `cloudflared` | Not installed anywhere I can reach | Installing it on this laptop wouldn't accomplish the goal — the tunnel needs to front whatever machine actually runs the live Node/Next.js services (`irm-pc`), so this is blocked on SSH access too. |

**Net effect:** almost everything in this phase requires either credentials I don't have (`irm-pc` SSH) or physical/LAN access I don't have (both Pis), plus a real answer about Aiven before the DB migration can be planned at all. I'm not fabricating progress on any of this. Once SSH access to `irm-pc` exists, I can pick this phase back up and actually execute the MySQL install, Cloudflare Tunnel, and the rest — flagging it here rather than stalling the whole project on it.

**Admin-terminal keep/retire decision** (you asked me to ask separately, not decide silently): given Tailscale-to-Pi doesn't exist yet, I'm defaulting to **keep both in-app terminals** (now properly authenticated per Phase 0) since they're currently the *only* path to the Pis at all. Revisit this once Tailscale reaches both Pis for real.

**Moving on to Phase 1** (pure code, no infrastructure dependency) rather than stalling here.

---

### Phase 0.5 — REVISITED (2026-08-04): mostly unblocked

Re-checked everything above rather than trusting the stale table. What actually changed:

| Item | Was | Now |
|---|---|---|
| Tailscale on both Pis | ⛔ not in tailnet | ✅ **both online and reachable** — `road-sentinel-pi4` at `100.98.53.95` (ping 38-154ms) and `road-sentinel-pi5` at `100.94.18.9` (ping 6-113ms), both verified live, not assumed from the status listing |
| Server PC | `irm-pc`, SSH-blocked | ✅ **`irm-pc` at `100.120.27.110`**, pingable (109-551ms). This supersedes `irm-pc` as the deployment target throughout |
| Aiven MySQL | ⛔ NXDOMAIN, unresolved question | ✅ **dropped entirely** per your decision — no recovery attempted, no `mysqldump`, credentials stripped from the repo. Replaced with a fresh local MySQL (below) |
| MySQL | ⛔ nothing usable | ✅ **local Docker MySQL 8.0 working**, schema initialized from `migrate.ts` against a genuinely empty database — see the migration-bug find below |
| `cloudflared` | not installed | ✅ installed (`winget install Cloudflare.cloudflared`, v2026.8.2); `tunnel.bat` added to launch quick tunnels for all three services |
| Tailscale **SSH** to the Pis | n/a | ⛔ **still blocked** — `ssh pi@100.98.53.95` returns `tailnet policy does not permit you to SSH as user "pi"` (also tried `dumporshemjoshua`, `shem`, `admin`, `ubuntu`, `root`). This is a **tailnet ACL setting**, not a Pi-side problem — needs an `ssh` rule in the Tailscale admin console's Access Controls. Network reachability itself is fine, so services on the Pis are reachable by IP:port regardless |

**Real bug found and fixed while standing up the fresh database** — this is the kind of thing that only surfaces against a genuinely empty DB: `migrate.ts` used `ALTER TABLE cameras ADD COLUMN IF NOT EXISTS homography_points ...`, but **MySQL has no `ADD COLUMN IF NOT EXISTS` clause** (that's MariaDB/Postgres syntax). Verified directly against real MySQL 8.0.46 — it's a hard parse error (`ER_PARSE_ERROR 1064`), not a silently-ignored no-op. Every migration run against a fresh database was failing at that statement and taking the whole server down with it. Fixed by dropping the unsupported clause and catching `ER_DUP_FIELDNAME`/`ER_DUP_KEYNAME` in `runMigrations()` instead, which preserves idempotency the way MySQL actually supports it. Confirmed working: fresh DB → `MySQL database connected successfully` → `Database migrations applied successfully` → `Camera seed complete (CAM-A-001, CAM-B-002)`.

**Storage location (your instruction):** MySQL data is a bind mount at `./.data/mysql` rather than a Docker named volume, so the database files live on **D:** alongside the repo instead of inside Docker Desktop's VHDX on C:. Checked actual free space first: C: has ~27 GB free (95% used), D: ~30 GB free (87% used) — and detections/recordings only grow, so this matters. Both `.data/` and `.tunnels/` are gitignored. MySQL's port binding was also tightened to `127.0.0.1:3307` (was `0.0.0.0`), so it's genuinely localhost-only — never exposed through Tailscale or the tunnel, per the plan's security requirement.

**Cloudflare quick tunnels** (`tunnel.bat`): opens one quick tunnel each for the Node API, AI service, and client. Deliberately does **not** tunnel MySQL. Note the real tradeoff, since it affects how you use them: quick-tunnel URLs are randomly regenerated on every restart, so they're fine for public/browser access from outside the tailnet but a poor fit for Pi→server traffic. The Pi setup scripts therefore default to the **Tailscale** address (`100.120.27.110`), which is stable — you can still override with a tunnel URL as the first argument if you want.

**Admin-terminal decision, revisited:** now that Tailscale actually reaches both Pis, the in-app terminals are no longer the *only* path. Still keeping them — they're already built, authenticated, and don't require the tailnet ACL fix that plain SSH does. Reasonable to revisit once Tailscale SSH is permitted.

---

### Phase 0.5 — SSH UNBLOCKED, deployment to `irm-pc` under way

**SSH now works to all three machines.** The earlier failures were my own wrong guesses at usernames, not a policy problem on the Pis:

| Machine | Address | SSH user | Status |
|---|---|---|---|
| Pi 4 | `100.98.53.95` | `roadsentinel` | ✅ in — Raspberry Pi 4 Model B Rev 1.5, kernel 6.18.22-v8+ |
| Pi 5 | `100.94.18.9` | `raspi5` | ✅ in — Raspberry Pi 5 Model B Rev 1.1, kernel 6.12.75+rpt-rpi-2712 |
| `irm-pc` | `100.120.27.110` | `malubay ivan ray` (spaces in the name) | ✅ in, after the public key was added |

**First real look at the deployed Pi state** — both are running, and the picture differs per Pi:

| | Pi 4 | Pi 5 |
|---|---|---|
| `roadsentinel-camera` | ✅ active (running) | ✅ active (running) |
| `roadsentinel-agent` | ✅ active (running) | ✅ active (running) |
| `roadsentinel-display` | ⛔ **not installed at all** | ⛔ **failed** (`status=2/INVALIDARGUMENT`, crash-looped 5× then gave up on 2026-08-12) |
| LED binary | `ledcat` ✅ built | `led-image-viewer` ✅ built |
| Pi model detection | `/dev/pio0` absent → Pi 4 ✅ correct | `/dev/pio0` present → Pi 5 ✅ correct |
| `display_manager.py` deployed | ❌ absent from `~/roadsentinel` | ✅ present |

So the Phase 2 "Pi 4 LED parity" work is real but **has never been deployed** — Pi 4 has no display service and no `display_manager.py` on disk. The rewritten `setup_pi4.sh` installs both; it just hasn't been re-run on the hardware.

Pi 5's display failure is separate and more interesting. Two findings:
1. Its service unit has a **dead Cloudflare quick-tunnel URL baked in** (`--api https://fruit-budapest-stocks-consecutive.trycloudflare.com`) from a tunnel that no longer exists — a concrete instance of exactly the "quick-tunnel URLs don't survive restarts" tradeoff, which is why the Pi scripts now default to the stable Tailscale address instead.
2. Running the same script manually **starts fine** and reaches `Display loop started` → `Clearing panel...`, so the script itself is not broken. `led-image-viewer` does warn `Can't set realtime thread priority=99: Operation not permitted` when not running as root, which affects color stability/flicker. Root-cause of the `exit 2` is still open — investigating.

**Deployment to `irm-pc` in progress** (see `docs/DEPLOYMENT.md` for the full guide and machine facts):
- ✅ Repo cloned to `D:\RoadSentinel` — on D: deliberately; C: has only ~11 GB free, D: ~292 GB
- ✅ Trained vehicle model copied over and byte-verified (5,401,861 bytes, exact match)
- ✅ All three `.env` files written, with **freshly generated** production secrets (`JWT_SECRET`, `PI_AGENT_TOKEN`, `ADMIN_PASSWORD`) — deliberately not reusing the dev laptop's values
- ✅ Python 3.12.10 venv created (the machine's default `python` is 3.14, which PyTorch/ultralytics don't support — `py -3.12` is the working interpreter)
- ⏳ PyTorch CUDA + AI dependencies installing
- ⏳ Node dependencies installing for both services
- ✅ MySQL 8.0.40 running natively (Docker isn't installed and C: can't host Docker Desktop + WSL2), service `RoadSentinelMySQL`, root and data on D:, port 3307, bound to `127.0.0.1` only. Complication found and worked around: `dev.mysql.com` returns **403 Forbidden** from `irm-pc`'s IP specifically (fine from the dev laptop), so the ZIP was downloaded here and copied across.
- ✅ `cloudflared` already present on `irm-pc`

---

### 🎉 Phase 0.5 / Phase 4 — DEPLOYED AND VERIFIED END TO END

**The full pipeline is live on `irm-pc` and serving publicly.**

| Component | Status | Evidence |
|---|---|---|
| MySQL 8.0.40 | ✅ | Service `RoadSentinelMySQL`, all 5 tables created, both cameras seeded |
| **`migrate.ts` fix proven** | ✅ | Migrations applied cleanly against a genuinely fresh production database — the exact scenario that failed before the `ADD COLUMN IF NOT EXISTS` fix |
| AI service | ✅ | `custom_model=True`, classes `{car, motorcycle, bicycle, bus, truck}`, **on CUDA** (`torch 2.5.1+cu121`, `cuda True`, RTX 3060 Ti) — the real trained weight, not the stock fallback |
| Node service | ✅ | `/health` and `/api/public/status` both return correct JSON against the live DB |
| Client | ✅ | Dashboard, `/status`, `/admin` all HTTP 200 publicly |
| Cloudflare tunnels | ✅ | All three assigned and reachable from the public internet |
| Persistence | ✅ | All six processes (3 services + 3 tunnels) registered as Scheduled Tasks — they survive SSH disconnect *and* reboot |
| **Camera B (Pi 5)** | ✅ **LIVE** | RTSP opened, Socket.IO connected to `irm-pc`, `sent=258 errors=0` |
| **Real detections** | ✅ | **7 vehicle detections persisted to MySQL** from the actual Busay camera; `CAM-B-002` shows `online` |
| Pi 5 LED display | ✅ | Running and driving the panel after the three bug fixes |

**Two deployment-mechanics bugs found and fixed along the way:**
1. *Processes started over SSH die with the session.* `Start-Process` from an SSH session puts the child in the session's job object, so every service died the moment the SSH connection closed — silently, several times, before I spotted it. Fixed by registering everything as **Scheduled Tasks**, which also gets boot-start for free (`install_services.ps1`, `install_tunnels.ps1`).
2. *`USERDOMAIN` is wrong on non-domain machines.* `Register-ScheduledTask` failed with "No mapping between account names and security IDs was done" because `$env:USERDOMAIN` reports `WORKGROUP`, which isn't a resolvable principal. The local-account form is `$env:COMPUTERNAME\$env:USERNAME`.

Also hit the documented PowerShell 5.1 native-stderr trap: `mysqld --initialize-insecure` writes ordinary progress to stderr, which under `$ErrorActionPreference='Stop'` became a fatal `NativeCommandError` and left a half-initialized data directory. Fixed by checking `$LASTEXITCODE` explicitly instead.

**The Pis were running pre-revamp code.** A significant discovery: everything from Phases 0–3 existed in git but had **never been deployed to the hardware**. The deployed `pi_agent.py` had no `PI_AGENT_TOKEN` support at all (so the authenticated `/admin` relay from Phase 0 could never have worked), and both Pi services pointed at **long-dead Cloudflare quick-tunnel URLs** — concrete proof of why the Pi scripts now default to stable Tailscale addresses instead. Current scripts are now deployed to the Pi 5.

**Measured, not assumed — the 30 FPS question:** Camera B sustains **~9-10 FPS**, not 30 (`fps=10.3`, `dropped=581`). The Phase 2 audit concluded there was no *code-level* bottleneck, and that still looks right — the sender is dropping stale frames by design under backpressure. But the delivered rate is well short of the 30 FPS requirement, so the constraint is upstream (camera configuration, RTSP substream, or network). This is the first real measurement of it; needs investigation before the 30 FPS claim can be made.

**Still open:**
- **Pi 4** — camera and agent run, but it has **no display service at all** and is still on pre-revamp code. Installing it needs `sudo`, and the `roadsentinel` account requires a password that passwordless SSH can't supply (the Pi 5's `raspi5` has NOPASSWD). Needs either the password or a manual `setup_pi4.sh` run.
- **Pi 4 multiplexing fix unverified** — the `multiplexing=0 → 1` change is committed and matches the project's own hardware notes, but can't be confirmed on the panel until the display service is installed.
- **Camera A offline** — `CAM-A-001` still shows `offline`; the Pi 4 camera service points at the same dead tunnel URLs and hasn't been repointed yet (same sudo blocker).
- **Quick-tunnel URLs rotate on restart.** Fine for browser access; that's why Pi→server traffic uses Tailscale.

---

## Phase 1 — Functionality correctness pass — ✅ COMPLETE

Full lint/typecheck sweep clean (`tsc --noEmit` on both `client/web` and `server/node-service`, `prettier --write`, `eslint --fix`, `python -m py_compile` on every changed `.py` file) before closing this phase.

1. **Every decorative no-op button wired — ✅, all real, none faked:**
   - **Analytics** "Export PDF"/"Export CSV" — real: CSV via `lib/export.ts`'s `downloadCsv()` (Blob download), PDF via a print-formatted window + `window.print()` (native "Save as PDF", no new dependency).
   - **Cameras** "Open Calibration Tool"/"View Calibration Guide" — real: see item 2 below.
   - **Reports** "Download" — the old version was hardcoded fake report names with a dead button. Replaced with 3 real, on-demand CSV reports (Detections/Incidents/Hourly Analytics) generated from what's actually in the database — no fake "saved reports" backend exists yet (that's genuinely Phase 2 scope), so I didn't pretend one does.
   - **History** "Play" — same issue: fake recordings, dead button, and there is **no video recording pipeline anywhere in this system** (confirmed — no camera-side recorder, no populated `recordings` table). Rather than leave a misleading fake video player, the page now shows a real, filterable **detection log** (queries `/api/detections`) and honestly states recording isn't implemented yet.
   - **Settings** "Save All Settings"/"Reset to Defaults" — real: `lib/settings.ts` persists to `localStorage` (no settings backend exists yet either). "Sound Alerts" actually plays a sound (Web Audio API beep, no asset) when a critical incident arrives on the Incidents page; "Email Notifications" is honestly captioned as saved-for-later since no email-sending capability exists (that's the Phase 2 "critical-incident alert hook" feature, not yet built).

2. **Speed-estimation split-brain — ✅ resolved by wiring homography into production** (the master doc's research recommended this over deleting it). This was the single biggest Phase 1 item:
   - `cameras` table gained a `homography_points JSON` column (`migrate.ts` + regenerated `mysql_schema.sql`), storing `{image_points: [[x,y]×4], real_points: [[x,y]×4] in meters}`.
   - `PUT /api/cameras/:id` accepts it (and can explicitly clear it back to `null`, distinct from "not provided").
   - `camera_sender.py` fetches it alongside `pixels_per_meter`/`speed_limit` and forwards it to `/api/detect` as a JSON form field.
   - `server/ai-service/app/models/traffic_detector.py`: new `_get_homography_matrix()` (cached per camera, `cv2.getPerspectiveTransform`), `_homography_speed()`, `_update_tracks_homography()`. `detect()` now prefers homography-corrected speed when a camera is calibrated, falling back to the old flat pixels-per-meter estimate otherwise.
   - **Verified with a real numeric test** (not just typecheck) run against the actual `cv2`/`numpy` in the ai-service venv: calibration points map back to themselves (error < 1cm), and homography speed measurably diverges from the naive flat-ppm estimate for identical pixel movement (11.5 km/h vs. 7.3 km/h in the test case) — demonstrating the perspective-correction is doing real work, not a no-op.
   - New client component `components/calibration-tool.tsx`: a real Calibration Tool (click 4 points on the live MJPEG stream, enter rectangle width/length, save) and Calibration Guide, wired into the Cameras page with a live calibrated/uncalibrated status indicator.
   - The old standalone, disconnected homography prototype (`inference/camera_calibration.py`) is **deleted** — its algorithm is now the real production implementation; keeping a redundant unconnected copy around was exactly the "duplicate logic" Phase 1 flags. `inference/speed_detection.py` (the *simpler*, non-homography standalone prototype) is kept — genuinely different purpose (offline testing without any server running) — with a header comment clarifying scope.
   - 🟡 Not hardware-verified (needs a real camera + Phase 4).

3. **Incident heuristic labeling — ✅ done end to end.** `IncidentDetector` now tags every result `is_heuristic: True/False`; auto-generated speeding incidents (real tracked speed) are tagged `False`. `camera_sender.py` forwards it as `metadata.is_heuristic`. Client: `AlertCard` shows an "ESTIMATED — no model trained" badge (with a tooltip explaining why) wherever `is_heuristic` is true, on both the dashboard and Incidents page.

4. **Duplicate detector-runner logic — ✅ resolved by clarifying scope, not merging** (merging would have reduced usefulness — they load different things: a local `.pt` file vs. the deployed HTTP service). Added header comments to `training/validate.py`, `testing/test_video.py`, `testing/test_images.py`, and `inference/speed_detection.py` cross-referencing each other's distinct authoritative purpose.

5. **Config surface cleanup — ✅.** `LOG_FILE` now actually controls the log path (`logger.ts`, error log derived alongside it) instead of being ignored. `FRAME_PROCESSING_RATE`, `VIDEO_RECORDING_ENABLED`, `MAX_RECONNECT_ATTEMPTS` **removed** from `.env.example` — none map to anything Node actually does (frame/reconnect handling lives entirely in `camera_sender.py` on the Pi, not in Node), so wiring them up would have meant inventing fake behavior just to make the flag "do something." Also removed the now-fully-dead `SUPABASE_*` block from `.env.example` (storage genuinely runs through the AI service's local disk, not Supabase) while the comment now says so explicitly.

6. **Unused dependencies — ✅ removed.** `@supabase/supabase-js`, `node-rtsp-stream`, `fluent-ffmpeg` (+ its `@types`) uninstalled from `server/node-service` — confirmed zero imports before removing, `tsc` still clean after.

7. **All 8 docs named in the plan fixed** against `documentation.md §15`'s exact drift list — ✅: `README.md`, `PROJECT_STRUCTURE.md`, `START_HERE.md`, `TRAINING_GUIDE.md`, `CAMERA_TEST_GUIDE.md`, `server/ai-service/README.md`, `training/README.md`, `raspi_scripts/README.md`. All now describe the real `training/train.py` CLI (not the nonexistent `train_vehicle_detector.py`/`quick_train.py`), the real `testing/` location for test scripts (not `server/ai-service/`, and the nonexistent `test_visual_pro.py`/`test_visual_optimal.py` references are gone), and the real `models/runs/` output layout (not `models/v1/v2/production`). `README.md`/`PROJECT_STRUCTURE.md`/`START_HERE.md`/`TRAINING_GUIDE.md` also now mention this Phase 1's real features (homography calibration, admin login) — full prose polish is still Phase 5's job (explicitly a fresh top-to-bottom pass), this was a factual-accuracy pass.

8. **LED backend choice — ✅ documented in `raspi_scripts/README.md`**, per the explicit Phase 1 requirement: unified `display_manager.py` (not `lcd/`/`lcd_pi4/`, which are superseded/historical) is the current driver for both Pis; current defaults are `LedcatBackend` (Pi 4) and `LedImageViewerBackend` (Pi 5); `RGBMatrixBackend` is named as the architecturally-better long-term candidate for Pi 5 once its chained-panel mirroring bug is hardware-verified-fixed, opt-in now via `--pi5-backend rgbmatrix` for testing.

**Not done / carried forward:** everything here is 🟡 code-complete-but-hardware-unverified where it touches physical devices (LED fixes, homography on a real camera, Pi-side auto-discovery persistence) — same access gap as Phase 0.5. No user-facing claim in the docs above says otherwise.

---

## Phase 2 — Feature completion & new features — ✅ COMPLETE

Full lint/typecheck sweep clean again before closing. Several pieces **live-verified with real tests**, not just typechecked (details below) — same standard as Phase 0/1.

**Stubbed features completed:**

1. **Recordings — ✅ real pipeline built, end to end, not just a table.**
   - `recordings` table added to `migrate.ts` + `mysql_schema.sql` (matches the `Recording` type that already existed in `types/index.ts` but had nothing behind it).
   - New `server/node-service/src/routes/recordings.ts`: `GET /api/recordings`, `GET /:id`, `POST /` — mounted at `/api/recordings`.
   - `raspi_scripts/camera/camera_sender.py`: new `Recorder` class, opt-in via `--record` (off by default — untested against real camera hardware). Segments the stream into fixed-length local `.mp4` files (`cv2.VideoWriter`), uploads each finished segment to the AI service's storage, registers it with Node, then deletes the local copy (SD cards are small). Counts vehicle-frames/incidents per segment from the same AI results already flowing through `ai_task`. Rotation/upload/registration all happen as background tasks — never blocks capture, same fire-and-forget pattern as the existing AI dispatch.
   - **Verified for real**: `cv2.VideoWriter` with the `mp4v` fourcc actually writes a valid, re-readable MP4 on this system (round-trip write→reopen→read test passed) — the core primitive isn't just assumed to work.
   - Client: `History` page now fetches and lists real recordings for the selected date, with an actual `<video controls>` player when one exists; falls back to an honest "no recordings for this date" message (expected — recording is opt-in and hasn't run on hardware yet) rather than a fake player.

2. **Reports backend** — already satisfied by Phase 1's work (real CSV generation from live data); no further action needed here beyond the new thesis-export report type (see New Features below).

3. **Settings persistence** — already satisfied by Phase 1 (`localStorage`); reconsidered whether this should be server-side instead, but the two settings that exist (email notif./sound alerts) are genuinely per-browser/per-operator preferences, so client-side storage is the semantically correct choice, not a shortcut.

4. **Night-vision/IR auto-switching — ✅ brought into the production path**, with an important correction: the "legacy script" (`set_ir_auto_all.py`) the master doc pointed at **doesn't actually exist anywhere in this repo** — it's referenced by `camera_reboot_autostart_setup.sh` but was never committed, so there was nothing to port. Wrote new ONVIF IR-auto logic from scratch instead: `camera_sender.py`'s new `set_ir_auto()` (ONVIF Imaging service, `IrCutFilter=AUTO`), opt-in via `--ir-auto` (+ `--onvif-port`/`--onvif-user`/`--onvif-pass`), best-effort — any failure logs a warning and the core camera pipeline continues unaffected. 🟡 Untested against real camera ONVIF hardware (no camera access).

5. **Pi 4 LED matrix build-out — ✅ done, symmetric with Pi 5 now.** `setup_pi4.sh` rewritten to match `setup_pi5.sh`'s structure: builds `ledcat`, installs `display_manager.py`, adds the `roadsentinel-display` systemd service (`--pi 4`), updated helper scripts and echoed instructions. `raspi_scripts/README.md` and the top-level `README.md` architecture diagram updated to describe both Pis symmetrically instead of "Pi 4 has none." 🟡 Hardware-unverified (same access gap).

6. **Always-on 30 FPS live feed — audited, not modified.** Read `camera_sender.py`'s capture loop line by line against the master doc's "AI throttling the live view" concern and **found the concern doesn't match the code**: AI dispatch is already fire-and-forget (`asyncio.create_task`, never awaited inline), frame push already runs on an independent queue-consumer task that drops stale frames under backpressure rather than blocking, and neither can stall the capture loop's own pacing. I made no changes here because there was no real bottleneck to fix — the architecture the master doc's research recommended is already what this code does. What I couldn't do: confirm the physical RTSP cameras can actually sustain 30 FPS (the DB's seeded `fps: 15` per camera may reflect an actual hardware ceiling, unrelated to software) — that's a Phase 4 hardware-measurement question; the client already has the FPS instrumentation needed to answer it (`components/video-feed.tsx`).

**New features** — per your standing instruction not to wait for a go-ahead, I made a default call on each (documented here for override) rather than blocking on a question: built the 4 that are self-contained and don't require external paid services or a not-yet-existing crash model; skipped the one that's explicitly conditional on something out of scope.

7. **Adaptive detection sampling — ✅ built and live-verified.** New `AdaptiveSampler` class in `camera_sender.py`: samples every eligible frame while a vehicle's been seen in the last 5s (keeps IoU-tracking/speed-calc continuity intact), backs off to every 3rd/6th/10th frame during longer idle stretches (tiered by how long the road's been empty) to cut AI-service load — never touches the live-view frame rate, only the AI-sampling rate. On by default, `--no-adaptive-sampling` to disable. **Verified with a standalone logic test** (stubbed the heavy cv2/aiohttp/socketio imports, exercised the real class): correct full-rate sampling near a detection, correct tiered throttling during a simulated 60s-idle period, correct reset back to full-rate immediately after a new detection, correct passthrough when disabled.

8. **Critical-incident webhook alert hook — ✅ built and live-verified end to end.** New `server/node-service/src/services/alert.service.ts`: generic JSON POST to `ALERT_WEBHOOK_URL` (works as-is with Slack/Discord incoming webhooks, Zapier, n8n, or a custom endpoint) when an incident at or above `ALERT_WEBHOOK_MIN_SEVERITY` (default `critical`) is created — wired into `POST /api/incidents`, fire-and-forget so a broken webhook can never delay the API response. Deliberately provider-agnostic rather than assuming SMTP/SMS credentials I don't have. **Verified with a real local HTTP listener**: critical incident → webhook fires with the right payload; medium incident (below threshold) → correctly does not fire; no `ALERT_WEBHOOK_URL` configured → correctly silent, no error. Not configured in this repo's own `.env` — it's an opt-in capability, inactive until you set a URL.

9. **Public no-admin-access live status page — ✅ built.** New `GET /api/public/status` (Node, unauthenticated, exposes only: clear/vehicle-incoming/incident state, camera online count, today's vehicle/incident tallies — no RTSP URLs, no camera feed, no config) and `client/web/app/status/page.tsx` — large, glanceable, auto-refreshing (3s) status display suitable for a phone. Mirrors the same `VEHICLE_ALERT_SECS` convention `display_manager.py`'s `SystemState` already uses, so the physical LED sign and this web page agree on "current state" rather than computing it two different ways. Linked from the sidebar for discoverability. Note: it still renders inside the shared sidebar layout (a fully chrome-free variant would need a Next.js route-group restructure) — reasonable to fold into Phase 3's design-system pass rather than doing layout surgery here.

10. **Thesis speed-violation export — ✅ built.** New `GET /api/analytics/violations?date=` — speed-limit violations bucketed by hour-of-day (all 24 hours always present, 0-filled), with avg/max speed per bucket — more useful for a blind-curve safety thesis than a flat speed histogram. Added as a 4th report type on the Reports page (CSV, reusing the existing on-demand-generation pattern from Phase 1).

11. **Skipped, as designed: model-confidence/drift indicator.** Explicitly conditional in the master doc on "once a real crash model exists" — none does, and training one is explicitly out of scope for me to do unattended. Building UI for a model that doesn't exist yet would be speculative work against a hypothetical future state, not a real feature — will revisit once `training/train.py --dataset accident` has actually been run.

**Explicitly out of scope, unchanged:** training the crash/incident model itself. `datasets/processed/busay_accident_detection/` is ready; `python training/train.py --dataset accident --model-size n --epochs 100` is the ready-to-run command whenever you want to spend the GPU hours.

---

## Phase 3 — Design overhaul — ✅ COMPLETE

**Palette decision (made without waiting for a go-signal, per your standing instruction):** went with the master doc's own explicit recommendation, **"Night Watch"** — near-black operations base (`#0B0E14`) with amber (`#F2B33D`) as the sole brand accent, red (`#E5484D`) reserved exclusively for critical-incident severity. Rationale documented in full in the new `client/web/DESIGN.md` (also required by the master doc's deliverable #6) — short version: a red-primary UI is wrong for an incident-safety dashboard because it stops meaning "critical" the moment the app loads. Didn't take the cyan/teal alternative — amber keeps the road-safety identity (traffic-signal caution) that a cooler "ops tool" color wouldn't carry.

1. **Design tokens — ✅.** `client/web/hero.ts` fully rewritten: light + dark HeroUI themes, all five semantic scales (primary=amber, secondary=info-blue, success=teal, warning=orange, danger=red) with correct light/dark-appropriate foreground text per color (danger uses white text; every other color uses dark text since Night Watch's non-red accents are all light/mid-toned — verified by contrast reasoning, not just guessed). `client/web/styles/globals.css`'s Tailwind v4 `@theme` block defines the same palette as CSS custom properties (`--color-bg/surface/surface-2/border/brand/success/warning/danger/info/fg/fg-muted`) plus `--ease-standard`/`--ease-emphasized` motion tokens — single source of truth, no more per-file hex values.
2. **Font pairing — ✅.** Space Grotesk (headings) + IBM Plex Sans (body) + IBM Plex Mono (stats/timestamps/IDs/terminal), via `next/font/google` in `client/web/config/fonts.ts` — replaces the previous plain-system-font stack (which itself never actually wired up correctly: the old `fontSans.variable` was a raw string like `"--font-sans"` applied as a literal className, not a real CSS-variable injection — silently a no-op). Plex Sans/Mono are a matched superfamily built for data-dense tooling; Space Grotesk avoids the Inter-only anti-pattern the master doc's "AI slop" checklist explicitly flags. **Verified for real**: ran `next build` — fonts fetched and self-hosted successfully (network access confirmed first), zero runtime font errors.
3. **Shared layout shell — ✅.** `animated-background.tsx` rebuilt as a slow, low-amplitude amber/info "aurora" (down from 4 saturated purple/burgundy blob orbs), wrapped in `prefers-reduced-motion` handling in `globals.css`. `sidebar.tsx` fully retoken'd; also removed the dead "Documentation" nav item (see #4). New `components/page-transition.tsx` — Framer Motion `AnimatePresence` wrapper mounted in `app/layout.tsx`, covers every route, itself reduced-motion aware via `useReducedMotion()`.
4. **Deleted unused starter leftovers — ✅.** Removed `/about`, `/blog`, `/pricing`, `/docs` (all confirmed to be genuinely unstyled/unwired stub content — read each before deleting) and the dead `components/navbar.tsx` (confirmed zero imports anywhere in the codebase before removing — the sidebar has always been the only real nav). Cleaned the now-dangling `/docs` reference out of `sidebar.tsx` and `config/site.ts`. Renamed `package.json`'s `"name"` from the leftover `"next-app-template"` to `"road-sentinel-web"`.
5. **Toasts wired to real events — ✅, using HeroUI's own native `Toast`/`ToastProvider`/`addToast` (already a project dependency, `@heroui/toast` — no new library pulled in, matching the master doc's "stay on HeroUI" guidance even though this project is pinned to HeroUI v2, not v3).** Mounted once in `app/providers.tsx`. Wired to: a new incident arriving over the dashboard's live socket (`app/page.tsx`) — critical incidents persist until dismissed (`timeout: 0`) instead of auto-clearing, and also trigger the existing `playAlertSound()` from Phase 1's settings; WebSocket connect/reconnect/disconnect (reconnect toast only fires after a genuine prior disconnect, not on first mount); Settings page save/reset (`app/settings/page.tsx`); Reports page CSV download success/failure (`app/reports/page.tsx`). Not decorative — every one of these traces to a real state change, not a button click for its own sake.
6. **Design rationale note — ✅.** `client/web/DESIGN.md` — before/after, the color reasoning, and the font reasoning, written to be directly citable in the thesis write-up.
7. **All shared components retoken'd** (stat-card, alert-card, camera-status, video-feed, calibration-tool) — alert-card's severity mapping specifically redone as a 4-step RAG-safe ramp (info→low, amber→medium, warning-orange→high, danger-red→critical **only**) rather than reusing raw Tailwind red/yellow/orange/blue, so severity chips can never accidentally imply "critical" at a lower tier.
8. **All 11 real app pages retoken'd**: dashboard, monitor, analytics, incidents, cameras, history, reports, settings, admin, status, error boundary. Also swapped the print-to-PDF export template's (`lib/export.ts`) text color and added an amber header rule for brand consistency in exported reports, while deliberately keeping it light-background/dark-text (correct for a printed document regardless of in-app dark theme).

**Process note:** delegated the mechanical retheming of 6 of the page files (monitor, analytics, incidents, cameras, history, admin) to a background subagent with an exact token-mapping table, after establishing the pattern myself on the shared components first. The agent hit a session/API limit partway through and its task reported "failed," but its actual file edits were sound — verified this directly (not assumed): `tsc --noEmit` clean, zero leftover old hex/raw-Tailwind-color patterns via repo-wide grep, and a full `next build` succeeded end-to-end (13 static pages generated, no errors). Finished the one file it hadn't reached (`app/error.tsx`) myself.

**Final verification (all real, all local, no fabrication):**
- `rm -rf .next && npx tsc --noEmit` — clean, zero errors, across the entire client.
- `npx eslint .` — clean, zero errors/warnings, entire client.
- `npx prettier --write` — clean across every touched file.
- `npx next build` — **production build succeeds**: compiled in 16.4s, 13/13 static pages generated, no type errors, no lint errors, Google Fonts fetched and self-hosted correctly. This is the strongest signal available without a browser — confirms routing, providers, HeroUI plugin wiring, and font loading all work together, not just that individual files parse.
- Repo-wide grep for every old theme hex value (`#ED9E59`, `#1B1931`, `#44174E`, `#862249`, `#A34054`, `#E8BCB8`) across `app/`, `components/`, `lib/` — zero remaining hits after the PDF-export fix.

**Not verified (needs a real browser, flagged honestly rather than assumed):** visual appearance, animation smoothness, actual toast timing/stacking behavior, responsive layout at narrow viewports, dark/light theme switch appearance. `npm run dev` wasn't started against a fresh instance to eyeball this — the existing stale dev instance on port 3001 was left untouched throughout (same rule as every prior phase). If you want a visual pass before Phase 4, say so; otherwise proceeding on the strength of the build/lint/typecheck verification above, per the "keep going" instruction.

---

## Phase 4 — Full functional audit & hardware verification — ⚠️ PARTIAL (hardware-blocked, as anticipated)

Ran every check that's possible from this machine against **live, isolated instances** (not code-reading, not simulation) — AI service on port 8000, Node service on port 8001... (actually used 3098 to avoid the stale pre-existing dev instance on 3001, which was never touched, same rule as every prior phase). Both instances killed cleanly afterward (`taskkill` on the exact listener PIDs from `netstat`, verified after). Full evidence below.

### 🔴 New, more serious finding: Aiven MySQL hostname doesn't exist (NXDOMAIN)

Previously flagged as "DNS not resolving" — re-checked now and it's worse than that. `nslookup roadsentinel-1e7c7c14-vandrepaul01-030a.l.aivencloud.com` against both the local resolver and `8.8.8.8` returns **"Non-existent domain"**, not a timeout or a transient failure. That specific hostname does not exist in DNS at all right now. This usually means the Aiven service was deleted, expired, or paused (Aiven typically deletes the DNS record on service termination), not a connectivity issue on this end. **This blocks the live database entirely** — everything DB-dependent (cameras, detections, incidents, analytics, recordings, public status) is running in degraded "warn and continue" mode, which is graceful (verified below, doesn't crash) but is not a working system. Recommend checking the Aiven dashboard directly to confirm whether that service still exists before doing anything else with Phase 0.5.

### ✅ AI service — all 9 documented endpoints, live-called, real responses

Started an isolated instance (`server/ai-service/venv`, port 8000) and called every endpoint for real with a genuine dataset image (`datasets/downloaded/test/images/bandicam-...jpg`):

| Endpoint | Result |
|---|---|
| `GET /` | ✅ `{"service":"Road Sentinel AI Service","status":"running"}` |
| `GET /health` | ✅ `{"status":"healthy",...}` |
| `GET /api/stats` | ✅ correct shape, model load state accurate before/after |
| `POST /api/detect/traffic` | ✅ real detection: 1 motorcycle, 83.6% confidence |
| `POST /api/detect/incidents` | ✅ ran heuristic, correctly returned empty (quiet scene) |
| `POST /api/detect` (combined) | ✅ both detections+incidents in one call |
| `POST /api/storage/upload` | ✅ file written, public URL returned |
| `GET /api/storage/list` | ✅ showed the uploaded file |
| `DELETE /api/storage/delete` | ✅ file removed, confirmed gone from a follow-up list call |

**Trained model verification — ✅ confirmed, not assumed.** `models/runs/vehicle/vehicle_yolo26n_20260203_032528/weights/best.pt` exists on disk (5.4MB, alongside all epoch checkpoints from the training run). The service log on first detection call read: `Custom vehicle model detected with classes: {0: 'car', 1: 'motorcycle', 2: 'bicycle', 3: 'bus', 4: 'truck'}` and `Traffic detector ready — custom_model=True` — this is the actual trained 5-class model running, not a silent fallback to stock `yolov8n.pt` (which would show `custom_model=False` and COCO's 80 classes). `/api/stats` correctly flipped `traffic_model.loaded` from `false` to `true` after the first call (confirms the lazy-load design is intentional, not a bug — the model loads on first use rather than at cold-start, keeping health checks fast).

No incident/crash model exists yet (`INCIDENT_MODEL_PATH=./models/incident.pt` doesn't resolve) — confirmed the service correctly logs the heuristic-fallback warning rather than crashing or silently pretending to have a real model. Unchanged from Phase 1/2 — training that model remains explicitly out of scope for me to run.

### ✅ Node service — CORS, auth, and rate-limiting fixes verified live (not just re-read)

Started an isolated instance (port 3098) — it starts in degraded mode when the DB is unreachable (warns, skips migrations/seeding, keeps serving) rather than crashing, which is itself correct, defensive behavior worth confirming rather than assuming.

- **CORS allowlist**: `curl` with `Origin: http://localhost:3000` (the configured `CORS_ORIGIN`) → response includes `Access-Control-Allow-Origin: http://localhost:3000`. Same request with `Origin: http://evil.example.com` → **no CORS header at all** in the response (a real browser would block the page from reading it). Confirmed the allowlist actually discriminates, not just present-but-permissive.
- **Admin namespace auth** (the Phase 0 fix): wrote a throwaway Socket.IO client test (deleted after) against the live instance —
  - No token → `connect_error: "Authentication required"` ✅ rejected
  - Bogus token → `connect_error: "Invalid or expired token"` ✅ rejected
  - Real token (obtained via an actual `POST /api/auth/login` call with the local `ADMIN_PASSWORD`) → **connected successfully** ✅ — confirms the middleware isn't just rejecting everything indiscriminately, it's genuinely checking token validity.
- **Login rate limiting**: 8 rapid wrong-password attempts → first 5 returned `401`, attempts 6-8 returned `429`. Working as designed.
- **Graceful DB-down degradation**: `/api/public/status`, `/api/analytics/violations`, `/api/recordings`, `/api/cameras` all returned clean `{"success":false,"error":"..."}` JSON (500) with the DB unreachable — no stack traces leaked to the client, no process crash, server stayed listening and responsive through the entire audit.
- **Node ↔ AI service integration**: startup log showed `🤖 AI Service: Connected` — confirmed the two services actually talk to each other over HTTP as configured, not just independently functional.

Did not audit every remaining route individually (`detections.ts`, `incidents.ts` POST paths, etc.) since they're all DB-dependent and the Aiven outage above makes that testing meaningless right now — would just be re-confirming the same "DB unreachable" error repeatedly. Worth a follow-up pass once the Aiven situation is resolved.

### 🟡 Blocked — needs physical hardware access (unchanged from Phase 0.5, not fabricated)

- Raspberry Pi camera checks (Pi 4/Cam A, Pi 5/Cam B): no Tailscale reach to either Pi yet.
- Real-world 30 FPS sustained-delivery measurement for both cameras: architecture was audited and found sound in Phase 2 (no code bottleneck), but actual sustained FPS numbers require live camera hardware.
- Camera B auto-discovery recovery path: needs a live camera to actually trigger a reconnect.
- LED matrix hardware checks (both Pis): needs physical access to see the panels.
- `irm-pc` SSH access: still blocked — my public key was provided earlier in this session for you to add to `authorized_keys`; unknown if that's been done.

None of these are marked done. Given the "keep going" instruction, proceeding to Phase 5 with these explicitly still open rather than blocking further progress on them — Phase 5 will commit and document what's verified vs. not, and **will not push** given the current state (see Phase 5 below for the exact reasoning).

---

## Phase 5 — Commit, push, and final documentation — ✅ COMMITTED, ⛔ NOT PUSHED (by design)

**Committed locally, 5 logically-grouped commits** (`git log --oneline` from oldest to newest):

1. `docs: ground-truth codebase audit + revamp planning/tracking docs` — `documentation.md`, `Summarization.md`, the master plan doc.
2. `feat(server): JWT admin auth, CORS allowlist, homography speed, and correctness fixes (Phases 0-1)` — all server/hardware-side Phase 0 security fixes + Phase 1 correctness fixes.
3. `feat(server): recordings, adaptive sampling, IR auto-switch, webhook alerts, public status, Pi 4 LED parity (Phase 2)` — all Phase 2 feature-completion work.
4. `feat(client): complete design overhaul — "Night Watch" design system (Phase 3)` — the entire `client/web` tree.
5. `docs: fresh top-to-bottom README pass + supplementary doc-drift fixes (Phase 5)` — final README rewrite + remaining doc-drift files.

**Grouping note for honesty**: a handful of files (`server.ts`, `camera_sender.py`, `setup_pi4.sh`/`setup_pi5.sh`, `main.py`, `admin/page.tsx`, `calibration-tool.tsx`, `export.ts`, `settings.ts`) were substantively touched across more than one phase. Splitting a single file's history into separate phase-accurate commits would require risky hunk-level surgery (`git add -p`) reconstructed from a compacted conversation memory — not worth the risk of producing a broken intermediate commit. Each file landed in the commit for its most substantial/most recent change, with commit bodies noting where earlier-phase work rides along. The full accurate phase-by-phase narrative lives in this document, not in git archaeology.

**Verified before every commit**: `tsc --noEmit` clean (both `client/web` and `server/node-service`), `eslint` clean, `prettier` clean, full `next build` succeeds. `.gitignore` re-confirmed to exclude `.env`, all `*.pt` model weights, `datasets/downloaded|processed`, and `node_modules`/`venv` — no secrets in any of these 5 commits (double-checked `git status --short` for stray `.env` files before every commit; none appeared, confirming `.gitignore` is doing its job).

### ⛔ Not pushing to `origin/main` — per your own standing rule

Phase 4's audit did not pass cleanly:
- The Aiven MySQL hostname is **NXDOMAIN** — genuinely gone from DNS, not just unreachable. Everything DB-dependent runs in degraded mode. This needs your attention on the Aiven dashboard before it's meaningful to call the backend "working."
- Raspberry Pi hardware checks (both LED matrices, both camera streams, sustained FPS, Camera B auto-discovery-in-practice) are entirely unverified — no Tailscale reach to either Pi yet.
- `irm-pc` SSH access is still blocked (my public key was provided earlier for you to add to `authorized_keys` — unknown if done).

Per your master plan's own explicit rule ("Never push to origin/main with a failing or unverified Phase 4 audit — fix and re-verify first"), I'm stopping here rather than pushing. Everything is committed locally on `main` and ready to push the moment you say so — either after resolving the blockers above, or if you'd rather push now and treat the remaining hardware verification as a follow-up pass, just say so and I will.

### What's needed from you to unblock the rest

1. **Aiven**: check the dashboard — does that MySQL service still exist? If it was deleted, either restore it or point `DB_HOST`/credentials at a replacement (self-hosted per Phase 0.5, or a new managed instance).
2. **`irm-pc` SSH**: add the public key I provided earlier in this session to `authorized_keys` so Phase 0.5's remaining Tailscale/Cloudflare Tunnel setup can proceed.
3. **Tailscale on both Pis**: needed before any of Phase 4's hardware checks can run for real.
4. Once those are in place, say the word and I'll re-run Phase 4's hardware-dependent checks and push if everything's clean.

---

## Phase 4 — Functional audit & hardware verification — not started
Expect this to be **partially blocked** by the same access gaps as Phase 0.5 (can't verify LED fixes or live camera FPS without reaching the Pis; can't do a full API audit against a production DB without resolving the Aiven question). Will do everything reachable and mark the rest clearly rather than claim a clean pass that isn't real.

## Phase 5 — Commit, push, final README — not started
**Will not push** if Phase 4 has unresolved failures, per your explicit rule — will commit locally and document exactly what's still open instead.
