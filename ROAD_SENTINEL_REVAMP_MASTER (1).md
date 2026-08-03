# Road Sentinel — Full Revamp Master Prompt

Single source of truth for the revamp. Give this file, alongside `documentation.md`, to Claude Code. Use `00_START_HERE_CLAUDE_CODE_PROMPT.md` as the actual chat message to kick things off — it just tells Claude Code to read both files and begin.

---

## 0. Ground truth

`documentation.md` (audited 2026-08-03, branch `main` @ `ed6c0fd`) is the single source of truth for what actually exists in this repo today. The five pre-existing top-level docs (`README.md`, `PROJECT_STRUCTURE.md`, `START_HERE.md`, `TRAINING_GUIDE.md`, `CAMERA_TEST_GUIDE.md`) are stale relative to the real code — don't trust them until Phase 1 fixes them.

## 1. Mission

Take Road Sentinel from "functioning but rough thesis prototype" to a polished, correct, secure, and fully wired system:
1. Fix what's broken or insecure.
2. Move hosting off Render onto the user's own PC, with Cloudflare Tunnel for public access and Tailscale for admin.
3. Finish what's half-built or stubbed.
4. Add features that make sense given the gaps found.
5. Completely redo the design — real design system, animated background, toasts, page transitions, a deliberate color palette — replacing the current ad hoc styling.
6. Fully verify everything actually works end to end — every API, the trained model, both Pis' cameras, live-feed frame rate, and the LED matrix hardware — not just "the code looks right."
7. Ship it: commit, push, and leave the README accurately describing the finished system.

Work through the phases below **in order**, and **pause after each phase** with a summary of what changed, before starting the next one. Don't run all seven phases unattended in a single pass.

---

## 2. Implementation research — read before touching any code

Before starting Phase 0, read this section in full. It's real technical research on how to correctly implement the trickiest fixes below — not just "what's broken," but "how this is actually solved in practice" — so nothing gets guessed at or half-implemented a second time.

### LED matrix (Phase 0 item, Phase 4 verification)

The disabled `RGBMatrixBackend` / pixel-mapping TODO (`display_manager.py:636`) is very likely a **known, already-solved Pi 5 problem**, not a bug unique to this codebase. Raspberry Pi 5 moved GPIO handling to a separate RP1 coprocessor, which broke the old direct-GPIO-bitbanging approach every HUB75 driver used to rely on. The two current fixes in the wild:
- `hzeller/rpi-rgb-led-matrix` (the library this project is almost certainly built on) now has **native Pi 5 support** via two backend flags: `--led-rp1-pio=0` (RIO backend, default — higher CPU usage, faster) and `--led-rp1-pio=1` (PIO backend — much lower CPU usage, uses the RP1 chip's programmable I/O blocks, similar to the RP2040). Before writing any new pixel-mapping code, try both flags against the existing panel — the mirroring/corruption issue this repo is fighting may simply be a wrong or missing `--led-rp1-pio` setting, not something requiring a custom fix.
- Adafruit's separate `Adafruit-Blinka-Raspberry-Pi5-Piomatter` ("PioMatter") library is an alternative if the above doesn't resolve it — same underlying RP1 PIO approach, different implementation, still alpha-stage as of research but actively maintained.
- If flicker (not mirroring) is the remaining symptom, `--led-gpio-mapping=adafruit-hat-pwm` and the "optimized kernel" install documented in the same repo's README are the standard next step.

**The two confirmed live bugs (Pi 4 intermittently shows garbage instead of proper text; Pi 5 shows garbage on update) point at two different, well-documented problems — don't treat them as one issue, and don't assume Pi 4's is a static config error:**
- **Pi 4 — intermittently garbled/scrambled text, not consistently blank.** A *static* misconfiguration (wrong font path, wrong panel option) would fail the same way every single time — the fact that it's sometimes fine and sometimes garbage is the signature of a **GPIO signal-timing problem**, a specifically well-known Pi 4 issue with this class of library: Pi 4's CPU clocks GPIO writes faster than older Pis, and if `--led-slowdown-gpio` is tuned too low for the panel/cable/chain length in use, the panel's shift registers can't latch data reliably — producing intermittent scrambled output that gets worse under load (heavier CPU use elsewhere = more timing variance). Multiple reported cases match this exactly: identical code and hardware working cleanly on a Pi 3 and producing "gibberish" on a Pi 4, resolved (or substantially improved) by raising `--led-slowdown-gpio` (commonly needs 2-5 on a Pi 4, sometimes higher depending on chain length and adapter). Check, roughly in order of likelihood: (1) `--led-slowdown-gpio` value — try raising it incrementally; (2) whether the onboard sound module (`snd_bcm2835`) is still enabled — it shares hardware with this library and must be blacklisted; (3) whether a `1-Wire` overlay (`dtoverlay=w1-gpio`, often added for temperature sensors) is active and conflicting with the same GPIO pins; (4) the panel's input logic chips — `74HCT245`/`74AHCT245` are compatible with the Pi's 3.3V signal levels, `74HC245` is not and is a known cause of exactly this kind of intermittent corruption; (5) CPU core isolation (`isolcpus=3` in `/boot/firmware/cmdline.txt`) to keep display-refresh timing consistent under load. Only fall back to font-loading/permissions checks if none of the above resolves it.
- **Pi 5 — garbage specifically when content changes/updates, fine when static** is a different class of bug: a **canvas double-buffering bug**. Drawing new content directly onto the canvas object that's currently being scanned out to the panel (instead of onto an offscreen canvas) produces exactly this symptom — correct while nothing changes, torn/garbled mid-update. The library's own examples are explicit about this: create an offscreen buffer with `CreateFrameCanvas()`, draw the *new* frame onto that offscreen buffer only, then call `SwapOnVSync()` to atomically swap it in — never call `SetPixel`/`DrawText`/`Clear()` etc. directly on the canvas that's currently live. Audit every single update path in `display_manager.py` for Pi 5 and confirm none of them skip the offscreen-canvas step, including any "fast path" added for quick/frequent updates — that's usually exactly where this shortcut sneaks in.

### Live-feed architecture — decoupling video from inference (Phase 2's always-on 30 FPS requirement)

Current best practice for this exact situation (edge camera → server → browser, needs both a smooth live view and periodic AI inference) is a **dual-loop architecture**: one high-frequency loop that only captures, encodes, and forwards frames for viewing, completely decoupled from a second, independent, lower-frequency loop that samples frames for inference. Concretely:
- The frame-forwarding loop (Pi capture → JPEG encode → Socket.IO `pi_frame` emit → client render) should run on its own timer/thread and never wait on an AI-service HTTP response.
- The AI-sampling loop should pull the latest available frame at its own pace (e.g. every Nth frame or every X ms) and post it to `/api/detect` independently — a slow or stalled `/api/detect` call must never block the live-view frame rate.
- For the actual transport, WebSocket-pushed JPEG frames (what this repo already does via `pi_frame`) is a legitimate low-latency approach for LAN/Cloudflare-Tunnel conditions — full WebRTC is only worth the added signaling/NAT-traversal complexity if sub-500ms glass-to-glass latency across the open internet is a real requirement, which doesn't sound like the case here. Don't over-engineer this into a WebRTC migration unless the 30 FPS goal genuinely can't be hit any other way.
- If frame drops under load are a problem, prefer "drop stale frames, always show the newest" over buffering/queueing — a queued-up backlog of old frames is worse than a skipped frame for a live monitoring view.

### Speed estimation — homography (Phase 1's split-brain decision)

The published literature on this is unanimous: **raw pixel-distance/frame-time speed (what `traffic_detector.py` currently does) is measurably less accurate than a homography-corrected approach** (what `inference/camera_calibration.py` already implements but doesn't feed into production) — perspective distortion means a vehicle far from the camera covers fewer pixels per real meter than one close to the camera, so uncorrected pixel-speed is systematically wrong depending on where in frame the vehicle is tracked. This strengthens the case in Phase 1 for wiring the homography implementation into production rather than deleting it. Practical calibration approach used across the research: pick 4+ known reference points on the road plane (lane markings, painted lines, a measured distance) and map them to their real-world coordinates via `cv2.getPerspectiveTransform` — exactly what `calibrate_perspective()` in `camera_calibration.py` already does. The main accuracy driver is **calibration point placement quality**, not the algorithm — so if this gets wired in, spend calibration effort on precise, well-distributed reference points on the actual Busay curve footage, not just 4 arbitrary clicks.

### Admin authentication (Phase 0 item)

The standard, minimal-footprint pattern for securing both the Express admin routes and the Socket.IO admin namespace is **JWT-based middleware on both layers**:
- HTTP side: a normal Express auth middleware checking a `Bearer` token on protected routes.
- Socket.IO side: `io.use((socket, next) => {...})` reading the token from `socket.handshake.auth.token` (sent during the client's connection handshake, not as a regular event), verifying it against the same JWT secret, and calling `next(new Error(...))` to reject the connection outright if invalid — this happens before any event handlers run, so an unauthenticated socket never reaches the admin/terminal event listeners at all.
- Keep it to one login → one short-lived JWT → both HTTP and socket auth use the same token/secret, rather than building two separate auth systems for the two transports.

### Crash/incident model training (Phase 2, out-of-band GPU job)

Published YOLOv8-based crash/accident detection work consistently reports strong results (accuracy in the high-80s/90s%, precision/recall in the 90s%) but flags two things directly relevant to this project's eventual training run:
- **Class imbalance is the main challenge**, since real accident frames are naturally rare relative to normal-traffic frames — comparable published datasets run roughly 40-60% accident-positive frames when curated specifically for this task (not a naturally-occurring ratio, a deliberately balanced one), achieved via weighted sampling/loss during training rather than just throwing an imbalanced raw dataset at the model.
- When the accident dataset is eventually trained (`training/train.py --dataset accident`, flagged as out-of-scope for Claude Code itself), check the class balance of `datasets/processed/busay_accident_detection/` first — if it's heavily skewed toward one class, that's worth flagging to the user before spending the GPU hours, since it will most likely undertrain on the rare "accident" class as-is.

### Design system references (already covered in Phase 3, restated here for completeness)

HeroUI Pro's dashboard templates, HeroUI v3's native Toast, `react-bits` for the animated background, and Framer Motion `AnimatePresence` for route transitions — see Phase 3 below for the full detail; nothing new to research beyond what's already specified there.

---

1. **Unauthenticated remote command execution.** `server/node-service/src/server.ts:187-274` lets any Socket.IO client run arbitrary shell commands on the Node server *and*, via `pi_agent.py`, on both physical Raspberry Pis — no login, token, or origin check anywhere. Add real authentication in front of the `/admin` route and the Socket.IO admin namespace (server-side check, not just hiding the client-side link). Gate both the "server" and "pi4"/"pi5" command-relay targets.
2. **CORS wildcard.** `server.ts:38` hardcodes `cors({ origin: "*" })`. Replace with an explicit origin allowlist read from a real env var (and make `CORS_ORIGIN` in `.env.example` actually do something — right now it's declared but ignored).
3. **Plaintext production secrets.** `render.env.txt` contains a live Aiven MySQL password and a live Supabase service-role key in cleartext. Rotate both credentials now, regardless of the Phase 0.5 hosting migration below — don't wait for the migration to stop the bleeding.
4. **Hardcoded machine-specific path.** The AI-service `.env`'s `TRAFFIC_MODEL_PATH` is an absolute Windows path on a different drive than this checkout. Make it relative/config-driven so it works on any machine.
5. **LED matrix — two distinct, confirmed bugs to fix (not just the disabled-backend TODO).** Both Pi 4 and Pi 5 are meant to drive an LED matrix (Pi 4 currently doesn't yet — see Phase 2). Two separate symptoms, likely two separate root causes — treat them as such, don't assume one fix solves both:
   - **Pi 4: intermittently shows garbage/scrambled output instead of proper text** (not consistently blank or consistently broken). This pattern — same code, sometimes fine, sometimes garbled — is the signature of a GPIO signal-timing problem, not a static config error: Pi 4's faster CPU can outpace the panel's ability to latch data if `--led-slowdown-gpio` is tuned too low for the panel/chain in use. Start by raising `--led-slowdown-gpio` incrementally; also check for the onboard sound module (`snd_bcm2835`) or a `1-Wire` overlay conflicting with the same GPIO pins, and confirm the panel's input logic chips are `74HCT245`/`74AHCT245` (3.3V-compatible) rather than `74HC245`. See §2 for the full diagnosis and evidence.
   - **Pi 5: works, but shows garbage specifically when it changes or updates** (not when static). This is a classic **canvas-swap/tearing bug**: drawing new content directly onto the currently-displayed canvas while it's actively being scanned out produces exactly this symptom — fine when static, corrupted mid-update. The fix is to confirm every update goes through the double-buffer pattern correctly — draw the new frame onto an **offscreen** canvas via `CreateFrameCanvas()`, then swap it in atomically with `SwapOnVSync()` — never call `SetPixel`/`DrawText`/etc. directly on the canvas object currently being displayed. Audit every code path in `display_manager.py` that updates the Pi 5 display and confirm none of them skip the offscreen-canvas step, even for "quick" updates.
6. **`training/train.py`'s `DATASETS_DIR` bug** (`train.py:47-48`) assumes the repo's parent folder is literally named `Road_Sentinel`. Fix to resolve relative to the script's own location, not an assumed folder name.
7. **Schema drift.** `server/database/mysql_schema.sql` and the migrations that actually run (`node-service/src/database/migrate.ts`) disagree on the hourly-analytics table name (`analytics_hourly` vs `hourly_analytics`) and on whether `recordings` exists at all. Pick `migrate.ts` as authoritative and regenerate `mysql_schema.sql` to match it exactly (or delete the static file and replace it with a real `mysqldump`/schema-export script).
8. **Camera B RTSP IP mismatch — treat as a moving target, not a one-time fix.** `setup_pi5.sh`/`camera_reboot_autostart_setup.sh` use `.108`; `node-service/src/database/seed.ts` seeds `.102`. The user can't currently verify which is correct on the physical network, **and the IP may genuinely change over time** (DHCP-assigned, not static) — so "pick the right one and hardcode it" isn't actually a durable fix here. Do this instead:
   - **Immediate:** make the three inconsistent references at least agree with each other for now — use whichever value is currently deployed and working on the running Pi 5 systemd service as the tiebreaker if determinable from the checkout; otherwise pick one consistently and clearly comment that it's provisional pending physical verification.
   - **Real fix — lean on the auto-discovery that already exists rather than fighting hardcoded IPs:** `camera_sender.py` already has ONVIF WS-Discovery + RTSP port-scanning auto-discovery that kicks in after repeated connection failures (`DISCOVERY_AFTER_FAILURES = 3`, per `documentation.md §9`). Strengthen this into the primary recovery path rather than a rarely-hit fallback — e.g., persist whatever IP discovery finds back to Node's camera config (so `seed.ts`'s hardcoded default stops being the thing that matters at all), and confirm the discovery logic actually gets exercised and verified in Phase 4, not just left as unverified dead code.
   - **Recommend to the user (not something Claude Code can do remotely):** set a DHCP reservation for Camera B's MAC address on the home router, so it always gets the same IP going forward. This is the actual permanent fix — it eliminates the whole class of "which IP is correct" problem rather than requiring the auto-discovery fallback to constantly compensate for an address that keeps drifting. Surface this as a suggested follow-up in whatever summary comes out of Phase 0, since it's outside what Claude Code itself can configure.
9. **Confidence threshold disagreement.** `ai-service/.env.example` (0.75) vs. live `.env` (0.5) vs. `seed.ts` (0.5) vs. `mysql_schema.sql` (0.75). Pick one default and align all four.

---

## 3.5 Phase 0.5 — Hosting migration: Render → self-hosted PC (RTX 3060 Ti)

New requirement, folded in alongside Phase 0 since it changes where several of Phase 0's fixes actually apply. Do this after Phase 0's code-level fixes, before Phase 1.

**What's moving:** `render.env.txt` confirms Node service + Next.js client are currently deployed on Render. Move both off Render entirely onto the user's dedicated PC (RTX 3060 Ti). The AI service's media storage already appears to run locally with a Cloudflare-Tunnel-fronted `STORAGE_BASE_URL` (per `documentation.md §7.1`/§12) — confirm that's still correct post-migration rather than rebuilding it; it may already be most of the way there. **MySQL also moves off Aiven onto the same PC** (updated scope — everything self-hosts now, not just the app services).

**Database migration — Aiven → self-hosted MySQL on `irm-pc`:**
- Install MySQL Community Server natively on Windows (the official MySQL installer for Windows), or run it in a container via Docker Desktop if that's already installed on the PC — either is fine, pick whichever is already set up or simpler for the user, don't assume.
- Migrate the data for real, don't just point at an empty new database: `mysqldump` the existing Aiven database, restore it into the new local instance, and verify row counts match before cutting over.
- Once local MySQL is confirmed working with the restored data, update `server/node-service`'s `DB_HOST`/`PORT`/`USER`/`PASSWORD`/`NAME`/`SSL` to point at the local instance (almost certainly `DB_HOST=localhost` or `127.0.0.1` now, `DB_SSL=false`, since it's the same machine as `node-service` — no TLS needed for a local loopback connection).
- **Since MySQL now only needs to be reachable from `node-service` on the same PC, it should never be exposed publicly at all** — not through the Cloudflare Tunnel, not even through Tailscale. Bind it to localhost only (or the PC's LAN interface at most, if something else on the local network genuinely needs it, which nothing currently does). This is a meaningful security improvement over the Aiven setup, not just a hosting change.
- Once the local database is verified working, the Aiven database can be fully decommissioned/deleted rather than just having its password rotated (Phase 0 item 3 still applies immediately regardless — rotate that credential now — but plan to close the Aiven account/instance entirely once this migration is confirmed, rather than paying to keep an unused credential-rotated database around).
- Reconcile this with Phase 0 item 7 (schema drift, `mysql_schema.sql` vs `migrate.ts`): since a fresh local database is being stood up here, use `migrate.ts` (already confirmed as the authoritative source) to initialize it directly — don't restore the old Aiven dump *and* separately worry about which schema file is "correct"; the migration scripts define the real schema going forward.

**Public access — Cloudflare Tunnel (quick tunnel, accepted by the user):**
- User's call: a **quick tunnel** (`cloudflared tunnel --url ...`, no account/domain needed) is fine here — Tailscale gives an easy way to reconfigure/restart things if the tunnel needs to be brought back up, so the tradeoffs below are accepted rather than avoided. Don't push for a named tunnel unless the user changes their mind.
- Correcting one earlier overstatement: quick tunnels **do** support WebSocket traffic fine (Cloudflare's edge supports WebSockets on all plans, including through `trycloudflare.com`) — that's not actually a blocker for the live feed. The real, accurate tradeoffs to design around instead:
  - **The hostname changes every time `cloudflared` restarts.** Anything that references the current tunnel URL (a bookmark, a QR code, the AI service's `STORAGE_BASE_URL`, any hardcoded client config) needs to be re-pointed after a restart — treat this as a routine "reconfigure after reboot" step alongside the Windows auto-start work below, not a one-time setup detail.
  - **Quick tunnels are explicitly positioned by Cloudflare as testing/demo infrastructure, not production** — there are real user reports of intermittent flakiness with no uptime guarantee. Acceptable here given the accepted-risk framing already established for this whole hosting setup, but worth knowing going in.
  - **Capped at 200 concurrent requests** — almost certainly fine for a small thesis-project audience, just noting it exists.
  - **No Cloudflare Access (Zero Trust) gating**, since that requires a hostname on a zone you control — a quick tunnel's random `trycloudflare.com` subdomain doesn't qualify. This means the Phase 0 JWT auth is the *only* gate in front of `/admin` (no free edge-level second factor available with this choice) — make sure that auth is solid, since there's no second layer behind it.
- Route both the Next.js client and the Node API through the same tunnel process (multiple local ports/hostnames can share one `cloudflared` instance).

**Confirmed hardware, per the user's Tailscale admin console:** the target machine is `irm-pc`, currently online and running **Windows** (not Linux, which the guidance below originally assumed — corrected here). The laptop (`minniedumpor`) is the admin machine, but its Tailscale connection has expired and needs re-authentication (`tailscale up` / re-sign-in) before it can reach `irm-pc` at all — do this first, it's a blocker for everything else in this phase.

**Remote administration — Tailscale (Windows target + both Raspberry Pis, corrected/expanded):**
- **Purpose, stated by the user directly — worth being explicit about since it shapes how this gets set up:** Tailscale exists so the user can run terminal commands on `irm-pc` from anywhere (not just the home network — that's the whole point of a Tailscale mesh, it works the same over the public internet as it does at home), and so **Claude Code itself, running wherever the user is working from, can reach into `irm-pc` to actually run and test things there** — not just review the checked-out code statically. This is the mechanism Phase 4's live functional audit depends on: verifying real API responses, confirming the trained model actually loads, checking the live camera feed's real FPS, and testing the LED matrix fixes all require *executing commands on the actual PC*, not just reading its code — set this access up with that end use in mind, not just as an occasional admin convenience.
- This access is for the user's own terminal work on `irm-pc` and both Pis — never the path public traffic or the Pis' camera feeds travel through. Keep this cleanly separate from the Cloudflare Tunnel's public-facing job.
- **New scope: add both Pi 4 and Pi 5 to the same tailnet**, not just the PC. Install with `curl -fsSL https://tailscale.com/install.sh | sh` then `tailscale up` on each Pi, authenticating them to the same account as `irm-pc`/`minniedumpor`. This gives direct, private access to each Pi individually for logs and restarts (`camera_sender.py`, `display_manager.py`) without going through the Node service at all — and, same as above, gives Claude Code a way to actually test against the physical Pis and their hardware, not just their code.
- **Unlike the Windows PC, the Raspberry Pis run Linux — Tailscale's own SSH-server feature works natively there.** Enable it per-device (`tailscale up --ssh`) rather than needing the OpenSSH-over-Tailscale-IP workaround required for the Windows PC above; it's simpler on the Pis specifically because of the OS difference.
- **Tailscale's own SSH-server feature does not run on Windows** (confirmed — it's Linux/macOS only as of current Tailscale docs; Windows can only be an SSH *client* through it, not a server). The correct approach on `irm-pc` specifically is: enable Windows's built-in **OpenSSH Server** (an optional Windows feature, no third-party install needed — `Add-WindowsCapability -Online -Name OpenSSH.Server`), then connect to it using `irm-pc`'s Tailscale IP from the laptop (or from wherever Claude Code is running). The connection is still fully encrypted end-to-end by Tailscale's WireGuard mesh even though it's plain OpenSSH doing the actual login on the Windows side.
- With direct Tailscale access now reaching the Pis individually, revisit whether the existing Socket.IO-based remote-shell relay to the Pis (`pi_agent.py`'s `subprocess.Popen` path, flagged unauthenticated in Phase 0 item 1) is still needed at all — same question as for the in-app admin terminal below, just extended to the Pi-command-relay path specifically. Ask the user rather than deciding silently: (a) keep it as a convenience, now properly authenticated per Phase 0, or (b) retire the Pi-command-relay feature entirely now that Tailscale SSH reaches each Pi directly and more safely.
- Separately, once OpenSSH-over-Tailscale exists for `irm-pc` itself, revisit whether the in-app browser-based admin terminal for the *server* (the Socket.IO remote-shell feature flagged in Phase 0 item 1) is still needed at all. Same two options as above, asked separately since it's a different code path: (a) keep the in-app terminal, now properly authenticated per Phase 0, as a convenience for mobile/browser access; or (b) retire it entirely in favor of Tailscale-network access, which removes an entire attack surface instead of just locking it down.

**Accepted risk (explicitly acknowledged by the user, don't relitigate):** no failover/redundancy is being built for this PC — if it restarts or loses power, the stack needs manual reinitialization, and that's accepted. One low-effort thing worth doing anyway, since it doesn't conflict with that acceptance: configure the Node service, AI service, and `cloudflared` to **auto-start on boot on Windows** — not systemd (that's Linux-only, doesn't apply here). Use either Windows's native Task Scheduler ("run at startup," works for a single script/batch launcher) or wrap each long-running process as a proper Windows Service (via `nssm` — the Non-Sucking Service Manager, the standard tool for this — or Node's own `node-windows`/`pm2` with `pm2-installer` for Windows service registration) so an ordinary reboot recovers on its own without a logged-in user session being required. Flag it as optional, not mandatory.

**Documentation impact:** this changes what Phase 5's final README needs to describe (real hosting setup, real domain, real access model for admin) — carry this section's outcome forward into that pass rather than writing Phase 5's README before this is settled.

---

## 4. Phase 1 — Functionality correctness pass

- **Wire every decorative, no-op button**: Analytics "Export PDF"/"Export CSV", Cameras "Open Calibration Tool"/"View Calibration Guide", History "Play", Reports "Download", Settings "Save All Settings"/"Reset to Defaults".
- **Resolve the speed-estimation split-brain.** Production (`traffic_detector.py:63-122`) uses raw pixel-distance/Δt speed. A real homography-based implementation already exists but is disconnected (`inference/camera_calibration.py`). Either wire the homography version into the production AI service and connect the client's Calibration Tool button to it, or make a deliberate decision to keep the simple version and delete the orphaned file — don't leave both half-present.
- **Decide the incident/crash model's fate.** No trained crash model exists anywhere in the repo; `IncidentDetector` always runs its brightness-variance heuristic fallback (explicitly labeled "simplified example" in source). Either flag this clearly in the API response (`isHeuristic: true`) so the client can visibly label these as estimated rather than real detections, or treat training the real model as a Phase 2 task (see below) — don't present heuristic output as equivalent to a real detection in the UI.
- **Consolidate duplicate detector-runner logic.** Three independent "run a detector against a video/image and draw boxes" implementations exist (`training/validate.py`, `testing/test_video.py`/`test_images.py`, `inference/speed_detection.py`), with no shared code and no shared model. Merge into one shared module, or clearly mark which is authoritative for which purpose.
- **Clean up the config surface.** `CORS_ORIGIN`, `LOG_FILE`, `FRAME_PROCESSING_RATE`, `VIDEO_RECORDING_ENABLED`, `MAX_RECONNECT_ATTEMPTS` are declared in `.env.example` but never read in code. Either wire them to real behavior or remove them so the env file stops lying about what's configurable.
- **Remove unused dependencies** (`@supabase/supabase-js`, `node-rtsp-stream`, `fluent-ffmpeg`) from `node-service/package.json` — confirmed zero imports.
- **Fix every doc** — `README.md`, `PROJECT_STRUCTURE.md`, `START_HERE.md`, `TRAINING_GUIDE.md`, `CAMERA_TEST_GUIDE.md`, plus the in-folder `server/ai-service/README.md`, `training/README.md`, `raspi_scripts/README.md` — using `documentation.md §15 Doc Drift Log` as the exact fix list (every wrong path, renamed script, and missing file is itemized there).
- **LED matrix backend choice.** Once the two confirmed bugs from Phase 0 (item 5) are fixed, decide and document which backend (`RGBMatrixBackend` Python bindings vs. the `led-image-viewer` subprocess fallback) is the intentional long-term approach for both Pis, and record that decision in `raspi_scripts/README.md` so it isn't re-litigated later.

---

## 5. Phase 2 — Feature completion & new features

**Complete the stubbed features:**
- Real `recordings` table + backend for the History page (currently a hardcoded fixture array with no `fetch` call at all) — `video_url`/`thumbnail_url`/`vehicle_count` tied to real incident/detection records.
- Real backend for the Reports page (also pure fixture) — generate from actual analytics data; implement the PDF/CSV export buttons for real.
- A settings page actually wired to persisted config (currently no-op switches with no state).
- Night-vision/IR auto-switching brought into the current production `camera_sender.py` path — it currently only exists in the legacy, separately-maintained autostart script (`set_ir_auto_all.py`).
- **Build out Pi 4's LED matrix to match Pi 5 (confirmed scope, not a question).** Pi 4 currently has no LED matrix at all; the goal is symmetric hardware — both Pis running the same `display_manager.py` driver, same status-text behavior, once the Phase 0 bug fixes (correct `RGBMatrixOptions`/font-loading/root-privilege setup for Pi 4, and the offscreen-canvas/`SwapOnVSync` fix for Pi 5's update-time corruption) are in place. Treat this as bringing up a second, correctly-configured instance of the already-working driver — reuse Pi 5's now-fixed code path rather than writing a second, separate implementation.
- **Always-on 30 FPS live feed.** `camera_sender.py` targets 30 FPS capture, but the AI-detection calls that ride along the same loop throttle it in practice, and the client falls back to MJPEG if no WebSocket frame arrives within 5s (`useMjpegFallback`, `components/video-feed.tsx`). Decouple the live-view path from the AI-sampling path completely: the 30 FPS feed (Pi capture → Socket.IO `pi_frame` → client render) should never be gated by how fast `/api/detect` responds. AI detection can keep running at its own lower, independent cadence. Profile the full pipeline (capture → JPEG encode → Socket.IO emit → client render) and fix whichever stage is the actual bottleneck so the live view holds a genuine, sustained 30 FPS rather than a best-effort target.

**New features worth adding, given what the audit surfaced** (propose these to the user rather than silently building all of them — confirm scope first):
- Adaptive detection sampling: throttle AI-service calls based on scene activity instead of a fixed capture rate, to reduce load when the road is empty.
- A simple critical-incident alert hook (email/SMS/webhook) so a "critical" severity incident notifies someone instead of only appearing in the dashboard.
- A model-confidence/drift indicator in the UI once a real crash model exists, so heuristic-era history is visibly distinguishable from real-model-era history.
- A public, no-admin-access "live status" page for the community near the blind curve (safe vs. caution vs. incoming-vehicle state only — no camera feed, no admin surface).
- Thesis-reporting exports: a speed-violation-by-time-bucket export suited for write-up figures.

**Explicitly out of scope for Claude Code to complete unattended:** actually training the crash/incident model (`training/train.py --dataset accident`) is a GPU job — the merged dataset is ready (`datasets/processed/busay_accident_detection/`), so flag this as "ready to run" and let the user kick it off themselves rather than attempting to fake or skip it.

---

## 6. Phase 3 — Complete design overhaul

**Problem being solved:** styling today is ad hoc per-page Tailwind (`bg-white/10 backdrop-blur-md`, hardcoded `#1B1931`/`#ED9E59` hex values scattered per file), inherited from an unrenamed HeroUI starter template (`"name": "next-app-template"` still in `package.json`), with four leftover unstyled template pages (`/about`, `/blog`, `/pricing`, `/docs`) never wired to the app at all. This phase replaces all of it with one deliberate system.

### Reference material (use for real technique/structure, not just namedropping)

**Stay on HeroUI — don't switch component libraries.** The client already runs Next.js 15 + HeroUI (`client/web/package.json` still says `"next-app-template"`, but the dependency itself is real and current). Swapping to shadcn mid-revamp would mean re-doing every component from scratch for no functional gain; HeroUI already covers everything needed here and has matured a lot:

- **Dashboard structure/interaction reference** — HeroUI Pro's own template gallery (`heroui.pro/docs/react/templates`) ships a real analytics-dashboard template (orders, tracker, settings, help pages) built natively in HeroUI — use it as the structural reference for navigation, density, and page composition instead of a shadcn-based admin template.
- **Toasts** — HeroUI v3 ships a native `Toast` component (stacking, auto-dismiss, promise support, swipe-to-dismiss) — use this directly rather than pulling in `sonner` or any shadcn-ecosystem toast library. One less dependency, and it already matches the rest of the component styling.
- **Page transitions** — HeroUI v3 moved off Framer Motion to native CSS transitions/keyframes for its own components (lighter, GPU-accelerated), but for *route-level* page transitions specifically, still wrap the Next.js App Router layout in Framer Motion's `AnimatePresence` (or the View Transitions API if the Next.js version supports it) — HeroUI doesn't cover cross-route transitions itself. Framer Motion is a fine, isolated dependency for just this one job.
- **Animated background** — use `react-bits` (github.com/DavidHDev/react-bits, ~26k★, framework-agnostic copy-paste components — installed via its own CLI/jsrepo, not tied to shadcn) for the aurora/mesh-gradient background specifically. It's plain React + Tailwind + CSS/Framer Motion, so it drops into a HeroUI project cleanly with no shadcn dependency at all. Pick **one** background style from its Backgrounds category (aurora or gradient-mesh, not a particle.js loop or looping video) and make sure it respects `prefers-reduced-motion`.
- **Anti-"AI slop" grounding** — pull the actual checklist from `Trystan-SA/claude-design-system-prompt`'s `ai-slop-check` skill and the design-token approach from `VoltAgent/awesome-claude-design`'s `DESIGN.md` format (both are library-agnostic advice, not shadcn-specific). Concretely avoid: Inter/Roboto/system-font-only typography, purple gradients on dark or white backgrounds, generic card-grid layouts with no context-specific character, motion that's decorative rather than meaningful.

### Color palette — recommendation (not literal red-as-primary)

Making the whole UI red is a poor fit **specifically because this is an incident-safety system**: if red is the dominant chrome color, it stops meaning "critical incident" the moment the app loads — undermining the exact RAG (red/amber/green) severity signaling the dashboard needs for incidents and speeding severity. Dashboard-color best practice (and accessibility research on red/amber confusion for color-blind users) both point the same direction: reserve red for one meaning only, and build the primary identity around something else.

**Recommended — "Night Watch":** a dark, near-black operations-monitoring base (comfortable for a 24/7 live-view screen) with **amber/traffic-signal yellow** as the primary brand accent — it keeps the road-safety story without triggering constant alarm, and is visually distinct from the red reserved for real incidents.

| Role | Color | Hex | Used for |
|---|---|---|---|
| Background | Near-black slate | `#0B0E14` | Page background |
| Surface | Dark slate | `#141922` | Cards, panels, sidebar |
| Primary/brand | Amber | `#F2B33D` | Nav highlights, primary buttons, logo, active states |
| Success | Teal-green | `#3DDC97` | Normal traffic flow, resolved status |
| Warning | Orange | `#F2994A` | Warning severity — distinct from both amber-brand and red-critical |
| Critical | Red | `#E5484D` | **Only** incident severity + destructive admin actions — nowhere else |
| Info | Blue | `#5B9DF5` | Links, informational badges |
| Text primary | Off-white | `#E8EAED` | Body text on dark surfaces |
| Text secondary | Muted gray-blue | `#8A93A6` | Secondary/meta text |

**Alternative** if you want a cooler identity instead of amber: swap the brand accent for **cyan/teal** (`#22D3B4`) — common in monitoring/ops tools, reads as "always-on" rather than "road/safety," and isolates red even more completely as the only warm color in the whole interface.

Pick one of these two before Claude Code starts building the token file, so it doesn't have to guess.

### Design deliverables for this phase

1. One design-tokens file (Tailwind config or CSS variables) — colors, type scale, spacing scale, motion durations/easings — used everywhere, replacing every scattered hex value in the codebase.
2. One deliberate font pairing (not Inter alone) — a heading face with actual character for a technical/safety-monitoring tool, plus a clean, readable body face. State the choice and the reasoning.
3. A rebuilt shared layout shell (sidebar/nav, page container, animated background) applied identically across every real page: dashboard, monitor, analytics, incidents, cameras, admin, history, reports, settings. Delete the unused HeroUI starter leftovers (`/about`, `/blog`, `/pricing`, `/docs`) rather than restyling them.
4. Toasts wired to real events — incident created, detection/connection error, settings saved — not decorative.
5. A page-transition wrapper applied at the layout level, covering every route.
6. A short (2-3 sentence) before/after design-rationale note, useful for referencing the design decision in the thesis write-up.

---

## 7. Phase 4 — Full functional audit & hardware verification

This is a live verification pass, not another read-through of the code. Actually run things and confirm they work, don't just re-check that the code looks correct. Everything in this phase runs **on the actual hardware** — `irm-pc` and both Raspberry Pis — over the Tailscale access set up in Phase 0.5 (that's exactly what that access exists for). If Tailscale connectivity to any of these devices isn't in place yet when this phase starts, that's a blocker — go back and finish Phase 0.5's Tailscale setup first rather than trying to audit from a local checkout alone.

**API audit — every endpoint, actually called:**
- AI service (`server/ai-service`): `GET /`, `GET /health`, `POST /api/detect`, `POST /api/detect/traffic`, `POST /api/detect/incidents`, `POST /api/storage/upload`, `DELETE /api/storage/delete`, `GET /api/storage/list`, `GET /api/stats` — call each one for real (reuse/extend `testing/test_ai.py`) and confirm the response shape matches what's documented. Flag anything that doesn't.
- Node service (`server/node-service`): every route under `routes/cameras.ts`, `routes/detections.ts`, `routes/incidents.ts`, `routes/analytics.ts`, plus the Socket.IO events (`subscribe_camera`, `pi_frame`, `pi_register`, `pi_output`, admin terminal events). Confirm each actually does what §7.2 of `documentation.md` says it does, post-Phase-0/1 fixes.
- Confirm the CORS and admin-auth fixes from Phase 0 actually block what they're supposed to block (attempt an unauthenticated admin action and confirm it's rejected).

**Trained model verification — confirm what's already been trained is still there and actually wired:**
- Confirm the existing trained vehicle model weight (`models/runs/vehicle/vehicle_yolo26n_20260203_032528/weights/best.pt`, or wherever it now lives after any Phase 0 path fix) is still present on disk and loads successfully via `ultralytics.YOLO`.
- Confirm `TrafficDetector` is actually loading *that* weight in the running AI service — not silently falling back to stock `yolov8n.pt` (`traffic_detector.py`'s fallback path). Log or surface which model is actually active on startup so this is never ambiguous again.
- Re-run inference against a known test image/video and sanity-check detections look right (reuse `testing/test_images.py` / `test_video.py`).
- If a crash/incident model has since been trained (per the Phase 2 "out of scope" GPU job), verify that weight the same way; if not, reconfirm the API is still correctly labeling incident output as heuristic (`isHeuristic: true` from Phase 1).

**Raspberry Pi camera check — both Pis, live:**
- Confirm Pi 4 (Camera A) and Pi 5 (Camera B) RTSP streams are both reachable and `camera_sender.py` connects and stays connected (including the ONVIF/RTSP auto-discovery fallback after repeated failures).
- Confirm the always-on 30 FPS live feed built in Phase 2 is genuinely sustained end to end for **both** cameras under normal conditions — measure actual delivered FPS client-side (`components/video-feed.tsx` already has FPS measurement logic; use it), not just the capture-side target.
- Verify Camera B's auto-discovery recovery path actually works rather than assuming a fixed IP is correct (per the Phase 0 item 8 fix) — since the IP isn't guaranteed stable, this is more important to confirm than any single hardcoded value.

**LED matrix & per-Pi camera hardware review:**
- Both Pi 4 and Pi 5 are confirmed to need an LED matrix (symmetric hardware, per the user). Confirm Pi 4's matrix, built in Phase 2, is actually up and displaying correctly — not just present in code.
- Specifically re-verify both Phase 0 LED bug fixes hold up under a real run: Pi 4 displays legible text consistently across repeated runs (not just occasionally, given the original symptom was intermittent), and Pi 5 no longer shows corruption when its display content changes or updates (test this by actually triggering several content changes back-to-back, not just checking a static screen).
- Confirm the earlier `RGBMatrixBackend` pixel-mapping TODO (`display_manager.py:636`) and any remaining RP1 PWM timing-drift symptoms are resolved for both Pis now that both are active.
- Note concrete improvement opportunities found during this hardware pass: e.g. daylight readability/brightness of the LED panel, restart/watchdog reliability, physical camera mounting/angle relative to the calibration assumptions in `inference/camera_calibration.py`, and any single point of failure (e.g. one Pi handling both a camera and the LED driver with no redundancy).

Produce a short audit report (pass/fail per item above, with evidence) before moving to Phase 5. If anything fails, fix it and re-verify before proceeding — don't ship on a failed audit.

---

## 8. Phase 5 — Commit, push, and final documentation

- Stage and commit all changes from Phases 0-4, with clear, logically-grouped commit messages (e.g. one commit per phase, or per major fix, rather than one giant commit) so the history stays readable for the thesis write-up.
- Push to `origin/main` — but only after Phase 4's audit passes. Don't push unresolved failures.
- Do a **final** pass on `README.md` — beyond the Phase 1 drift fixes (which only corrected stale paths/commands), rewrite it to accurately describe the finished, revamped system: the real architecture, the real feature set post-Phase-2, the new design system, and current setup instructions that match reality. This is a fresh top-to-bottom pass, not just patching the old drift.
- Confirm `.gitignore` still correctly excludes secrets, model weights, and datasets before pushing (per the concerns raised in Phase 0).

---

## 9. Ground rules for the whole revamp

- Read §2 (Implementation research) before starting Phase 0 — it exists specifically so the trickiest fixes (LED matrix, live-feed architecture, homography, auth, crash-model training) aren't reinvented or guessed at from scratch.
- Work phase by phase. Summarize what changed and pause for review after each phase — don't run all seven in one unattended pass.
- Don't guess a placeholder domain for the Cloudflare Tunnel in Phase 0.5, and don't silently decide whether to keep or retire the in-app admin terminal — both need the user's input.
- Every functionality fix should cite the specific file/line it addresses (documentation.md already has most of these).
- If a Phase 2 decision needs the user's input (e.g. "keep incident detection heuristic for now, or is training it in scope this pass?"), stop and ask rather than guessing.
- Preserve everything already working correctly — vehicle detection, speed estimation, dual-camera live view, MySQL logging, LED display — this is a revamp, not a rewrite from scratch.
- Crash/incident model training itself is a GPU job for the user to run, not something to fake, skip silently, or simulate.
- Never push to `origin/main` with a failing or unverified Phase 4 audit — fix and re-verify first.