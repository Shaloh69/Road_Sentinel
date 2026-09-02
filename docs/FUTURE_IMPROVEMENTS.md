# Road Sentinel — Future Improvements

A working checklist of what is left, ordered by whether it blocks deployment.

Status key: `[ ]` not started · `[~]` partially done · `[x]` done
Verification: ✅ confirmed on real hardware · 🟡 code-complete, unverified · ⛔ blocked

Last reviewed: 2026-09-03.

---

## 1. Blocks a real deployment

Things that would make the system dishonest or unsafe in the field.

- [ ] **Train the crash/incident model.** `incident_detector.py:118-161` currently
      falls back to a **brightness-variance heuristic** explicitly labelled
      "simplified example" in its own source. No `incident.pt` exists anywhere
      in the repo. The dashboard, the database schema and the sign's `STOP`
      state all present incident output as real detection.
      *This is the single largest gap between what the system claims and what
      it does.* It is a GPU training job — run deliberately, never in the
      background.
- [ ] **Verify the Pi 5 sign end-to-end.** The STM32 port compiles but has
      never been flashed to a board. Refresh rate, USB CDC enumeration and the
      panel mapping are all unproven on that hardware. ⛔ needs the board.
- [ ] **Re-verify the ESP32 sign after the library restructure.** Behaviour is
      unchanged and it compiles, but it has not been re-flashed since moving to
      `LEDMatrixDrivers/`. 🟡
- [ ] **Confirm the udev rule, connect handshake and liveness check** in
      `led_sign_bridge.py` against a real Pi. None have run on hardware. 🟡
- [ ] **Resolve the Camera B IP disagreement.** `setup_pi5.sh` and the legacy
      autostart script use `192.168.8.108`; `node-service`'s seeded DB default
      is `.102`. Whichever is stale silently points Camera B at the wrong
      device.
- [ ] **One clean end-to-end run on real hardware** — both cameras, both signs,
      dashboard, database — observed and recorded. Nothing below should be
      trusted until this exists.

---

## 2. Accuracy and correctness

- [ ] **Wire homography into production speed math.** A real implementation
      exists (`inference/camera_calibration.py`) but is a standalone script the
      server never calls. Production speed uses **uncorrected pixel distance**,
      which is systematically wrong with perspective — a vehicle far up the
      curve reads slower than one close to the camera.
- [ ] **Make the client's Calibration Tool functional.** The buttons in
      `app/cameras/page.tsx:311-327` are decorative — no handler.
- [ ] **Reconcile confidence thresholds.** Four sources disagree:
      `ai-service/.env.example` (0.75), the live `.env` (0.5), `seed.ts` (0.5),
      `mysql_schema.sql` (0.75). Pick one, make the rest derive from it.
- [ ] **Day/night handling.** ONVIF IR auto-switching exists only in the legacy
      `camera_reboot_autostart_setup.sh` path, not in the production
      `camera_sender.py`. No IR or low-light logic in the models themselves —
      and a blind curve at night is exactly when the system matters most.
- [ ] **Measure speed accuracy against ground truth.** No validation of the
      speed figures against a known reference exists. For a thesis this is the
      claim most likely to be challenged.

---

## 3. On-Pi inference (decided: Pi-only, no accelerator)

Moving detection onto the Pis so the sign keeps working when the network does
not. This is the strongest resilience win available: today, if Tailscale or
`irm-pc` is unreachable, the Pi has nothing to say and the sign falls back to
`NO DATA`.

- [ ] **Benchmark first, decide second.** Export `best.pt` to NCNN, run on both
      Pis at 640 and 320, record fps, CPU load and temperature under the full
      pipeline. Everything below depends on those numbers.
- [ ] **Export the model to NCNN.** Format matters more than the Pi generation:
      NCNN/ONNX-int8 is roughly 3× faster than Ultralytics PyTorch on ARM.
- [ ] **Drop the detection rate to ~5 fps.** A car at 60 km/h moves 16.7 m/s;
      at 5 fps that is a sample every 3.3 m — ample for *"is a vehicle
      coming."* 30 fps of inference is spending compute on a question that does
      not need it.
- [ ] **Stop forwarding every frame.** Local detection *removes* work: no
      per-frame JPEG POST to `irm-pc`. Send small detection JSON instead. The
      restructured pipeline is lighter than today's even with inference added.
- [ ] **Gate the MJPEG push on an actual viewer.** Node's Socket.IO already
      knows whether a dashboard client has the page open. Nobody watching →
      inference gets the whole Pi; someone watching → full-rate feed and
      detection briefly drops, which does not matter because a human is
      standing there looking at the camera.
- [ ] **Cap inference threads** (`OMP_NUM_THREADS=3`, or `taskset`) and run it
      as a separate lower-priority process. If inference saturates all four
      cores the capture thread is starved and the feed *stutters and tears*
      rather than smoothly slowing — which reads as broken far more than a
      steady 15 fps does.
- [ ] **Enable hardware H.264 decode on the Pi 4** (`h264_v4l2m2m`). Default
      OpenCV builds use software decode and ignore it. Note the asymmetry: the
      **Pi 5 has no hardware H.264 decoder at all** (it kept only HEVC), so the
      faster CPU is the one paying full price for decode. The Pi 4 has the
      lever and needs it more.
- [ ] **Record only on incident.** `cv2.VideoWriter` with `mp4v` is software
      MPEG-4 encoding on every frame while active — more expensive than the
      live feed itself.
- [ ] **Gate the incident model behind a vehicle trigger.** Two models running
      continuously will not fit on a Pi.
- [ ] **Plan for thermals.** Sustained four-core inference in a sealed outdoor
      enclosure in Cebu. Pi 4 throttles at 80 °C, Pi 5 at 85 °C and runs
      hotter. Ambient inside a closed box could sit at 45-55 °C. Heatsink,
      ventilation, or accept the throttling — but measure it.

**Expected outcome, from published benchmarks — not yet measured on this
hardware:**

| | Live feed | Detection |
|---|---|---|
| Pi 5 | 20-30 fps sustainable | ~5 fps |
| Pi 4 | ~15 fps | ~3 fps |

The Pi 4 is genuinely tight and is where a real trade-off lands. Treat both
rows as estimates until the benchmark above exists.

---

## 4. Web client

- [x] Live view (WebSocket binary frames, MJPEG fallback) ✅
- [x] Authentication and an authenticated `/admin` namespace ✅
- [~] **History and Reports pages** — now issue real `fetch` calls, but were
      built as fixture shells. Confirm every panel reads live data and none
      still renders hardcoded numbers.
- [ ] **Recording playback.** The `recordings` table now exists in
      `migrate.ts`; the client has no player.
- [ ] **Surface sign state in the dashboard.** The web UI cannot currently show
      what either physical sign is displaying — an obvious operator need, and
      an easy win now that `INFO` reports live firmware state.

---

## 5. LED signs

- [x] ESP32 sign: hand-written driver, legible text at 250 fps ✅
- [x] Portable core shared across boards ✅
- [x] STM32 Black Pill port written and compiling 🟡
- [ ] **Flash and verify the STM32 sign.** ⛔ needs the board.
- [ ] **STM32 DMA → GPIO BSRR.** A timer can trigger DMA2 to stream
      precomputed words straight into `GPIOx->BSRR`, driving the panel with
      almost no CPU. Must be DMA2 — GPIO is on AHB1 and DMA1 cannot reach it.
      The precomputed `bsrrData` table is already exactly the data such a
      stream needs, so this is a natural extension rather than a rewrite.
- [ ] **Binary Code Modulation** for 256 levels per channel instead of 2. Not
      needed for a warning sign — saturated colour is the right choice there —
      but it is the main gap if this ever becomes a general-purpose library.
      Watch the known pitfall: the most-significant bitplane stays lit for half
      the frame and reads as flicker at low brightness unless split.
- [ ] **Ambient brightness control.** The sign runs at a fixed brightness; a
      panel set for daylight is blinding at night on an unlit curve.
- [ ] **Publish as its own GitHub repository.** The core is panel-specific, not
      project-specific — only `mode_status.cpp` hard-codes Road Sentinel's
      states. There is no working public driver for 1/8-scan FM6124 panels, and
      the measurement method in `DEBUG_LOG.md` is the genuinely reusable part.

---

## 6. Codebase health

- [ ] **Fix `training/train.py`'s `DATASETS_DIR` resolution.** Assumes the
      repo's parent folder is literally named `Road_Sentinel`; fails with a
      misleading "Dataset not found" on any other checkout — including this one.
- [ ] **Deduplicate detector-visualisation code.** Three independent
      implementations of "run a detector on a video and draw boxes" exist
      across `training/validate.py`, `testing/test_video.py` and
      `inference/speed_detection.py`.
- [ ] **Single schema source of truth.** `migrate.ts` is authoritative;
      `mysql_schema.sql` is generated reference and the two have drifted. Either
      generate it automatically or delete it.
- [ ] **Remove unused dependencies** from `node-service/package.json`:
      `@supabase/supabase-js`, `node-rtsp-stream`, `fluent-ffmpeg` — zero
      imports anywhere in `src/`.
- [ ] **Deal with `models/runs/segment/`** — trained segmentation weights with
      no corresponding training code in `training/`. Orphaned artifact.
- [ ] **Retire or document `camera_reboot_autostart_setup.sh`.** A second,
      undocumented camera-launch path with its own hardcoded IPs, independent
      of the systemd services.
- [ ] **Fix the hardcoded absolute `TRAFFIC_MODEL_PATH`** in the AI service
      `.env` — a Windows path on a different drive root that breaks on any
      other machine.

---

## 7. Operations

- [x] Tailscale to both Pis and the server ✅
- [x] Systemd services with restart-on-failure ✅
- [x] udev rule pinning `/dev/roadsentinel-sign` 🟡
- [x] Headless WiFi re-provisioning portal ✅
- [ ] **Alerting.** Nothing notifies anyone when a camera, a Pi or a sign goes
      down. The system currently fails silently, which for a safety device is
      the wrong failure mode.
- [ ] **Log rotation** on both Pis — services append indefinitely.
- [ ] **Automated backup** of the MySQL database.
- [ ] **Document power-loss recovery.** Services restart, but nothing verifies
      that a Pi comes back cleanly from an unexpected cut — which is the normal
      case at a roadside installation.
- [ ] **A watchdog for the whole chain.** The sign's 15-second timeout protects
      against a dead Pi. Nothing protects against a Pi that is alive but whose
      camera has silently stopped delivering frames.

---

## 8. Deliberately not doing

Recorded so they are not re-proposed.

- **Hailo AI HAT+ / Coral accelerator** — would give 30 fps on-device inference
  comfortably. Ruled out by decision: Pi-only.
- **Driving the panel from Pi GPIO** — ~80 configurations failed on these
  1/8-scan FM6124 panels, hzeller's own `demo` failed identically, and the
  whole path has been deleted. Do not revisit without new hardware.
- **`line_decoder = TYPE595`** — tested; it broke the full-screen fill that
  works under binary addressing. These are not 595-type panels.
