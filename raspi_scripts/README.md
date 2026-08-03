# RoadSentinel — Raspberry Pi Scripts

Scripts and utilities that run **on the Raspberry Pis** at the Busay blind-curve installation site: Pi 4 (Camera A + LED) and Pi 5 (Camera B + LED) — symmetric hardware as of Phase 2.

---

## Files

| File | Purpose |
|---|---|
| `camera/camera_sender.py` | **Production camera pipeline** (both Pis) — RTSP → AI service → Node. Includes ONVIF/RTSP auto-discovery, homography-calibration forwarding, optional `--ir-auto` day/night switching, and optional `--record` local recording. |
| `display_manager.py` | **Current, unified LED matrix driver** (both Pis) — auto-detects Pi 4 vs. Pi 5 and picks the right backend. See [LED Matrix Status Display](#led-matrix-status-display) below. |
| `pi_agent.py` | Authenticated admin-terminal relay — connects to Node's `/admin` Socket.IO namespace with `PI_AGENT_TOKEN`, runs commands sent from the web dashboard. |
| `setup_pi4.sh` | systemd install: Camera A + LED (`roadsentinel-camera`, `roadsentinel-display`, `roadsentinel-agent`). |
| `setup_pi5.sh` | systemd install: Camera B + LED (`roadsentinel-camera`, `roadsentinel-display`, `roadsentinel-agent`). |
| `lcd_pi4/fix_gpio_timing.sh` | Pi 4 GPIO-timing diagnostic/fixer (sound module, 1-Wire overlay, core isolation) — see below. |
| `color_test.py`, `test_display.py` | LED hardware bring-up/diagnostic scripts. |
| `camera_reboot_autostart_setup.sh` (repo root) | **Legacy, separate** ffplay-based desktop-autostart path — see below. |
| `lcd/`, `lcd_pi4/` | Earlier, per-model LED drivers — **superseded** by the unified `display_manager.py`, kept for reference. |

---

## Setup

```bash
# Pi 4 (Camera A)
PI_AGENT_TOKEN=<value from server/node-service/.env> bash setup_pi4.sh

# Pi 5 (Camera B + LED)
PI_AGENT_TOKEN=<value from server/node-service/.env> bash setup_pi5.sh
```

`PI_AGENT_TOKEN` is required (Phase 0) — it must exactly match `PI_AGENT_TOKEN` in `server/node-service/.env`, or the Pi's admin-terminal relay is rejected by Node's `/admin` namespace.

**Camera B's RTSP IP is provisional, not guaranteed static** (DHCP-assigned). `camera_sender.py`'s auto-discovery (ONVIF WS-Discovery + port-554 subnet scan, triggered after 3 consecutive connection failures) is the real recovery mechanism — it now persists whatever IP it finds back to Node (`PUT /api/cameras/:id`), so the database becomes the source of truth instead of the hardcoded default in `setup_pi5.sh`/Node's `seed.ts`. A DHCP reservation for Camera B's MAC on your router would eliminate this class of problem entirely.

---

## `camera_reboot_autostart_setup.sh` — Legacy Path

This is a **separate, older** mechanism from the systemd services above: a one-time installer that wires two `ffplay`-based desktop preview windows into the Pi's desktop login session (not `camera_sender.py`, no AI/Node integration). It generates `~/camera_scripts/launch_both_cameras.sh` and installs `~/.config/autostart/roadsentinel-cameras.desktop`, and separately runs `set_ir_auto_all.py` (ONVIF) for day/night IR switching before starting the streams.

```bash
bash camera_reboot_autostart_setup.sh          # first-time setup
~/camera_scripts/test_launch_both_now.sh        # test without rebooting
~/camera_scripts/disable_camera_autostart.sh    # remove the autostart entry
```

Requires Desktop Autologin (`raspi-config → System Options → Boot / Auto Login → Desktop Autologin`). Camera IPs used here (`.104`/`.108`) are hardcoded in the generated script, independent of `camera_sender.py`'s auto-discovery — keep both paths' IPs in sync manually if you use this legacy path alongside the production one.

---

## LED Matrix Status Display

### Current backend choice (Phase 1 decision, recorded here per the revamp plan)

**`display_manager.py`** (this folder, not `lcd/` or `lcd_pi4/`) is the **single, unified driver for both Pis** — it auto-detects the Pi model (`/dev/pio0` presence) and picks a backend automatically:

| Pi | Default backend | Library | sudo? |
|---|---|---|---|
| **Pi 4** | `LedcatBackend` | hzeller `ledcat` (subprocess, piped frames) | Yes (`/dev/mem` GPIO) |
| **Pi 5** | `LedImageViewerBackend` | hzeller `led-image-viewer` (subprocess, PPM file) | No (`/dev/pio0` coprocessor) |

`lcd/` and `lcd_pi4/` contain earlier, per-model implementations that predate the unified driver (their own git history is placeholder-message-only commits) — **superseded**, kept only for reference. Install scripts still live under `lcd_pi4/install.sh` (builds the shared hzeller C library/binaries both backends above use) and `lcd/README.md`/`lcd_pi4/README.md` (hardware wiring notes, still accurate).

Both Pis use the same **₱149 HUB75 adapter board** (hzeller "regular"/Active3 GPIO mapping).

### Phase 0 bug fixes applied (code-level; hardware verification pending Tailscale access — see `Summarization.md`)

- **Pi 4 — intermittent garbled/scrambled output.** Signature of a GPIO signal-timing problem (Pi 4's CPU outpaces the panel's shift registers at low `--led-slowdown-gpio`). `display_manager.py`'s default raised **4 → 6**. If still garbled once verified on hardware, raise further and run `lcd_pi4/fix_gpio_timing.sh` (report-only by default; `--fix` applies) — it checks/fixes the other three common causes in order of likelihood: the onboard sound module (`snd_bcm2835`, shares hardware with the LED driver and must be blacklisted), a `1-Wire` overlay on the same GPIO pins, and CPU core isolation (`isolcpus`). The adapter board's input logic chips must be `74HCT245`/`74AHCT245` (3.3V-compatible) — `74HC245` causes exactly this symptom and can only be checked by reading the physical board.
- **Pi 5 — garbage specifically when content changes.** The *active* backend (`LedImageViewerBackend`) restarts a subprocess on every content change rather than using true double-buffering (a separate, correctly-double-buffered backend, `RGBMatrixBackend`, exists but is disabled for an unrelated reason — see below). Added a settle delay in `_restart()` between confirming the old process exited and launching the new one, to reduce the chance of the new process racing the RP1 coprocessor's leftover state. **This is a reasoned mitigation based on the git history's established RP1-timing-drift pattern, not a hardware-verified fix** — confirm on real hardware before relying on it.
- **`RGBMatrixBackend`** (Python bindings, real `CreateFrameCanvas()`/`SwapOnVSync()` double buffering — architecturally the better long-term choice for Pi 5) remains **disabled by default**. Audited in Phase 0: it has no offscreen-canvas violation: the disabling reason is a separate, unresolved bug — `SetImage` mirrors output on this display's 2-panel chain. Opt into it for testing with `--pi5-backend rgbmatrix` once hardware is reachable; do not flip the default without verifying the mirroring issue is actually resolved, since this is safety-relevant hardware.

### Testing

```bash
sudo python3 display_manager.py --test              # cycles fake alerts, no API needed
sudo python3 display_manager.py --pi 4               # force Pi 4 backend
       python3 display_manager.py --pi 5               # force Pi 5 backend (no sudo)
       python3 display_manager.py --pi 5 --pi5-backend rgbmatrix   # test the alternate backend
       python3 display_manager.py --emulator --test   # any OS, no Pi hardware (pip install RGBMatrixEmulator)
```
