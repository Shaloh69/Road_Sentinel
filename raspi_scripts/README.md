# RoadSentinel — Raspberry Pi Scripts

Scripts and utilities that run **on the Raspberry Pi** at the Busay blind-curve installation site.

---

## Files

| Script | Purpose |
|---|---|
| [`camera_reboot_autostart_setup.sh`](#camera_reboot_autostart_setupsh-analysis) | Installs both RTSP camera streams as a desktop autostart entry |
| [`lcd/display_manager.py`](lcd/display_manager.py) | HUB75 128×32 RGB LED matrix — **Raspberry Pi 5** (Adafruit PioMatter) |
| [`lcd_pi4/display_manager.py`](lcd_pi4/display_manager.py) | HUB75 128×32 RGB LED matrix — **Raspberry Pi 4 Model B** (hzeller rgbmatrix) |

---

## `camera_reboot_autostart_setup.sh` — Analysis

### What It Does

This script is a **one-time installer** run on the Raspberry Pi to wire the two IP cameras into the desktop login session. You run it once; after that the cameras start automatically on every boot.

### Generated Files (after running the script)

```
~/camera_scripts/
  live_cam.sh                  ← (must exist before running — plays one RTSP stream via ffplay)
  camera_env.sh                ← (must exist before running — environment vars)
  launch_both_cameras.sh       ← GENERATED: starts both cameras + IR auto mode
  disable_camera_autostart.sh  ← GENERATED: removes the .desktop entry
  test_launch_both_now.sh      ← GENERATED: test without rebooting
~/.config/autostart/
  roadsentinel-cameras.desktop ← GENERATED: desktop autostart entry
```

### Step-by-Step Breakdown

```
1. Validates that live_cam.sh and camera_env.sh exist in ~/camera_scripts/
2. Generates launch_both_cameras.sh:
   a. Waits CAMERA_BOOT_WAIT seconds (default 12) for network to settle
   b. Kills any previously running camera ffplay processes (stale streams)
   c. Calls set_ir_auto_all.py (ONVIF) to set IR mode on both cameras
   d. Launches live_cam.sh 1 sub  (Camera A — 192.168.8.104:554)
   e. Launches live_cam.sh 2 sub  (Camera B — 192.168.8.108:554)
3. Installs ~/.config/autostart/roadsentinel-cameras.desktop
   → This tells the LXDE/GNOME desktop session to run launch_both_cameras.sh at login
4. Creates disable_camera_autostart.sh (removes the .desktop file)
5. Creates test_launch_both_now.sh (sets CAMERA_BOOT_WAIT=0 for instant test)
```

### Camera IPs / Streams

| Camera | IP | RTSP Stream |
|---|---|---|
| Camera A (North) | `192.168.8.104` | `rtsp://192.168.8.104:554/cam/realmonitor` |
| Camera B (South) | `192.168.8.108` | `rtsp://192.168.8.108:554/cam/realmonitor` |

Both use the `sub` (sub-stream) channel — lower resolution, lower latency.

### IR Auto Mode (ONVIF)

Before starting the ffplay streams, `set_ir_auto_all.py` is called using the `~/venvs/onvif/` Python venv. This uses the ONVIF protocol to set both cameras to **auto IR** (day/night switching). Runs silently — errors go to `~/camera_logs/ir_auto.log`.

### Autostart Requirements

> The `.desktop` autostart file only works when Raspberry Pi OS boots to the **desktop with auto-login**.
> Configure this in: `raspi-config → System Options → Boot / Auto Login → Desktop Autologin`

### How to Run

```bash
# First time setup
bash camera_reboot_autostart_setup.sh

# Test immediately (no reboot)
~/camera_scripts/test_launch_both_now.sh

# Disable autostart later
~/camera_scripts/disable_camera_autostart.sh

# Check logs
tail -f ~/camera_logs/camera1.log
tail -f ~/camera_logs/camera2.log
tail -f ~/camera_logs/ir_auto.log
```

### Potential Issues / Notes

| Issue | Fix |
|---|---|
| `live_cam.sh` missing | Create it before running this script |
| Cameras don't appear after reboot | Make sure Desktop Autologin is enabled in raspi-config |
| IR script fails silently | Check `~/camera_logs/ir_auto.log` |
| `DISPLAY` not set | The script exports `DISPLAY=:0` — only works if the desktop is running |
| Network not ready at boot | Increase `CAMERA_BOOT_WAIT` (default 12s) in the environment |

---

## LED Matrix Status Display

Two separate implementations — pick the one matching your Pi:

| Pi model | Folder | Library | sudo needed? |
|---|---|---|---|
| **Raspberry Pi 5** | [`lcd/`](lcd/) | Adafruit PioMatter (`pip install`) | No |
| **Raspberry Pi 4 Model B** | [`lcd_pi4/`](lcd_pi4/) | hzeller rpi-rgb-led-matrix (build from source) | **Yes** |

Both use the same **₱149 Chinese HUB75 adapter board** (hzeller "regular" / active3 GPIO mapping)
and display identical screens — color-coded REAL/TEST alerts on a 128×32 full-RGB panel.

See [`lcd/README.md`](lcd/README.md) for Pi 5 setup.
See [`lcd_pi4/README.md`](lcd_pi4/README.md) for Pi 4 setup.
