# LCD Pi 4 Setup — Step by Step

Quick-start guide for the HUB75 128×32 RGB LED matrix on **Raspberry Pi 4 Model B**.
For full reference (wiring diagrams, screen layouts, troubleshooting) see [README.md](README.md).

---

## What You Need

- Raspberry Pi 4 Model B (any RAM)
- ₱149 Chinese HUB75 adapter board ("Raspberry Pi to Hub75" from Shopee)
- 1× 128×32 HUB75 panel **or** 2× 64×32 HUB75 panels chained together
- Dedicated 5V 4A power supply for the panel(s) — **not** the Pi's USB power

---

## Step 1 — OS

Flash **Raspberry Pi OS (64-bit)** using Raspberry Pi Imager.
Enable SSH + set WiFi credentials in the customisation step before writing.

After first boot:

```bash
sudo apt update && sudo apt upgrade -y
sudo reboot
```

---

## Step 2 — Copy scripts to Pi

From your dev machine (or clone the repo on the Pi directly):

```bash
# Option A — scp from dev machine
scp -r raspi_scripts/lcd_pi4 pi@<PI_IP>:~/raspi_scripts/lcd_pi4

# Option B — clone repo on the Pi
git clone https://github.com/<your-repo>/RoadSentinel.git ~/RoadSentinel
cp -r ~/RoadSentinel/raspi_scripts/lcd_pi4 ~/raspi_scripts/lcd_pi4
```

---

## Step 3 — Build and install the library

```bash
cd ~/raspi_scripts/lcd_pi4
bash install.sh
```

This takes about **2 minutes**. It will:
- Install `python3-dev`, `build-essential` via apt
- Clone `hzeller/rpi-rgb-led-matrix` from GitHub
- Create `~/venvs/led_venv`
- Compile and install the `rgbmatrix` Python bindings from C source via `pip`
- Install `Pillow` and `requests`

---

## Step 4 — Verify

```bash
source ~/venvs/led_venv/bin/activate
sudo $VIRTUAL_ENV/bin/python3 -c "from rgbmatrix import RGBMatrix; print('OK')"
```

Expected output: `OK`

> **Tip:** Always use `sudo $VIRTUAL_ENV/bin/python3` — not `sudo python3`.
> The venv Python is different from the system Python; sudo needs the full path.

---

## Step 5 — Connect the hardware

1. **Power off the Pi** before connecting anything.
2. Plug the ₱149 HUB75 adapter board onto the Pi's **40-pin GPIO header**.
3. Connect the **HUB75 ribbon cable** from the adapter to the panel's **INPUT** port.
   - If chaining two 64×32 panels: INPUT of Panel 1 → OUTPUT of Panel 1 → INPUT of Panel 2.
   - Look for the "IN" / "OUT" label or arrow on the panel PCB.
4. Connect the panel power leads (red = 5V, black = GND) to your **dedicated 5V PSU**.
   - Do NOT power the panel from the Pi's USB.
5. Power on the PSU, then boot the Pi.

---

## Step 6 — Test the display

```bash
source ~/venvs/led_venv/bin/activate
cd ~/raspi_scripts/lcd_pi4

sudo $VIRTUAL_ENV/bin/python3 display_manager.py --test
```

You should see:
1. **Color bars** (3 seconds) — confirms all RGB channels work
2. **Static test screen:**
   ```
   ROAD SENTINEL             [TST]
   A: ON   B: ON   SIMULATED
   Veh:999               45km/h
   192.168.8.x         0h00m
   ```
3. Every ~20 seconds — amber flashing alert (TEST mode indicator)

**If the display is garbled or flickering**, try adjusting the GPIO slowdown:
```bash
sudo $VIRTUAL_ENV/bin/python3 display_manager.py --test --slowdown 3
# or: --slowdown 5
```

---

## Step 7 — Run in real mode

Make sure the Node Service is running (on the Pi or on another machine).

```bash
# Node Service on the same Pi:
sudo $VIRTUAL_ENV/bin/python3 display_manager.py

# Node Service on another machine:
sudo $VIRTUAL_ENV/bin/python3 display_manager.py --api http://192.168.8.50:3001
```

---

## Step 8 — Autostart on boot

Add the display manager to `~/camera_scripts/launch_both_cameras.sh`,
after the existing `nohup` camera lines:

```bash
# ── LED matrix display ───────────────────────────────────────────────────────
source "$HOME/venvs/led_venv/bin/activate"
nohup sudo "$VIRTUAL_ENV/bin/python3" \
      "$HOME/raspi_scripts/lcd_pi4/display_manager.py" \
      --api "http://localhost:3001" \
      >> "$LOG_DIR/led_matrix.log" 2>&1 &
```

This runs automatically whenever `launch_both_cameras.sh` is called (which runs at
desktop login via the `.desktop` autostart entry installed by `camera_reboot_autostart_setup.sh`).

Check the log:
```bash
tail -f ~/camera_logs/led_matrix.log
```

---

## Common Problems

| Problem | Fix |
|---|---|
| `ModuleNotFoundError: rgbmatrix` | Run `bash install.sh` again — library must be compiled |
| `Permission denied: /dev/mem` | Add `sudo` before the command |
| Wrong Python used with sudo | Use `sudo $VIRTUAL_ENV/bin/python3`, not `sudo python3` |
| Color bars missing / blank panel | Check 5V PSU is connected to the panel, not the Pi USB |
| Garbled / flickering display | Try `--slowdown 3` or `--slowdown 5` (default is 4) |
| Only left or right half works | Ribbon cable between the two panels is loose or reversed |
| Stats show `N/A` | Node Service not reachable — verify `--api` URL and service status |
