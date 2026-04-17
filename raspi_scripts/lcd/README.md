# HUB75 128×32 RGB LED Matrix — Raspberry Pi 5 Setup Guide

Full-color RGB LED matrix panel for the Road Sentinel installation at Busay.
Shows live system status with color-coded **REAL** and **TEST** alerts.

---

## ⚠️ Pi 5 Important Note

> The old **hzeller/rpi-rgb-led-matrix** library **does NOT work on Raspberry Pi 5.**
>
> Pi 5 replaced the Broadcom direct GPIO with the **RP1 peripheral chip** — the old library relied on direct BCM GPIO access which no longer exists.
>
> **Use Adafruit PioMatter instead.** It drives HUB75 panels through the RP1's built-in PIO blocks (same concept as the RP2040). It IS pip-installable — no building from source needed.

---

## Hardware

### What You Need

| Item | Notes |
|---|---|
| Raspberry Pi 5 | Any RAM variant |
| **₱149 Chinese HUB75 adapter board** *(what you have)* | "Raspberry Pi to Hub75" adapter from Shopee — uses hzeller "regular" GPIO mapping. **No level shifter** — see voltage note below. |
| **OR Adafruit RGB Matrix Bonnet** | Has 74HCT245 level shifter built in. Use `--pinout bonnet`. ~₱550 / $9.95 |
| HUB75 RGB LED Matrix panel(s) | See panel configs below |
| 5V 4A+ power supply (per panel) | Use a **dedicated PSU** — do NOT power panels from the Pi USB |
| Short ribbon cable | Usually included with the panel |

### ⚠️ Voltage Note — ₱149 Adapter Board

The cheap ₱149 adapter is a **passive breakout only** — it has no level shifter.

- Raspberry Pi 5 GPIO outputs **3.3V**
- HUB75 panels expect **5V logic**

In practice most modern HUB75 panels accept 3.3V signals at their inputs and work fine, but:
- Colors may appear slightly dimmer or washed out vs 5V driving
- If the display shows garbled/flickering output, the panel requires true 5V logic — in that case add a 74HCT245 level shifter or switch to the Adafruit Bonnet

### Panel Configurations (both produce 128×32)

| Config | Panels | `display_manager.py` args |
|---|---|---|
| **A — Two 64×32 chained** (most common, easiest to source locally) | 2× P3/P4/P5 64×32 panels | *(default, no args needed)* |
| **B — Single 128×32 panel** | 1× 128×32 panel | same defaults since WIDTH=128 is hardcoded |

> P3, P4, or P5 pitch 64×32 panels are widely available on Shopee/Lazada (~₱400–700 each).

---

## Wiring

### Option A — ₱149 Chinese HUB75 Adapter Board (Default — what you have)

This is the **default pinout** (`active3`). No extra flag needed.

```
[Raspberry Pi 5]
      ↕  (plug adapter board onto 40-pin GPIO header)
[₱149 HUB75 Adapter Board]
      ↕  (HUB75 16-pin ribbon cable)
[Panel 1 INPUT]  →  [Panel 1 OUTPUT] → ribbon → [Panel 2 INPUT]
```

**GPIO Pin Mapping (active3 / hzeller regular):**

| HUB75 Signal | Pi GPIO | Pi Pin# |
|---|---|---|
| R1 | GPIO 11 | Pin 23 |
| G1 | GPIO 27 | Pin 13 |
| B1 | GPIO 7  | Pin 26 |
| R2 | GPIO 8  | Pin 24 |
| G2 | GPIO 9  | Pin 21 |
| B2 | GPIO 10 | Pin 19 |
| A  | GPIO 22 | Pin 15 |
| B  | GPIO 23 | Pin 16 |
| C  | GPIO 24 | Pin 18 |
| D  | GPIO 25 | Pin 22 |
| CLK | GPIO 17 | Pin 11 |
| LAT/STB | GPIO 4 | Pin 7 |
| OE  | GPIO 18 | Pin 12 |
| GND | GND | Pin 6/9/14/… |

The adapter board handles all of this — just plug it onto the 40-pin header and connect the HUB75 ribbon to the panel.

**Power:** Connect the panels' power leads (red=5V, black=GND) to a **separate 5V 4A PSU** — never the Pi USB.

### Option B — Adafruit RGB Matrix Bonnet

Use `--pinout bonnet` flag.

```
[Raspberry Pi 5]
      ↕  (plug bonnet directly onto 40-pin GPIO header)
[Adafruit RGB Matrix Bonnet]
      ↕  (HUB75 16-pin ribbon cable)
[Panel 1 INPUT]  →  [Panel 1 OUTPUT] → ribbon → [Panel 2 INPUT]
```

The bonnet includes a **74HCT245 level shifter** (3.3V → 5V). No voltage issues.

```bash
python3 display_manager.py --pinout bonnet
```

### Panel Chaining (two 64×32 → 128×32)

```
[Bonnet/Pi] ──ribbon──► [Panel 1 INPUT]   [Panel 1 OUTPUT] ──ribbon──► [Panel 2 INPUT]
```

Look for the **arrow or "IN"/"OUT"** label printed on the panel PCB — the input
and output ports look identical but are NOT interchangeable.

---

## Step-by-Step Setup on Raspberry Pi 5

### Step 1 — Flash Raspberry Pi OS (64-bit)

Use **Raspberry Pi Imager** → select:
- Device: **Raspberry Pi 5**
- OS: **Raspberry Pi OS (64-bit)**  ← must be 64-bit
- Enable SSH and set WiFi credentials in the customisation step

### Step 2 — First Boot: Update firmware & OS

```bash
sudo rpi-update          # needed especially on older Pi 5 boards
sudo apt update && sudo apt upgrade -y
sudo apt install -y python3-pip python3-venv
sudo reboot
```

### Step 3 — Verify the PIO device exists

```bash
ls -l /dev/pio0
```

Expected output:
```
crw-rw---- 1 root gpio ... /dev/pio0
```

If **`/dev/pio0` does not exist** → firmware is too old, re-run `sudo rpi-update` and reboot.

### Step 4 — Add udev rule (run once, allows non-root PIO access)

```bash
echo 'SUBSYSTEM=="*-pio", GROUP="gpio", MODE="0660"' \
  | sudo tee /etc/udev/rules.d/99-com.rules
sudo reboot
```

After reboot verify:
```bash
ls -l /dev/pio0
# Should show: crw-rw---- 1 root gpio ...  ← "gpio" group, not root-only
```

### Step 5 — Add your user to the gpio group

```bash
sudo usermod -aG gpio $USER
# Log out and back in (or reboot) for this to take effect
```

### Step 6 — Create Python virtual environment

```bash
python3 -m venv ~/venvs/led_venv
source ~/venvs/led_venv/bin/activate
```

### Step 7 — Install dependencies

```bash
# Inside the venv:
pip install Adafruit-Blinka-Raspberry-Pi5-Piomatter adafruit-blinka
pip install numpy Pillow requests
```

Or using the requirements file:
```bash
cd ~/raspi_scripts/lcd
pip install -r requirements.txt
```

### Step 8 — Verify the install

```bash
python3 -c "import adafruit_blinka_raspberry_pi5_piomatter as p; print('PioMatter OK:', p.__version__)"
```

---

## Running the Display

### Test mode — verify hardware works (no API needed)

```bash
source ~/venvs/led_venv/bin/activate
cd ~/raspi_scripts/lcd

python3 display_manager.py --test
```

What you should see:
1. **Color bars** — red, orange, yellow, green, cyan, blue, white (3 seconds)
2. **Status screen** — fake vehicle/camera data with `[TST]` label
3. Every ~20 seconds: **amber flashing alert** with `[TEST]` header

### Real mode — live data from Node Service

```bash
python3 display_manager.py
# or with explicit API URL:
python3 display_manager.py --api http://192.168.8.50:3001
```

### Fire one test alert immediately, then run real mode

```bash
python3 display_manager.py --trigger-alert
```

### Adafruit Bonnet (if you have the Adafruit board instead)

```bash
python3 display_manager.py --pinout bonnet
```

---

## Screen Layout & Color Meaning

### Normal screens (rotate every 5 seconds)

```
┌──────────────────────────────────┐
│ ROAD SENTINEL        14:22:07   │  white
│ A: ONLINE    B: ONLINE          │  green=online  red=offline
│ Veh: 2,847      42 km/h avg    │  cyan / yellow
│ 192.168.8.50         2h14m     │  gray
└──────────────────────────────────┘
```

### TEST alert (amber — from `--test` or `--trigger-alert`)
```
┌──────────────────────────────────┐
│ [TEST] SPEEDING  Camera A ██████│  ← amber background (flashing)
│ SIMULATED                       │  amber text
│ 85 km/h on Camera A             │  gray
│ 14:22:07                        │  gray
└──────────────────────────────────┘
```

### REAL alert (from live API incidents)
| Severity | Header color | When |
|---|---|---|
| `critical` | **Red** | Crash / collision |
| `high` | **Orange** | Speeding >limit |
| `medium` | **Yellow** | Wrong-way, stopped |
| `low` | **Cyan** | Congestion |

---

## Autostart (add to launch_both_cameras.sh)

```bash
# Add this line in ~/camera_scripts/launch_both_cameras.sh
# after the camera nohup lines:

source "$HOME/venvs/led_venv/bin/activate"
nohup python3 "$HOME/raspi_scripts/lcd/display_manager.py" \
      --api "http://localhost:3001" \
      >> "$LOG_DIR/led_matrix.log" 2>&1 &
```

---

## Running from Console (Better Performance)

The Adafruit guide recommends running matrix scripts from the **console**, not the desktop session:

```bash
# Option A — SSH into the Pi and run from there (easiest)
ssh pi@192.168.8.50
source ~/venvs/led_venv/bin/activate
python3 ~/raspi_scripts/lcd/display_manager.py

# Option B — Switch to a TTY console on the Pi
# Press Ctrl+Alt+F1 on a connected keyboard

# Option C — Configure Pi to boot to console by default
sudo raspi-config
# → System Options → Boot / Auto Login → Console Autologin
```

---

## Troubleshooting

| Symptom | Fix |
|---|---|
| `/dev/pio0` does not exist | Run `sudo rpi-update` then reboot |
| `Permission denied: /dev/pio0` | Check udev rule (Step 4) and gpio group (Step 5) |
| `ModuleNotFoundError: piomatter` | Run Step 7 install commands inside the venv |
| Color bars don't appear | Check power — panels need 5V from a **separate** PSU |
| Only half the panel works | Ribbon cable between the two panels is loose or reversed |
| Display is garbled | HUB75 ribbon pin 1 (red stripe) may be flipped — flip the connector |
| Display shows garbled/wrong colors | Try `--pinout bonnet` if you have the Adafruit Bonnet; default is `active3` for the ₱149 adapter board |
| Stats show `N/A` | Node Service not reachable — check `--api` URL and that the service is running |
| Pi 5 not booting after rpi-update | Use a known-good power supply (≥5V 3A official PSU) |

---

## Reference Links

- [Adafruit PioMatter — GitHub](https://github.com/adafruit/Adafruit_Blinka_Raspberry_Pi5_Piomatter)
- [RGB Matrix Panels with Raspberry Pi 5 — Adafruit Learn](https://learn.adafruit.com/rgb-matrix-panels-with-raspberry-pi-5)
- [PioMatter on PyPI](https://pypi.org/project/Adafruit-Blinka-Raspberry-Pi5-Piomatter/)
- [hzeller/rpi-rgb-led-matrix Pi 5 issue tracker](https://github.com/hzeller/rpi-rgb-led-matrix/issues/1603)
