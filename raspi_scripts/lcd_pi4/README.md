# HUB75 128×32 RGB LED Matrix — Raspberry Pi 4 Model B Setup Guide

Full-color RGB LED matrix panel for the Road Sentinel installation at Busay.
Shows live system status with color-coded **REAL** and **TEST** alerts.

> **Pi 5 users:** Use [`../lcd/`](../lcd/) instead — Pi 5 requires Adafruit PioMatter,
> not this library.

> **Quick setup?** See [SETUP.md](SETUP.md) for a concise step-by-step guide.

---

## Key Difference vs Pi 5

| | Raspberry Pi 4 Model B | Raspberry Pi 5 |
|---|---|---|
| GPIO chip | Broadcom BCM2711 (direct) | RP1 peripheral chip |
| Library | **hzeller/rpi-rgb-led-matrix** | Adafruit PioMatter |
| Install | Build from C source | `pip install` |
| Needs sudo? | **Yes** (needs `/dev/mem`) | No (uses `/dev/pio0`) |
| Folder | `raspi_scripts/lcd_pi4/` | `raspi_scripts/lcd/` |

---

## Hardware

### What You Need

| Item | Notes |
|---|---|
| Raspberry Pi 4 Model B | Any RAM variant (1GB/2GB/4GB/8GB) |
| **₱149 Chinese HUB75 adapter board** | "Raspberry Pi to Hub75" adapter from Shopee — uses hzeller "regular" GPIO mapping |
| HUB75 RGB LED Matrix panel(s) | See panel configs below |
| 5V 4A+ power supply (per panel) | Use a **dedicated PSU** — do NOT power panels from the Pi USB |
| Short ribbon cable | Usually included with the panel |

### ⚠️ Voltage Note — ₱149 Adapter Board

The ₱149 adapter is a **passive breakout only** — no level shifter.

- Pi 4 GPIO outputs **3.3V**
- HUB75 panels expect **5V logic**

Most modern HUB75 panels accept 3.3V signals and work fine in practice.
If the display shows garbled or flickering output, the panel requires true 5V logic —
add a 74HCT245 level shifter between the adapter and the panel.

### Panel Configurations (both produce 128×32)

| Config | Panels | `display_manager.py` args |
|---|---|---|
| **A — Two 64×32 chained** (most common) | 2× P3/P4/P5 64×32 panels | *(default, no args needed)* |
| **B — Single 128×32 panel** | 1× 128×32 panel | same defaults |

> P3, P4, or P5 pitch 64×32 panels are widely available on Shopee/Lazada (~₱400–700 each).

---

## Wiring

### ₱149 Chinese HUB75 Adapter Board (Default)

```
[Raspberry Pi 4 Model B]
      ↕  (plug adapter board onto 40-pin GPIO header)
[₱149 HUB75 Adapter Board]
      ↕  (HUB75 16-pin ribbon cable)
[Panel 1 INPUT]  →  [Panel 1 OUTPUT] → ribbon → [Panel 2 INPUT]
```

**GPIO Pin Mapping (hzeller "regular" — default):**

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

The adapter board handles all this — just plug it onto the 40-pin header, then
connect the HUB75 ribbon cable from the adapter to the first panel's INPUT port.

**Power:** Connect the panel power leads (red=5V, black=GND) to a **separate 5V 4A PSU**.

### Panel Chaining (two 64×32 → 128×32)

```
[Adapter/Pi] ──ribbon──► [Panel 1 INPUT]   [Panel 1 OUTPUT] ──ribbon──► [Panel 2 INPUT]
```

Look for the **"IN"/"OUT"** label or arrow printed on the panel PCB. The two ports
look identical but are NOT interchangeable.

---

## Step-by-Step Setup on Raspberry Pi 4

### Step 1 — Flash Raspberry Pi OS (64-bit)

Use **Raspberry Pi Imager** → select:
- Device: **Raspberry Pi 4**
- OS: **Raspberry Pi OS (64-bit)**
- Enable SSH and set WiFi credentials in the customisation step

### Step 2 — First Boot: Update OS

```bash
sudo apt update && sudo apt upgrade -y
sudo reboot
```

### Step 3 — Build and install the library

The hzeller library is **not on PyPI** — it must be compiled from C source.
Use the provided install script:

```bash
cd ~/raspi_scripts/lcd_pi4
bash install.sh
```

This will:
1. Install build dependencies (`python3-dev`, `build-essential`)
2. Clone `hzeller/rpi-rgb-led-matrix` from GitHub
3. Build the Python bindings (takes ~2 minutes)
4. Create a venv at `~/venvs/led_venv` and install rgbmatrix + Pillow + requests

### Step 4 — Verify the install

```bash
source ~/venvs/led_venv/bin/activate
sudo $VIRTUAL_ENV/bin/python3 -c "from rgbmatrix import RGBMatrix; print('rgbmatrix OK')"
```

> **Why sudo?** The hzeller library writes directly to `/dev/mem` for fast GPIO access.
> This requires root. Always call `$VIRTUAL_ENV/bin/python3` (not just `python3`) so sudo
> uses the venv's Python, not the system Python.

---

## Running the Display

### Test mode — verify hardware works (no API needed)

```bash
source ~/venvs/led_venv/bin/activate
cd ~/raspi_scripts/lcd_pi4

sudo $VIRTUAL_ENV/bin/python3 display_manager.py --test
```

What you should see:
1. **Color bars** — red, orange, yellow, green, cyan, blue, white (3 seconds)
2. **Static test screen** — fixed layout, no rotating/sliding content:
   ```
   ROAD SENTINEL         [TST]
   A: ON   B: ON   SIMULATED
   Veh:999           45km/h
   192.168.8.x       0h00m
   ```
3. Every ~20 seconds: **amber flashing alert** with `[TEST]` header

### Real mode — live data from Node Service

```bash
sudo $VIRTUAL_ENV/bin/python3 display_manager.py
# or with explicit API URL:
sudo $VIRTUAL_ENV/bin/python3 display_manager.py --api http://192.168.8.50:3001
```

### Fire one test alert immediately

```bash
sudo $VIRTUAL_ENV/bin/python3 display_manager.py --trigger-alert
```

### GPIO slowdown tuning

Pi 4 needs a slowdown value to avoid display glitches. Default is `4`.

```bash
sudo $VIRTUAL_ENV/bin/python3 display_manager.py --test --slowdown 4
# If garbled: try --slowdown 3 or --slowdown 5
```

---

## Screen Layout & Color Meaning

### Normal screens (rotate every 5 seconds — real mode only)

```
┌──────────────────────────────────┐
│ ROAD SENTINEL        14:22:07   │  white
│ A: ONLINE    B: ONLINE          │  green=online  red=offline
│ Veh: 2,847      42 km/h avg    │  cyan / yellow
│ 192.168.8.50         2h14m     │  gray
└──────────────────────────────────┘
```

### Test mode screen (static — no rotation)

```
┌──────────────────────────────────┐
│ ROAD SENTINEL             [TST] │  white / amber badge
│ A: ON   B: ON   SIMULATED       │  green / amber
│ Veh:999               45km/h   │  cyan / yellow
│ 192.168.8.50         0h05m     │  gray
└──────────────────────────────────┘
```

### TEST alert (amber — from `--test` or `--trigger-alert`)

```
┌──────────────────────────────────┐
│ [TEST] SPEEDING  Camera A       │  ← amber background (flashing)
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

## Autostart

Add to `~/camera_scripts/launch_both_cameras.sh` (after the camera nohup lines):

```bash
source "$HOME/venvs/led_venv/bin/activate"
nohup sudo "$VIRTUAL_ENV/bin/python3" "$HOME/raspi_scripts/lcd_pi4/display_manager.py" \
      --api "http://localhost:3001" \
      >> "$LOG_DIR/led_matrix.log" 2>&1 &
```

---

## Troubleshooting

| Symptom | Fix |
|---|---|
| `ModuleNotFoundError: rgbmatrix` | Run `install.sh` — the library must be built from C source |
| `Permission denied: /dev/mem` | Run with `sudo` — required for Pi 4 direct GPIO access |
| `sudo python3` uses wrong Python | Use `sudo $VIRTUAL_ENV/bin/python3` (not `sudo python3`) |
| **"snd_bcm2835 sound module" → program exits** | Already fixed by default (`disable_hardware_pulsing=True`). If still failing, disable onboard audio in `raspi-config → Advanced → Audio → None` and reboot |
| **"one-wire protocol enabled" warning** | Warning only — display still works. To silence: `raspi-config → Interface Options → 1-Wire → No` |
| Color bars don't appear | Check power — panels need 5V from a **separate** PSU |
| Display garbled/flickering | Adjust `--slowdown` (try 3, 4, or 5) |
| Slight flicker on display | Normal with `disable_hardware_pulsing=True`. Add `--hardware-pulse` after disabling onboard audio |
| Only half the panel works | Ribbon cable between the two panels is loose or reversed |
| Display is corrupted stripes | HUB75 ribbon pin 1 (red stripe) may be flipped |
| Stats show `N/A` | Node Service not reachable — check `--api` URL |
| Works but very dim | 3.3V GPIO into 5V panel — add level shifter or accept lower brightness |
| **Both panels show identical content (mirrored)** | See "Fixing Panel Mirroring" below |

---

## Fixing Panel Mirroring

Both 64×32 panels showing the same content is almost always a panel scan-mode mismatch.

### Step 1 — Run the C-level demo first

This tests the library directly, bypassing Python bindings.
If the C demo also mirrors, the fix is a hardware parameter, not a code fix.

```bash
cd ~/rpi-rgb-led-matrix
make examples-api-use          # only needed once

# Standard 64×32 chained panels:
sudo ./examples-api-use/demo \
  --led-rows=32 --led-cols=64 --led-chain=2 \
  --led-gpio-mapping=regular -D1 -t 5

# If still mirrored — try 1:8 multiplexed mode:
sudo ./examples-api-use/demo \
  --led-rows=32 --led-cols=32 --led-chain=4 \
  --led-multiplexing=1 --led-gpio-mapping=regular -D1 -t 5
```

`-D1` = scrolling text demo, `-t 5` = run 5 seconds.

### Step 2 — Match the working C-demo flags in Python

Try these combinations in order until content is no longer mirrored:

| Attempt | Command |
|---|---|
| **A** (default — standard panels) | `sudo python3 display_manager.py --test` |
| **B** (1:8 scan panels) | `sudo python3 display_manager.py --test --cols 32 --chain 4 --multiplexing 1` |
| **C** (interlaced scan) | `sudo python3 display_manager.py --test --scan-mode 1` |
| **D** (128 as single logical panel) | `sudo python3 display_manager.py --test --cols 128 --chain 1` |

Once you find the working combination, add the flags to the autostart command in `launch_both_cameras.sh`.

### Why this happens

Some P3/P4/P5 64×32 panels use **1:8 multiplexing** internally — the PCB routes data
as if it were two separate 32×16 tiles stacked. The hzeller library needs
`--led-cols=32 --led-chain=4 --led-multiplexing=1` to address these panels correctly.
Standard "direct drive" panels work with the default `cols=64 chain=2`.

---

## Reference Links

- [hzeller/rpi-rgb-led-matrix — GitHub](https://github.com/hzeller/rpi-rgb-led-matrix)
- [hzeller wiring.md — GPIO pin table](https://github.com/hzeller/rpi-rgb-led-matrix/blob/master/wiring.md)
- [Python bindings README](https://github.com/hzeller/rpi-rgb-led-matrix/tree/master/bindings/python)
