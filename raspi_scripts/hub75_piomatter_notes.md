# HUB75 + Raspberry Pi 5 + piomatter — Research Notes
> Last updated: 2026-05-09 (added hzeller findings)

---

## 1. Root Causes of Flickering / Random Colors / White Characters

### [CRITICAL] Missing Common Ground — Most Likely Primary Cause
- **Two separate power supplies (Pi adapter + LED adapter) with NO GND wire between them**
- GPIO signals from Pi are referenced to Pi's GND
- LED panel is referenced to its own power supply GND
- Without a shared GND, the panel cannot interpret HIGH/LOW signals — everything is undefined
- **Fix:** One jumper wire from any Pi GND pin (physical 6/9/14/20/25/30/34/39)
  to the negative terminal of the LED panel power supply

### [CRITICAL] show() Called in Tight Infinite Loop — Confirmed Code Bug
- **Adafruit examples NEVER call show() in a tight infinite loop**
- piomatter uses PIO state machines on the RP1 chip — after show() is called,
  the PIO hardware continuously refreshes the panel INDEPENDENTLY of Python
- Calling show() again while PIO is mid-scan causes the scan to restart → brief blackout
- In a tight loop: restart every ~1ms = constant micro-blackouts = flickering + noise
- **Fix:** Call show() ONCE when content changes, then sleep. Do NOT hammer show().

### [CRITICAL] OE (Output Enable) Pin Floating — hzeller confirmed
- OE- is **active LOW** (LOW = panel ON, HIGH = panel OFF)
- hzeller docs: "If OE/CLK/LAT float → erratic pixels, random dots, flickering, garbage output"
- **OE floating = panel randomly fires at full brightness → white characters/noise**
- GPIO 18 (Pi physical pin 12) → HUB75 pin 15 (OE) — must be connected, cannot float
- Same for CLK (GPIO17) and LAT (GPIO4) — all three must be solid connections

### [CRITICAL] Disable Pi On-Board Audio — hzeller confirmed
- hzeller explicitly warns: **disable on-board sound** — it uses the same PWM timing hardware as GPIO
- On Pi 5 this still applies; audio driver competing for timing resources causes GPIO noise
- Fix on Pi: `sudo nano /boot/firmware/config.txt` → add `dtparam=audio=off` → reboot
- Or check: `sudo systemctl disable --now pulseaudio`

### [SECONDARY] 3.3V Logic vs 5V HUB75 — hzeller confirmed
- Pi outputs 3.3V; HUB75 panels expect 5V logic
- hzeller: "3.3V works on most panels but reduces noise margin; use level shifter if glitching"
- Proper fix: **74AHCT245** (must have "T" — NOT 74HC245 or 74AHC245; "T" has correct VIH for 3.3V→5V)
- hzeller Active-3 adapter uses 4x 74HCT245/74AHCT245 chips
- Without level shifter: keep jumper wires as short as possible (< 20cm)

### [SECONDARY] Power Supply — hzeller specs
- ~3.5A per 32×32 panel at full white
- Wire gauge: **2.5mm² copper minimum per meter per panel** (≈ 13 AWG)
- Capacitors: 6400µF total low-ESR near panel input (e.g. 2× 3300µF in parallel)
- Voltage drop limit: <50mV from PSU to panel connector
- Pi must share GND with panel PSU — hzeller explicitly calls this out

### [MINOR] Slowdown — hzeller's library only, not piomatter
- hzeller's C++ library has `--led-slowdown-gpio` (range 0-10, default 1)
- This slows GPIO write speed for faster Pis / slower panels
- **piomatter does NOT have this parameter** — it handles timing internally via PIO
- Our `--slowdown` arg in color_test.py was dead code — removed

---

## 2. Correct show() Pattern (from Adafruit source)

### What they do (play_gif.py, quote_scroller.py):
```python
while True:
    framebuffer[:] = np.asarray(new_frame)
    matrix.show()          # triggers PIO DMA update
    time.sleep(0.033)      # ~30fps — PIO refreshes display independently between calls
```

### What we were doing (WRONG):
```python
while True:
    fb[:] = arr
    matrix.show()          # hammered at maximum Python speed
    # no sleep — causes PIO restart every ~1ms
```

### Key insight:
`show()` submits a new frame to the PIO. The PIO then scans the panel continuously
until the next `show()` call. Calling `show()` again mid-scan restarts the scan,
creating micro-blackouts. A sleep of ~33ms (30fps) is sufficient and stable.

---

## 3. Correct PioMatter Initialization (from Adafruit source)

```python
geometry = piomatter.Geometry(
    width=128, height=32,
    n_addr_lines=4,              # 4 for 32px tall (1/16 scan), 3 for 16px tall (1/8 scan)
    rotation=piomatter.Orientation.Normal,
)
canvas = Image.new('RGB', (128, 32), (0, 0, 0))
framebuffer = np.asarray(canvas) + 0    # +0 = make mutable writable copy

matrix = piomatter.PioMatter(
    colorspace=piomatter.Colorspace.RGB888Packed,
    pinout=piomatter.Pinout.AdafruitMatrixBonnet,  # or Active3 / Active3BGR
    framebuffer=framebuffer,
    geometry=geometry,
)
```

**No slowdown parameter exists in the constructor.**

---

## 4. Pinout Choice

| Pinout | Use when |
|--------|----------|
| `AdafruitMatrixBonnet` | Using Adafruit bonnet/hat board |
| `Active3` | Direct jumper wires, standard wiring |
| `Active3BGR` | Direct jumper wires, R and B lines physically crossed |

For direct jumper wires following standard HUB75 Active3 mapping:
use `Active3` first. If Red appears Blue, switch to `Active3BGR`.

---

## 5. GPIO → HUB75 Wiring (Active3, Port 1 — single panel)

| HUB75 Signal | HUB75 Pin | BCM GPIO | Pi Physical Pin |
|---|---|---|---|
| R1  | 1  | GPIO 11 | Pin 23 |
| G1  | 2  | GPIO 27 | Pin 13 |
| B1  | 3  | GPIO  7 | Pin 26 |
| GND | 4  | GND     | Pin 6/9/14 |
| R2  | 5  | GPIO  8 | Pin 24 |
| G2  | 6  | GPIO  9 | Pin 21 |
| B2  | 7  | GPIO 10 | Pin 19 |
| GND | 8  | GND     | Pin 6/9/14 |
| A   | 9  | GPIO 22 | Pin 15 |
| B   | 10 | GPIO 23 | Pin 16 |
| C   | 11 | GPIO 24 | Pin 18 |
| D   | 12 | GPIO 25 | Pin 22 |
| CLK | 13 | GPIO 17 | Pin 11 |
| LAT | 14 | GPIO  4 | Pin  7 |
| OE  | 15 | GPIO 18 | Pin 12 |
| GND | 16 | GND     | Pin 6/9/14 |

**IMPORTANT:** HUB75 pins 4, 8, 16 are all GND — connect ALL THREE to Pi GND
AND to the negative terminal of the LED panel's power supply.

---

## 6. Power Wiring Checklist

- [ ] LED panel powered by separate 5V power supply (2A minimum, 4A recommended)
- [ ] Pi GND connected to LED panel power supply GND (common ground)
- [ ] HUB75 GND pins 4, 8, 16 all connected
- [ ] VCC/5V to panel from power supply (NOT from Pi 5V pin)
- [ ] Jumper wires as SHORT as possible

---

## 7. addr-lines Values

| Panel height | Scan type | n_addr_lines |
|---|---|---|
| 16px  | 1/8  | 3 |
| 32px  | 1/16 | 4 |
| 64px  | 1/32 | 5 |

128×32 panel → n_addr_lines=4 ✓

---

## 8. hzeller GPIO Map (Regular / Active-3, Chain 1) — Verified Match

This is the authoritative hzeller table for Chain 1 ("regular" wiring).
**piomatter Active3 uses IDENTICAL GPIO numbers** — confirmed.

| Signal | BCM GPIO | Pi Physical Pin | HUB75 Pin |
|--------|----------|-----------------|-----------|
| R1     | GPIO 11  | Pin 23          | 1         |
| G1     | GPIO 27  | Pin 13          | 2         |
| B1     | GPIO  7  | Pin 26          | 3         |
| GND    | GND      | Pin 6/9/14/20   | 4         |
| R2     | GPIO  8  | Pin 24          | 5         |
| G2     | GPIO  9  | Pin 21          | 6         |
| B2     | GPIO 10  | Pin 19          | 7         |
| GND    | GND      | Pin 6/9/14/20   | 8         |
| A      | GPIO 22  | Pin 15          | 9         |
| B      | GPIO 23  | Pin 16          | 10        |
| C      | GPIO 24  | Pin 18          | 11        |
| D      | GPIO 25  | Pin 22          | 12        |
| CLK    | GPIO 17  | Pin 11          | 13        |
| LAT    | GPIO  4  | Pin  7          | 14        |
| OE-    | GPIO 18  | Pin 12          | 15        |
| GND    | GND      | Pin 6/9/14/20   | 16        |

**ALL three GND pins on HUB75 (4, 8, 16) must be connected to Pi GND
AND to the negative terminal of the LED panel power supply.**

---

## 9. Pre-flight Checklist (do these before every test)

- [ ] Pi GND → LED PSU GND (common ground wire)
- [ ] HUB75 pin 4, 8, 16 all connected to GND
- [ ] OE (GPIO18, pin 12) → HUB75 pin 15 — solid connection, not floating
- [ ] CLK (GPIO17, pin 11) → HUB75 pin 13
- [ ] LAT (GPIO4, pin 7) → HUB75 pin 14
- [ ] Panel powered by separate 5V supply, NOT from Pi 5V pin
- [ ] Audio disabled on Pi: `dtparam=audio=off` in /boot/firmware/config.txt
- [ ] Jumper wires < 20cm (level shifter needed if longer)

---

## 10. Sources

- Adafruit piomatter examples: https://github.com/adafruit/Adafruit_Blinka_Raspberry_Pi5_Piomatter/tree/main/examples
- hzeller wiring guide: https://github.com/hzeller/rpi-rgb-led-matrix/blob/master/wiring.md
- hzeller Active-3 adapter: https://github.com/hzeller/rpi-rgb-led-matrix/tree/master/adapter/active-3
- Adafruit learn: https://learn.adafruit.com/rgb-matrix-panels-with-raspberry-pi-5
