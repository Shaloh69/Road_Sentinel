# HUB75 + Raspberry Pi 5 + piomatter — Research Notes
> Last updated: 2026-05-09

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

### [SECONDARY] OE (Output Enable) Pin Floating
- OE is active LOW on HUB75 (LOW = panel ON, HIGH = panel OFF)
- If OE is floating, panel randomly enables/disables → white characters, noise
- GPIO 18 (Pi physical pin 12) → HUB75 pin 15 (OE)
- Verify this jumper is actually connected

### [SECONDARY] 3.3V Logic vs 5V HUB75
- Pi outputs 3.3V; HUB75 expects 5V logic
- Many cheap panels accept 3.3V but it degrades signal margin
- Level shifter (74AHCT245) is the proper fix; without it, keep wires SHORT

### [MINOR] Slowdown Argument — Currently a Dead Arg
- Our `--slowdown` arg is parsed but NEVER passed to piomatter
- The Adafruit examples do not show any slowdown parameter in PioMatter or Geometry
- piomatter does not expose slowdown — it handles timing internally via PIO
- Do not rely on slowdown for noise reduction

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

## 8. Sources

- Adafruit piomatter examples: https://github.com/adafruit/Adafruit_Blinka_Raspberry_Pi5_Piomatter/tree/main/examples
- hzeller wiring guide: https://github.com/hzeller/rpi-rgb-led-matrix/blob/master/wiring.md
- Adafruit learn: https://learn.adafruit.com/rgb-matrix-panels-with-raspberry-pi-5
