# HUB75 Wiring Reference — Road Sentinel LED Matrices

## ⚠️ READ FIRST: the panels are 1/8 scan, not 1/16

Confirmed 2026-09-01 from the panel's own silkscreen: **pin 12 is `NC`**,
where a 1/16-scan panel would have `D`. Only `A`, `B`, `C` exist, so there are
8 row addresses, not 16.

The adapter board is built for 1/16-scan panels and *does* wire D to pin 12 —
see the schematic below. The panel simply ignores it. The adapter is not at
fault; the software configuration was.

**This matters because in hzeller `--led-rows` selects how many address lines
are driven:**

| `--led-rows` | Address lines used | Scan rate |
|---|---|---|
| 16 | A, B, C | **1/8 — these panels** |
| 32 | A, B, C, **D** | 1/16 |
| 64 | A, B, C, D, E | 1/32 |

Using `--led-rows=32` makes the library drive a D line the panel does not
have, so row addressing is wrong no matter what else is set. That is why
sweeping multiplexing 0-17, row-addr-type 0-5, six RGB sequences and six
PioMatter pinouts all failed identically — none of them can undo a phantom
address line.

For a 64x32 panel that is 1/8 scan, the panel is internally arranged more like
128x16, so the starting point is:

```
--led-rows=16 --led-cols=64 --led-chain=2 --led-multiplexing=<sweep 0..17>
```

The multiplexing value then remaps that internal layout onto the visible
128x32. On the ESP32 library the equivalent is `mxconfig.gpio.d = -1`.

Panel: `P5户外全彩 KLB 6124` — P5 **outdoor**. Outdoor panels are commonly 1/8
or 1/4 scan; 1/16 is typically indoor. That was a clue available from the
label all along.

---


Transcribed from the project's wiring schematic. Header pin numbers are
identical across Pi 3 / 4 / 5, so this applies to both Pis.

## Panel facts (counted on the hardware, not inferred)

- **32 LEDs vertically, 64 LEDs horizontally** per panel — physically counted
  2026-09-01, so 64x32 is confirmed, not assumed.
- **Two panels, daisy-chained** (panel 1 OUT → panel 2 IN), mounted side by
  side. Total display: **128x32**.
- Label: `P5户外全彩 KLB 6124` — P5 **outdoor**, 5 mm pitch.
- 32 rows means **no E line is needed** (E is only for 1/32-scan 64-row
  panels). The adapter does not wire E anyway.

So the geometry the software should describe is 128x32, reached as
`--led-cols=64 --led-chain=2`. What remains genuinely uncertain is the **scan
rate** — how those 32 rows are addressed internally — not the dimensions.

---

## The three sources, side by side

Three separate pinouts matter here and they do **not** all agree. Recording all
three so the discrepancy is not rediscovered.

| HUB75 pin | Panel silkscreen | Adapter schematic | ESP32 library default |
|---|---|---|---|
| 1 | R1 | R1 → GPIO11 | GPIO25 |
| 2 | G1 | G1 → GPIO27 | GPIO26 |
| 3 | B1 | B1 → GPIO7 | GPIO27 |
| 4 | GND | GND | GND |
| 5 | R2 | R2 → GPIO8 | GPIO14 |
| 6 | G2 | G2 → GPIO9 | GPIO12 |
| 7 | B2 | B2 → GPIO10 | GPIO13 |
| 8 | GND | GND | GND |
| 9 | A | A → GPIO22 | GPIO23 |
| 10 | B | B → GPIO23 | GPIO19 |
| 11 | C | C → GPIO24 | GPIO5 |
| **12** | **NC** | **D → GPIO25** | GPIO17 (set to `-1`) |
| 13 | CLK | CLK → GPIO17 | GPIO16 |
| 14 | LAT | STROBE → GPIO4 | GPIO4 |
| 15 | OE | OE → GPIO18 | GPIO15 |
| 16 | GND | GND | GND |

**The disagreement is pin 12.** The adapter drives `D`; the panel says `NC`.
The adapter is designed for 1/16-scan panels and is not at fault — it simply
provides a signal this panel does not use.

### What that does to the display

With `--led-rows=32`, hzeller drives four address lines and expects 16
distinct row addresses. The panel decodes only A/B/C, so it can distinguish 8.
Two different row-pairs therefore collapse onto the same address and show
overlapping content — which is exactly the banding observed on the hardware.

### Careful, though — this is not fully settled

`--led-rows=16` (three address lines, matching the panel) produced a
**completely dark** panel rather than a correct one. So either the column
count has to change with it — a 1/8-scan 64x32 panel is internally arranged
closer to 128x16, meaning `--led-cols` must double as `--led-rows` halves —
or the NC reading is a miscount of the connector positions.

Configurations still worth trying, all with three address lines:

```
--led-rows=16 --led-cols=128 --led-chain=2   # canvas 256x16
--led-rows=16 --led-cols=64  --led-chain=4   # canvas 256x16
--led-rows=16 --led-cols=128 --led-chain=1   # canvas 128x16
```

On the ESP32 library the equivalent is simply `mxconfig.gpio.d = -1`, which is
one of the reasons that route is attractive: it expresses "this panel has no D
line" directly, rather than through a geometry encoding.

---

## Pin map (as wired)

### Shared control lines — both panels

| Signal | GPIO | Header pin |
|---|---|---|
| STROBE / LATCH | GPIO4 | 7 |
| CLOCK | GPIO17 | 11 |
| OE (output enable) | GPIO18 | 12 |
| A (addr) | GPIO22 | 15 |
| B (addr) | GPIO23 | 16 |
| C (addr) | GPIO24 | 18 |
| D (addr) | GPIO25 | 22 |

### LED1 — first panel

| Signal | GPIO | Header pin |
|---|---|---|
| R1 | GPIO11 (SCLK) | 23 |
| G1 | GPIO27 | 13 |
| B1 | GPIO7 | 26 |
| R2 | GPIO8 | 24 |
| G2 | GPIO9 (MISO) | 21 |
| B2 | GPIO10 (MOSI) | 19 |

### LED2 — second panel

| Signal | GPIO | Header pin |
|---|---|---|
| R1 | GPIO12 | 32 |
| G1 | GPIO5 | 29 |
| B1 | GPIO6 | 31 |
| R2 | GPIO19 | 35 |
| G2 | GPIO13 | 33 |
| B2 | GPIO20 | 38 |

Power: +5V from header pins 2/4, grounds on 6/14/20/25/30/34/39.

## What this mapping means

Both blocks match **hzeller's `regular` GPIO mapping exactly** — LED1 on
parallel chain 1 and LED2 on parallel chain 2, as defined in
`lib/hardware-mapping.c`:

```c
/* Parallel chain 1 */
.p0_r1 = GPIO_BIT(11), .p0_g1 = GPIO_BIT(27), .p0_b1 = GPIO_BIT(7),
.p0_r2 = GPIO_BIT(8),  .p0_g2 = GPIO_BIT(9),  .p0_b2 = GPIO_BIT(10),
/* Parallel chain 2 */
.p1_r1 = GPIO_BIT(12), .p1_g1 = GPIO_BIT(5),  .p1_b1 = GPIO_BIT(6),
.p1_r2 = GPIO_BIT(19), .p1_g2 = GPIO_BIT(13), .p1_b2 = GPIO_BIT(20),
```

So `--led-gpio-mapping=regular` is correct.

## Actual panel topology: CHAINED (confirmed on the hardware)

The schematic above shows the **adapter board's** available outputs — it
documents which GPIO each of its HUB75 connectors is wired to, not how the
panels are physically cabled.

In the real installation the two panels are **daisy-chained**: panel 1's
output connector feeds panel 2's input, both driven from the parallel-1 data
lines. They are mounted **side by side** to read as a single 128×32 banner.
Both Pi 4 and Pi 5 use this same arrangement.

So the existing configuration is correct and should not be changed:

```
--led-chain=2 --led-parallel=1     # correct for this installation
--led-rows=32 --led-cols=64        # two 64x32 panels -> 128x32
```

(An earlier revision of this file claimed the panels were wired in parallel,
inferred from the schematic alone. That was wrong — corrected here after
confirming the physical wiring.)

## Pi 5 caveat — separate issue

On the Pi 5 this wiring question is necessary but not sufficient. The Pi 5
replaced directly-addressable BCM GPIO registers with the RP1 chip, so
hzeller's library cannot drive HUB75 correctly there at all, regardless of
wiring. Use Adafruit's **PioMatter** (`adafruit-blinka-raspberry-pi5-piomatter`),
which drives the panel from RP1's PIO state machines. `display_manager.py`
now defaults the Pi 5 to that backend.

The Pi 4 has conventional GPIO and can keep using the hzeller path
(`ledcat`), so on the Pi 4 the parallel-vs-chain fix above should be
sufficient on its own.


## Resolved: pin 12 is D, not NC (2026-09-03)

The three pinouts recorded above disagreed about pin 12 — the panel silkscreen
read `NC`, the adapter schematic read `D`. **The schematic was right.**

Measured with `RAWSPAN` on the ESP32: of 256 shift-register positions clocked
out, only the last 64 reached the panel, and that content appeared on both
panels simultaneously. A 64-wide panel holding 64 positions per row is clocked
1:1 — that is 1/16 scan, which requires four address lines.

Wire HUB75 pin 12 to ESP32 **GPIO 17**, then set `#define HAVE_D_LINE 1` in
`esp32_display/src/main.cpp` and reflash.

Reading `NC` off the silkscreen and trusting it over the schematic is what
sent roughly eighty software configurations chasing a wire. Where two sources
disagree about hardware, the cheap move is to measure rather than to pick the
more convenient one.

🟡 Unverified — requires the physical wire and someone to report what the panel
then shows.
