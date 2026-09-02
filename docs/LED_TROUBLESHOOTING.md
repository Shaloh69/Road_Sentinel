# HUB75 LED panel — troubleshooting log

A record of what has been ruled out on the Busay panels, so the same ground is
not re-covered. Panel: `P5户外全彩 KLB 6124` — **P5 outdoor**, two 64×32 chained
side by side to 128×32, on the LEADWAY Pi→HUB75 adapter.

## Current status

**Unresolved.** Panel lights and responds to content, but renders green-only
horizontal bands rather than full-colour full-panel output.

## Confirmed fixed

**SPI was holding five of six data pins.** `dtparam=spi=on` gave the kernel
GPIO 7, 8, 9, 10, 11 — B1, R2, G2, B2, R1 in HUB75 terms. Disabled in
`/boot/firmware/config.txt` (backup: `config.txt.bak-before-hub75`), along with
`dtparam=audio=on` and a blacklist of `snd_bcm2835` for the PWM conflict
hzeller's docs require avoiding.

This one matters beyond itself: **every test run before it was invalid.** Any
result from before that reboot should be discarded rather than reasoned from.

**Pi 5 was using a driver that cannot work.** hzeller's library pokes BCM GPIO
registers the Pi 5 does not have — its logs showed
`RP1 PIO framebuffer transfer failed: -1` then `SIGABRT`, restart-looping every
1.4s. `PioMatterBackend` existed in `display_manager.py` but was unreachable
dead code; it is now the Pi 5 default.

## Ruled out

| Hypothesis | How it was ruled out |
|---|---|
| Undervoltage / brownout | `vcgencmd get_throttled` = `0x0` on **both** Pis, including immediately after driving the panel. Proposed confidently and **wrong**. |
| Wrong GPIO mapping | Schematic transcribed and compared line by line against hzeller's `hardware-mapping.c` — exact match for `regular`. See `HUB75_PINOUT.md`. |
| Parallel vs chained topology | Initially misread the schematic as parallel; confirmed on hardware to be chained. `--led-chain=2 --led-parallel=1` is correct. |
| Multiplexing mode | Swept **0–17**, including the outdoor-specific 12–17. |
| Row address type | Swept 0–5. |
| RGB sequence | Swept all six. |
| Panel geometry | Tried rows=32 and rows=16 across several col/chain combinations. |
| PioMatter pinout | Swept all six (Active3, Active3BGR, both Bonnet and both Hat variants). |
| Software layer | hzeller's own `demo` reproduces it — not our code. |

## Leading hypothesis: one-pin ribbon offset

Not yet checked physically. It is the only single cause found that explains
**both** symptoms simultaneously.

HUB75 pin order:

```
1:R1  2:G1  3:B1  4:GND  5:R2  6:G2  7:B2  8:GND
9:A  10:B  11:C  12:D  13:CLK  14:LAT  15:OE  16:GND
```

Shift the ribbon by one position and:

- R1 → lands on G1 → **red renders as green** (observed)
- B1 → lands on GND → **blue disappears entirely** (observed)
- A → lands on B → **address lines shift → row banding** (observed)

The breadth of the software sweep above is itself evidence for a mechanical
cause: seventeen multiplexing modes, six address types, six colour sequences
and five geometries failing identically is not what a misconfiguration looks
like.

**To check:** ribbon fully seated across all 16 pins at both ends, no exposed
pin row at either edge; panel `IN` connector (arrow pointing away from the Pi),
not `OUT`.

## Known-good reference

Pi 4 is the better test platform: conventional GPIO, so hzeller works natively
with no RP1 complication. Same panel, adapter and wiring as Pi 5. If a config
fails on Pi 4, the problem is not Pi-5-specific.

## Alternative libraries

If a mechanical fault is excluded and hzeller still cannot drive these panels:

| Library | Notes |
|---|---|
| [Adafruit PioMatter](https://github.com/adafruit/Adafruit_Blinka_Raspberry_Pi5_Piomatter) | **Pi 5 only.** Drives HUB75 from RP1's PIO blocks — the supported Pi 5 path. Already installed and wired in. No multiplexing option, so unusual scan patterns need a hand-built `map` array. |
| [flaschen-taschen](https://github.com/hzeller/flaschen-taschen) | Same author. Network display server — send pixels over UDP. Decouples rendering from driving, useful if the driver is fine but our frame-feeding is suspect. |
| [rpi-rgb-led-matrix Python bindings](https://github.com/hzeller/rpi-rgb-led-matrix/tree/master/bindings/python) | Same C++ core, so it will not fix a driver-level problem — but removes `ledcat` piping as a variable. |
| ESP32 + [ESP32-HUB75-MatrixPanel-DMA](https://github.com/mrcodetastic/ESP32-HUB75-MatrixPanel-DMA) | Different approach: a ~$5 ESP32 drives the panel over DMA, Pi sends it content over WiFi. Widely reported to handle awkward outdoor panels that the Pi libraries struggle with, and sidesteps GPIO contention entirely. Worth considering if the panels prove incompatible. |

## Panel spec notes

`P5户外` = P5 outdoor. Outdoor panels are typically **1/8 or 1/4 scan**;
1/16 is usually indoor. The adapter wires **no E line** (GPIO15), so it cannot
drive 1/32-scan (64×64) panels at all.

Sources: [hzeller/rpi-rgb-led-matrix](https://github.com/hzeller/rpi-rgb-led-matrix),
[issue #934 — P5 8S outdoor](https://github.com/hzeller/rpi-rgb-led-matrix/issues/934),
[Adafruit Pi 5 RGB matrix guide](https://learn.adafruit.com/rgb-matrix-panels-with-raspberry-pi-5/overview)

---

# ESP32 route (2026-09-02)

After the Pi path was abandoned, the panel was moved to an ESP32 running
`ESP32-HUB75-MatrixPanel-DMA`, with the Pi driving it over USB serial. See
`esp32_display/README.md` for the architecture and `HUB75_PINOUT.md` for
wiring.

## Proven working on hardware ✅

- Firmware compiles, flashes, boots. PlatformIO Core runs **on the Pi 4**
  (`~/.pio-venv`), so build and flash happen on the device the ESP32 is
  already plugged into — `bash ~/esp32_display/flash.sh`.
- Serial protocol round-trips: `PING`→`PONG`, `OK` acknowledgements,
  `ERR` on bad input.
- **Every colour renders cleanly full-screen.** The red/green/blue/pink/
  yellow/cyan/magenta/orange/white cycle was correct on both panels.
  Wiring, power, colour channels and both panels are therefore proven good.

Two build bugs were caught by compiling on the Pi rather than shipping blind:

- The PlatformIO package owner is `mrfaptastic`, not `mrcodetastic` — the
  GitHub org was renamed but the registry entry kept the original owner.
- The HUB75 library needs Adafruit GFX but does not declare it, so the build
  fails on a missing header until GFX and BusIO are listed explicitly.

## Still unresolved ❌ — coordinate mapping

Solid fills are perfect; **positioned drawing is not**. Text, rectangles and
single rows land in the wrong places.

The distinction matters and explains why this looked contradictory for so
long: `fillScreen` writes the entire framebuffer, so every physical LED lights
regardless of whether the logical→physical mapping is right. Only positioned
drawing exercises the mapping. Solid-colour tests can therefore never
distinguish a correct configuration from a broken one — a mistake that cost
several rounds here.

### Measured symptoms

| Test | Result |
|---|---|
| Full-screen colour | correct, every time |
| One 1px logical row | appears as **two** bands |
| Logical row y=16 | lands on the same pixels as y=0 (aliases) |
| Logical row y=16, 4 bands | rows past the addressable range fold back and compound |
| Any single row | **tilted — drift accumulates per column, left to right** |
| Small block (16x8) | invisible — pixels scatter too sparsely to see |
| `drawLine` across the panel | visible, because it crosses many rows |

### What those measurements mean

**Doubling — initially misread.** One logical row appearing as two physical
bands was treated as a scan-rate fault for several rounds. At the *raw* level
it is the opposite: it is what a correct 1/8-scan configuration must do. A full
raw row fills all 256 shift-register positions, and those positions feed four
64-pixel segments spread over two physical rows. Two bands is the signature of
the geometry being right. Reading it as a fault sent the search in the wrong
direction; recorded here so it is not re-derived.

**Aliasing at 16** — y and y+16 landing together confirms only 16 row-slots are
addressable, not 32. Consistent with 3 address lines.

**Tilt, accumulating per column** — this is the real unresolved symptom. A
*single* 1-pixel row cannot appear tilted unless its data wraps into the next
physical row part-way across. That is a **row-length** mismatch, not a
scan-rate one: fewer pixels are being written per row than the panel's shift
register holds, so no row ever aligns to a row boundary. Tilt persisted even at
raw 256x16 with the remapping layer bypassed, so it is not the remapping layer.

### Why 16 rows, when the panel is 32 tall

The panel shows 32 rows but has only 3 address lines (pin 12 is `NC`), giving
8 addresses. Each address drives two rows at once — one via `R1/G1/B1`, one
via `R2/G2/B2` — so 16 rows are driven per pass. Covering 32 visible rows with
16 drive slots means the shift register is twice as long: **128 positions per
64-wide panel, 256 for two chained**.

```
256 x 16 = 4096 pixels     (how the hardware clocks it)
128 x 32 = 4096 pixels     (what you draw on)
```

Same panel, same pixels. The remapping layer translates between them, so
application code still works in 128x32.

### Shift-register probe (RAWSPAN)

First test that measured the hardware rather than sweeping presets. The 256
shift-register positions were split into quarters, each drawn in its own
colour at raw y=0, all four at once:

| Shift register | Colour | Observed |
|---|---|---|
| 0-63 | red | not visible |
| 64-127 | green | not visible |
| 128-191 | blue | not visible |
| 192-255 | white | **two bands, lower area** |

Only the last-drawn quarter appeared. Two readings fit: **overwriting** (all
four land on the same pixels, last one wins) or **truncation** (positions
0-191 never arrive). They imply opposite fixes, so each quarter was then drawn
*alone*, cleared between:

| Quarter | Alone |
|---|---|
| 0-63 red | not visible |
| 64-127 green | not visible |
| 128-191 blue | not visible |
| 192-255 white | visible — **both panels**, lower area, slightly tilted |

Truncation, not overwriting. Only the final 64 shift-register positions reach
the panel, and their content appears on **both** panels at once.

## Leading conclusion: the D line is missing

A per-panel register of 64 positions per row means each row of a 64-wide panel
is clocked 1:1 — which is **1/16 scan**, and 1/16 scan needs **four** address
lines. Only A, B and C are wired; HUB75 pin 12 (`D`) was read as `NC` off the
panel silkscreen and left unconnected. The adapter schematic disagreed and
showed pin 12 as `D`; the schematic appears to have been right.

One missing wire predicts every symptom collected across both the Pi and ESP32
attempts:

| Symptom | Consequence of no D line |
|---|---|
| `y=16` aliases onto `y=0` | 3 address lines wrap every 8 rows |
| One logical row → two bands | two addresses lit for one row |
| Tilt accumulating per column | 256 bits clocked into a 64-position row |
| Both panels showing the same thing | both hold the same last-clocked data |
| Only the last 64 positions visible | the rest is clocked off the end |
| Solid fills perfect | fills do not depend on addressing |

This also explains why ~80 software configurations failed identically. No
multiplexing mode, row-address type or scan mapping can synthesise a missing
address line, so the search space never contained the answer. The breadth of a
failing sweep was itself the signal to stop sweeping and measure.

**Fix:** wire HUB75 pin 12 to a free ESP32 GPIO (17), set `gpio.d`, and
configure the natural geometry — 64x32 per panel, chain 2, no four-scan
remapping. 🟡 Requires physical rewiring; unverified until someone connects it
and reports what the panel does.

## Geometries tried

| Physical config | Chain | Result |
|---|---|---|
| 64x32 | 1 | renders, doubled + tilted |
| 64x32 | 2 | renders, doubled + tilted |
| 128x16 | 1 | panel dark |
| 128x16 | 2 | panel dark |
| 256x16 (raw, no remap layer) | 2 | renders; two bands (**expected**), still tilted |

All five `setPhysicalPanelScanRate` mappings (`NORMAL_TWO_SCAN`,
`NORMAL_ONE_SIXTEEN`, `FOUR_SCAN_32PX_HIGH`, `FOUR_SCAN_16PX_HIGH`,
`FOUR_SCAN_64PX_HIGH`) were swept at 64x32 with no combination correct.

## Diagnostic commands in the firmware

Added specifically to make this debuggable without reflashing each time —
reflashing is ~20s, a serial command is ~1s:

| Command | Purpose |
|---|---|
| `FILL:r,g,b` | solid colour (proves the panel, not the mapping) |
| `RECT:x,y,w,h,r,g,b` | positioned rectangle — probes the mapping |
| `RECTPX:...` | same rectangle via `drawPixel` only, to isolate the library's optimised fill path |
| `RAWROW:y,r,g,b` | one row in **raw physical** coordinates, bypassing the remap layer entirely |
| `RAWSPAN:x0,x1,y,r,g,b` | a raw span — lights part of the shift register, so which position drives which segment becomes observable |
| `RAWCLS` | clear, raw |
| `SCAN:0-4` | swap scan mapping at runtime |
| `DIAG` | primitives test pattern |

`RAWROW` removes one ambiguity: drawing through the remap layer means a wrong
geometry and a wrong scan mapping produce identical garbage, so the two cannot
be told apart.

`RAWSPAN` removes the next one. A full raw row lights every shift-register
position simultaneously, so there is no way to tell which position drove which
physical segment — the very fact a custom mapping needs. Lighting one quarter
at a time in a distinct colour makes that correspondence directly readable off
the panel. This is the measurement that should have come first: roughly eighty
preset configurations were swept before anything measured what the hardware
actually does, and a sweep cannot converge when the search space does not
contain the answer.

## Note on test design

A recurring error worth recording: several tests could not distinguish the
configurations they were meant to compare. Full-screen white looks identical
under every scan mode. Four colours drawn at once merged where rows aliased,
making positions unreadable. Tests that only reveal the happy path are worse
than no test, because they consume a hardware observation and return nothing.
Single-row, single-colour, held-on-screen probes proved far more informative.
