# ESP32 LED Display

Drives the 128×32 roadside HUB75 sign from an ESP32, with the Raspberry Pi
sending state over USB serial.

Written for someone who has never touched this board. If you only read one
section, read **Panel facts** — every one of those numbers cost debugging time.

---

## Why this exists

The Pi could not drive these panels. Roughly 80 hzeller/PioMatter
configurations were tried — multiplexing 0-17, row-addr-type 0-5, all RGB
sequences, several geometries, both panel init types, two drivers, six
pinouts — and hzeller's own `demo` failed identically, so it was never our
code. Full record in `docs/LED_TROUBLESHOOTING.md` and `DEBUG_LOG.md`.

Moving to the ESP32 removes kernel GPIO contention, the Pi 5 RP1
incompatibility, and 3.3V level marginality in one step, and gives direct
access to the FM6124 init sequence these panels need.

## Why serial and not WiFi

The ESP32 has no network stack at all. The Pi stays the only device needing
WiFi — one fewer thing to provision on site, one fewer credential to rotate,
and a wired link that cannot drop the way site WiFi does.

It also localises failures: if the sign shows the **wrong thing**, the bug is
in the Pi bridge; if it shows it **wrongly**, the bug is in this firmware.

---

## Panel facts

Confirmed against the physical hardware. Do not "correct" these from generic
HUB75 assumptions — several are counterintuitive and each has already been got
wrong once.

| | |
|---|---|
| Board | ESP32, original 30-pin dev board (not S2/S3) |
| Panels | 2 × 64×32 P5 **outdoor**, label `P5户外全彩 KLB 6124` |
| Topology | daisy-chained (panel 1 OUT → panel 2 IN), side by side → **128×32** |
| **Scan rate** | **1/8** — A, B, C only. **There is no D line.** |
| **Driver IC** | **FM6124**, read off the chip marking |
| Power | dedicated 5V 8A PSU into the panels' own terminals, ground common with the ESP32 |

### Two things that are not obvious

**1. FM6124 is not a plain shift register.** It has internal configuration
registers that must be written with a specific latch sequence at init. The
library only emits that sequence when `mxconfig.driver` is set to `FM6124`.
Leave it at the default and the controller powers up in an undefined state —
and then *no* geometry, pin order, or scan mapping can compensate. This single
flag being unset is the likeliest explanation for the entire failure history
above.

**2. A 1/8-scan 64×32 panel is internally shaped like 128×16.** Three address
lines give 8 addresses; each drives two rows at once (upper half via R1/G1/B1,
lower via R2/G2/B2), so one pass paints 16 rows. The other 16 rows are not
missing — they are folded into extra *columns* in the panel's shift register.
So the DMA layer must be told 128×16, not 64×32:

```
physical (told to the DMA driver) : 128 × 16, chain 2   ->  256×16
logical  (what the firmware draws): 128 × 32
translation                       : VirtualMatrixPanel + FOUR_SCAN_32PX_HIGH
```

Configure 64×32 directly and the driver clocks 64 positions into a register
that physically holds 128 — which is what produced the doubled, tilted output
recorded in the debug log.

## Wiring

12 signal wires plus grounds. HUB75 connector pin numbers in brackets.

| Signal | HUB75 pin | ESP32 GPIO | | Signal | HUB75 pin | ESP32 GPIO |
|---|---|---|---|---|---|---|
| R1 | 1 | **25** | | A | 9 | **23** |
| G1 | 2 | **26** | | B | 10 | **19** |
| B1 | 3 | **27** | | C | 11 | **5** |
| GND | 4 | GND | | D | 12 | **not connected** |
| R2 | 5 | **14** | | CLK | 13 | **16** |
| G2 | 6 | **12** | | LAT | 14 | **4** |
| B2 | 7 | **13** | | OE | 15 | **15** |
| GND | 8 | GND | | GND | 16 | GND |

> **Do not wire pin 12.** These are 1/8-scan panels with no D line. An earlier
> revision of this repo concluded the opposite and recorded it as *resolved*;
> that conclusion is retracted — see `DEBUG_LOG.md`. Driving a fourth address
> line the panel cannot decode makes row addressing wrong no matter what else
> is set.

**Panel power is separate.** 5V 8A into the panels' own terminals, supply
ground tied to ESP32 ground. The 16 data wires carry no useful power — two
64×32 panels draw 4-8A and will brown out anything fed through signal lines.

---

## Build and flash (PlatformIO)

The board is plugged into the Pi, so the Pi is the natural place to build —
no cable-swapping at the installation site.

```bash
ssh roadsentinel@100.98.53.95            # Pi 4, over Tailscale
cd ~/esp32_display

~/.pio-venv/bin/pio run                                      # compile
~/.pio-venv/bin/pio run -t upload --upload-port /dev/ttyUSB0 # flash
~/.pio-venv/bin/pio device monitor -b 115200                 # watch
```

`./flash.sh` wraps this and stops the bridge service first.

**Close the monitor before starting the bridge.** Both use the same USB serial
port and two processes cannot hold it at once. The failure mode is a silent
board with no error message — a confusing thing to debug.

On boot the monitor prints `READY` and the panel shows
"ROAD SENTINEL / waiting for Pi".

### Project layout

```
esp32_display/
├── platformio.ini      build config, pinned library versions
├── include/config.h    EVERY hardware fact — pins, geometry, driver, timings
├── src/
│   ├── main.cpp        setup + non-blocking scheduler
│   ├── display.*       matrix init, the 128×32 logical canvas, text helpers
│   ├── protocol.*      serial command parsing
│   ├── mode_status.*   Mode A — production status screens
│   ├── mode_char.*     Mode B — single-character test
│   └── mode_sand.*     Mode C — falling-sand pixel test
├── README.md           this file (finished-state reference)
└── DEBUG_LOG.md        what has already been ruled out, and why
```

Change a hardware value in `include/config.h` and nowhere else.

---

## Display modes

### Mode A — system status (production)

The four states match the rest of the system exactly. `clear`,
`vehicle_incoming` and `incident` come from Node's `/api/public/status`
(`server/node-service/src/routes/public-status.ts:47`), which already computes
them for the public web page — so the sign and the website cannot disagree.

| State | Shows | Colour | Behaviour |
|---|---|---|---|
| `clear` | ROAD CLEAR | green `#3DDC97` | static, large — the common case |
| `vehicle` | VEHICLE / SLOW DOWN | amber `#F2B33D` | inverting flash, 500ms |
| `incident` | INCIDENT / AHEAD - SLOW DOWN | red `#E5484D` | flashes full↔dim red |
| `offline` | `-- NO DATA --` | blue `#5B9DF5` | static |

Colours are the exact Night Watch tokens from `client/web/hero.ts`, so the sign
and dashboard share one palette. Red is reserved system-wide for confirmed
incidents and must not be used for anything else.

Flashing states invert foreground and background rather than blinking to black,
so the text stays legible in both phases.

**Longer messages scroll.** `TEXT:` lines wider than 128px scroll horizontally
at ~25 px/sec; lines that fit are drawn centred and still, because static text
is easier to read and most production messages fit. Each line scrolls
independently — this is how a speed value or vehicle count gets shown without
truncation.

**The board decides when it has no data.** If no command arrives for 15
seconds it switches itself to `-- NO DATA --`. The bridge deliberately never
sends `STATE:offline`, so a dead serial cable produces the same honest result
as a dead API, rather than the sign confidently holding a stale "ROAD CLEAR".
This timeout applies **only** in Mode A — a bring-up test is never yanked off
the panel mid-observation.

### Mode B — single character

One large character (size 4, ~28px tall) centred, with amber ticks at the four
canvas corners. Between a solid fill (which proves only that the LEDs light)
and full text (which can fail for many reasons at once), one big glyph isolates
whether character rendering lands on the right pixels — you know instantly what
an "A" should look like. The corner ticks catch a uniformly-shifted canvas,
which otherwise looks fine.

### Mode C — falling sand

A per-pixel falling-sand simulation over the full 128×32 canvas at ~30fps,
with grains in four Night Watch colours.

**This is the panel bring-up test, not decoration.** A solid fill writes every
pixel and therefore looks identical under every scan mapping, correct or not —
it is the least informative test available here. Sand grains are individually
addressable pixels with predictable motion, so a wrong mapping is not merely
"wrong-looking" but diagnostic:

| What you see | Points at |
|---|---|
| Grains fall down, pile up from the bottom | **working** |
| Correct on panel 1, panel 2 dark | chain length |
| Grains teleport half a canvas sideways | `PHYS_W` folding |
| Each grain doubled, 16 rows apart | address lines |
| Falls in 4-row bands that never connect | scan mapping |
| Red and blue swapped | RGB pin order, not a mapping fault |

It does not auto-trigger. It runs only when asked, so the production sign never
animates when it should be reporting road state.

---

## Command protocol

Newline-terminated ASCII at 115200 baud. Every command is answered `OK` or
`ERR <reason>`, so the Pi can tell "board is wedged" from "board rejected
that". The parser is non-blocking — a 30fps animation is not stalled by a
half-received command.

| Command | Effect |
|---|---|
| `STATE:clear` | ROAD CLEAR, green |
| `STATE:vehicle` | VEHICLE / SLOW DOWN, amber, flashing |
| `STATE:incident` | INCIDENT / AHEAD, red, flashing |
| `STATE:offline` | `-- NO DATA --`, blue |
| `TEXT:line1\|line2` | two-line message; either line scrolls if too wide |
| `MODE:status\|char\|sand` | switch display mode |
| `CHAR:A` | Mode B — show one character |
| `SAND:reset` | restart the sand simulation |
| `SAND:rate,N` | grains added per frame, 1-16 (default 3) |
| `BRIGHT:0-255` | panel brightness (default 90) |
| `PING` | replies `PONG` |
| `INFO` | reports the live config — geometry, driver, scan mapping |
| `HELP` or `?` | lists commands |

Diagnostics:

| Command | Purpose |
|---|---|
| `FILL:r,g,b` | solid colour — proves LEDs, power and RGB order. Proves **nothing** about mapping |
| `CLS` | clear to black |
| `DIAG` | primitives test — fillRect, drawRect, drawLine, drawPixel, text |
| `SCAN:0-4` | swap scan mapping at runtime (0 TWO_SCAN, 1 ONE_SIXTEEN, 2 FOUR_32, 3 FOUR_16, 4 FOUR_64) |
| `RAWSPAN:x0,x1,y,r,g,b` | a span in **raw physical** 256×16 coordinates, bypassing the remap layer |
| `RAWCLS` | clear, raw |

`INFO` exists so "did my reflash actually take?" is a one-second question:

```
canvas=128x32 phys=128x16 chain=2 driver=FM6124 d_line=-1
scan=FOUR_SCAN_32PX_HIGH bright=90 mode=0
```

Test by hand with any serial terminal at 115200:

```
MODE:sand
STATE:vehicle
TEXT:AVG SPEED|42 KPH - SLOW DOWN AHEAD
BRIGHT:40
```

---

## Pi side

```bash
pip install pyserial requests

# Check the panel with no server involved — cycles every screen
python3 raspi_scripts/esp32_display_bridge.py --test

# Normal operation
python3 raspi_scripts/esp32_display_bridge.py --api http://100.120.27.110:3001
```

As a service:

```ini
[Unit]
Description=Road Sentinel ESP32 Display Bridge
After=network-online.target

[Service]
ExecStart=/usr/bin/python3 /home/roadsentinel/roadsentinel/esp32_display_bridge.py --api http://100.120.27.110:3001
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

---

## If the panel is wrong after flashing

Change **one thing at a time** and write down what happened — stacking two
speculative fixes in one flash cycle means learning nothing from either. Log
results in `DEBUG_LOG.md`.

1. `MODE:sand` first. Its failure modes are specific (see the table above) and
   will usually name the fault directly.
2. If output is offset by exactly one pixel column, set `PANEL_CLKPHASE` to
   `true` in `include/config.h`.
3. If content appears in bands that never join up, sweep `SCAN:0` through
   `SCAN:4` over serial — one second per test instead of a 20-second reflash.
   If a value other than `2` is right, change it in `display.cpp` and record why.
4. If red and blue are swapped, that is RGB pin order, not mapping — recheck
   the wiring table, do not touch scan settings.
5. If a single panel is fine but the chain is not, the fault is chain length or
   the ribbon between panels, not the scan mapping.

Do not resume sweeping the configuration space blindly. That was tried ~80
times and it does not converge — `DEBUG_LOG.md` records why.
