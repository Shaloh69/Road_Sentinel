# ESP32 LED Display

Drives the HUB75 sign from an ESP32 instead of the Pi's GPIO, with the Pi
sending state over USB serial.

## Why this exists

The Pi could not drive these panels. Roughly 80 hzeller/PioMatter
configurations were tried — multiplexing 0-17, row-addr-type 0-5, all RGB
sequences, several geometries, both panel init types, two drivers, six
pinouts — and hzeller's own `demo` failed identically, so it was never our
code. Full record in `docs/LED_TROUBLESHOOTING.md`.

The ESP32 library expresses the thing we could not express on the Pi:
`gpio.d = -1` says "this panel has no D line" directly, rather than encoding
it in a geometry the Pi library may not model for this panel. It also removes
kernel GPIO contention, the Pi 5 RP1 incompatibility, and 3.3V level
marginality in one step.

## Why serial and not WiFi

The ESP32 has no network stack at all. The Pi remains the only device needing
WiFi, which means one fewer thing to provision on site, one fewer credential
to rotate, and a wired link that cannot drop the way site WiFi does.

It also makes failures easy to localise: if the sign shows the wrong thing,
the bug is in the Pi bridge; if it shows it wrongly, the bug is in the
firmware.

## Hardware

**Per sign:** one ESP32 dev board, one USB cable to the Pi, 16 jumper wires,
and a 5V 8A supply for the panels.

Wiring is in `raspi_scripts/HUB75_PINOUT.md`. Summary:

| HUB75 | ESP32 | | HUB75 | ESP32 |
|---|---|---|---|---|
| R1 (1) | 25 | | A (9) | 23 |
| G1 (2) | 26 | | B (10) | 19 |
| B1 (3) | 27 | | C (11) | 5 |
| GND (4) | GND | | **D (12)** | **not connected** |
| R2 (5) | 14 | | CLK (13) | 16 |
| G2 (6) | 12 | | LAT (14) | 4 |
| B2 (7) | 13 | | OE (15) | 15 |
| GND (8) | GND | | GND (16) | GND |

**Panel power is separate.** 5V 8A into the panels' own terminals, supply
ground tied to ESP32 ground. The 16 data wires carry no useful power — two
64x32 panels draw 4-8A and will brown out anything trying to feed them
through signal lines.

## Flashing (PlatformIO)

```bash
cd esp32_display

pio run                          # compile
pio run -t upload                # flash
pio device monitor               # serial monitor at 115200
pio run -t upload -t monitor     # flash then watch
```

The library is pinned in `platformio.ini` and fetched automatically — nothing
to install by hand.

On first run the monitor should print `READY`, and the panel should show
"ROAD SENTINEL / waiting for Pi".

**Close the monitor before starting the bridge.** Both use the same USB serial
port, and two processes cannot hold it at once. The failure mode is a silent
board with no error message, which is a confusing thing to debug.

For an ESP32-S3 board instead of the classic ESP32, uncomment the second
`[env:...]` block in `platformio.ini` and build with
`pio run -e esp32-s3-devkitc-1`.

### Project layout

```
esp32_display/
├── platformio.ini     build config, library pin
├── src/main.cpp       firmware
└── README.md
```

## Pi side

```bash
pip install pyserial requests

# Check the panel with no server involved - cycles every screen
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
ExecStart=/usr/bin/python3 /home/USER/roadsentinel/esp32_display_bridge.py --api http://100.120.27.110:3001
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

## Protocol

Newline-terminated ASCII at 115200. Every command is acknowledged, so the Pi
can distinguish "board is wedged" from "board rejected that".

| Command | Effect |
|---|---|
| `STATE:clear` | ROAD CLEAR, green |
| `STATE:vehicle` | VEHICLE / SLOW DOWN, amber, flashing |
| `STATE:incident` | INCIDENT / AHEAD, red, flashing |
| `STATE:offline` | `-- NO DATA --`, dim blue |
| `TEXT:line1\|line2` | arbitrary two-line message |
| `BRIGHT:0-255` | panel brightness |
| `PING` | replies `PONG` |

Test by hand with any serial terminal at 115200:

```
STATE:vehicle
TEXT:HELLO|BUSAY
BRIGHT:40
```

## Two deliberate design decisions

**The board decides when it has no data.** If no command arrives for 15
seconds it switches itself to `-- NO DATA --`. The bridge never sends
`STATE:offline` on a failed poll — letting the board time out means a dead
serial cable produces the same honest result as a dead API, rather than the
sign confidently holding a stale "ROAD CLEAR" while the system is down.

**State comes from `/api/public/status`, not recomputed here.** That endpoint
already derives the state for the public web page, so the physical sign and
the website cannot disagree — which they would if this applied the same rule
independently.

## If the panel is wrong after flashing

Try in this order, one at a time:

1. `mxconfig.clkphase = false` → `true` (pixels shifted by one)
2. Uncomment `mxconfig.driver = HUB75_I2S_CFG::FM6124`
3. If only part of the panel lights, set `gpio.d = 17` — that would mean the
   panel does have a D line and the `NC` silkscreen reading was a miscount
