# Road Sentinel — Raspberry Pi scripts

Everything that runs on the two Pis. Each Pi owns one camera and one roadside
LED sign.

| | Pi 4 | Pi 5 |
|---|---|---|
| Camera | `CAM-A-001` | `CAM-B-002` |
| Sign controller | **ESP32** (`/dev/ttyUSB*`) | **ESP32** (`/dev/ttyUSB*`) |
| Tailscale | `100.98.53.95` | `100.94.18.9` |
| User | `roadsentinel` | `raspi5` |

## Files

| File | Role |
|---|---|
| `camera/camera_sender.py` | Captures RTSP frames, posts to the AI service, forwards detections/incidents to Node |
| `led_sign_bridge.py` | Polls Node for road state, drives the sign over USB serial. Serves **both** boards |
| `pi_agent.py` | Socket.IO relay for the dashboard's admin terminal |
| `wifi_portal.py`, `setup_wifi_portal.sh` | Headless WiFi re-provisioning from a phone — see `WIFI_PORTAL.md` |
| `99-roadsentinel-sign.rules` | udev rule pinning `/dev/roadsentinel-sign` to the sign board |
| `setup_pi4.sh`, `setup_pi5.sh` | Full provisioning: packages, venv, scripts, systemd units, udev |

## Setup

```bash
# On each Pi, from a clone of the repo
./raspi_scripts/setup_pi4.sh      # Pi 4
./raspi_scripts/setup_pi5.sh      # Pi 5
```

Installs three services: `roadsentinel-camera`, `roadsentinel-display`,
`roadsentinel-agent`.

## The LED sign

**The Pi does not drive the panel.** It sends state over USB serial to a
microcontroller, which does the drawing. Firmware lives in
[`../LEDMatrixDrivers/`](../LEDMatrixDrivers/) — wiring in each port's
`WIRING.md`.

The split: the Pi keeps everything needing a network or a decision — polling
Node, deciding road state, holding alerts, reconnecting. The board knows only
how to draw four screens. **If the sign shows the wrong thing the bug is here;
if it shows it wrongly the bug is in the firmware.** That separation is most of
the value of this design.

### Why not the Pi's GPIO

It was tried, hard, and it does not work on these panels. Roughly 80
hzeller/PioMatter configurations were swept — multiplexing 0-17, row-addr-type
0-5, every RGB sequence, several geometries, two drivers, six pinouts — and
hzeller's own `demo` failed identically, so it was never our code.

The whole Pi-GPIO path (`display_manager.py`, `ledcat`, `led-image-viewer`,
PioMatter, the RGBMatrixEmulator configs) has been **deleted**. Moving to a
microcontroller also removed the `/dev/mem` root requirement, the SPI/audio
GPIO conflicts, and the Pi 5 RP1 incompatibility in one step.

Full account, including two conclusions that were recorded as settled and later
retracted: [`../LEDMatrixDrivers/esp32/DEBUG_LOG.md`](../LEDMatrixDrivers/esp32/DEBUG_LOG.md).

### States

Taken from Node's `/api/public/status`, so the sign and the public web page
cannot disagree.

| State | Sign shows | Colour |
|---|---|---|
| `clear` | `SAFE` | green, static |
| `vehicle_incoming` | `VEHICLE INCOMING` / `SLOW DOWN` | yellow, flashing |
| `incident` | `STOP` | red, flashing |
| *(no data)* | `NO DATA` | blue |

Urgent states are held briefly after the last event — 12s for an incident, 8s
for a vehicle — so a single frame's detection cannot flicker the sign off
again immediately. A sign that blinks between two messages reads as broken.

**The board decides when it has no data.** If nothing arrives for 15 seconds it
switches itself to `NO DATA`. The bridge deliberately never sends that state,
so a dead serial cable produces the same honest result as a dead API rather
than the sign confidently holding a stale `SAFE`.

### Day/night brightness

The panel is sized for daylight legibility, which after dark is glare in the
face of a driver entering the curve. The bridge therefore drives brightness on
a clock:

| Time | Level |
|---|---|
| 05:30 → 06:00 | ramps 5 → 255 |
| 06:00 → 17:45 | **255** (full) |
| 17:45 → 18:15 | ramps 255 → 5 |
| 18:15 → 05:30 | **5** (night floor) |

```bash
--day-brightness 255  --night-brightness 5
--dawn 05:30  --dusk 17:45  --ramp-minutes 30
--brightness N          # pin a fixed level, disabling the schedule
```

Three things worth knowing:

**The schedule is on the Pi because the board has no clock** — no RTC, no WiFi,
no notion of the date. The Pi is the only part of the sign that knows the time.

**Night is 5, never 0.** Zero would blank a safety sign. Measured on the board,
brightness 5 also raises refresh from 334 to 500 fps, since a lower OE duty
cycle leaves more of each frame to scan.

**It waits for the clock.** Pi 4 has no RTC and boots believing whatever date
it shut down on, until NTP corrects it seconds later. A schedule that trusted
that clock could run the sign at 5/255 through the morning. The bridge checks
`systemd-timesyncd` first and holds **day** brightness until time is
trustworthy — failing toward legible, because an unreadable sign is a worse
failure than a bright one.

Fixed clock times rather than a solar almanac: at Busay's latitude (~10.3°N)
sunrise and sunset shift by only about 20 minutes across the year, which is
less than the ramp.

> Both Pis must be on `Asia/Manila`. Pi 5 was found on `Asia/Singapore` —
> the same +0800 offset, so nothing misbehaved, but it was corrected.
> Check with `timedatectl`.

### Testing without a server

```bash
python3 led_sign_bridge.py --test          # cycles every screen, then exits
```

### Diagnosing a dark sign

```bash
ls -l /dev/roadsentinel-sign               # udev rule matched? board attached?
sudo systemctl status roadsentinel-display
sudo journalctl -u roadsentinel-display -n 50
```

The bridge logs the board's `INFO` string on every connect — firmware
configuration, mapping and live refresh rate — so the service log already
records what was running when a problem occurred.

If `/dev/roadsentinel-sign` is missing but the board is plugged in:

- Check the USB cable is a **data** cable, not charge-only — this is by far the
  most common cause.
- Confirm the board enumerates at all: `dmesg | tail -20` after replugging.

Talk to the board directly with any serial terminal at 115200; `HELP` lists
every command.
