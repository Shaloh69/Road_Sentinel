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
