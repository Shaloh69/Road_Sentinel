# HUB75 Wiring Reference — Road Sentinel LED Matrices

Transcribed from the project's wiring schematic. Header pin numbers are
identical across Pi 3 / 4 / 5, so this applies to both Pis.

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
