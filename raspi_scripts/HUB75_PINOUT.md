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

## ⚠️ The panels are wired PARALLEL, not CHAINED

This is the important part, and it contradicts the configuration the code
has been using.

Each panel gets its **own** set of six RGB data lines on a different parallel
port. They are *not* daisy-chained — a chained setup would run one panel's
output connector into the next panel's input and use only one set of data
lines.

The software has been asking for the opposite:

```
--led-chain=2 --led-parallel=1     # wrong: says one chain of two panels
--led-chain=1 --led-parallel=2     # right: two parallel chains of one panel
```

Told to drive a 2-deep chain, the library clocks all 128 columns of pixel
data out of the **parallel-1 pins only**, while the parallel-2 pins (LED2)
receive nothing coherent. Neither panel then shows what it should — and this
happens even on a solid-color frame, which is exactly the observed symptom.
No amount of `--led-slowdown-gpio` or `--led-multiplexing` tuning fixes it,
because the timing was never the problem.

## Geometry consequence

With `parallel=2, chain=1`, the library's framebuffer is **64 wide × 64 tall**
(two 64×32 panels stacked), not 128×32 side by side. If the panels are
physically mounted side by side to read as one 128×32 banner, the frame has
to be split: left half to parallel chain 1, right half to chain 2.

Options:
- Set `--led-parallel=2 --led-chain=1` and render to a 64×64 buffer, mapping
  the two halves to the correct panels.
- Or rewire the panels as an actual chain (LED1 output → LED2 input, both on
  the parallel-1 data lines) and keep `--led-chain=2 --led-parallel=1`.

Rewiring is the smaller change if a true 128×32 layout is wanted.

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
