# HUB75Sign — LED matrix drivers for Road Sentinel

Hand-written HUB75 drivers for 1/8-scan FM6124 outdoor panels, with a portable
core shared across microcontrollers.

Two signs run on this code:

| Board | Drives | Refresh | Serial link |
|---|---|---|---|
| ESP32 dev board | Raspberry Pi 4 sign | 250 fps (measured) | USB via CP2102/CH340 |
| WeAct Black Pill (STM32F411CE) | Raspberry Pi 5 sign | 125 fps (by design) | USB CDC, native |

---

## Why this exists

`ESP32-HUB75-MatrixPanel-DMA` cannot drive these panels, and the reason is
worth stating precisely because it is not a criticism of that library:

**It exposes a DMA framebuffer whose column index is *assumed* to equal the
shift-register clock position.** On these panels it does not, and nothing in
its API lets you correct for it. Every scan-mapping preset adjusts a layer
sitting on top of that assumption, so none of them can converge.

That cost roughly eighty configuration attempts on the Pi, then five scan
mappings, two line decoders, five geometries and a custom mapping on the ESP32
— none of which could have worked. A bit-banged test written in an afternoon
drove the same panel correctly on the first try.

Here the driver emits the clock pulses itself, so position `p` **is** the p-th
pulse. There is nothing left to assume. Full trail, including two wrong
conclusions that got recorded as settled along the way, in
[`esp32/DEBUG_LOG.md`](esp32/DEBUG_LOG.md).

---

## Panel facts — measured, not assumed

Two bit-banged measurements produced every number here, after the project
twice reached a confident wrong conclusion by inferring instead of measuring.

| Property | Value |
|---|---|
| Panels | 2 × 64×32 P5 outdoor, `P5户外全彩 KLB 6124` |
| Driver IC | **FM6124** — needs a register init sequence or nothing displays |
| Scan | **1/8** — A/B/C only, **no D line** |
| Rows per address | 4, spaced 8 apart |
| Address 0 | the **bottom** row — row order inverted |
| Register | 256 positions per channel; each panel owns 128 consecutive |
| Within a panel | first 64 = upper line, second 64 = the line 8 rows below |
| Channels | R1/G1/B1 and R2/G2/B2 feed the two 16-row halves |

Orientation is four flags in `panel_config.h`, each settled by reading the
sign rather than by theory. If a sign is remounted, change flags — never the
mapping.

---

## Layout

```
LEDMatrixDrivers/
├── shared/HUB75Sign/       the portable core — identical on every board
│   ├── panel_config.h      every hardware fact, incl. per-board pin maps
│   ├── framebuffer.*       3-bit pixel store
│   ├── panel_map.*         the measured scan mapping
│   ├── hub75.h             the interface a port must implement
│   ├── display.*           Adafruit_GFX bound to the framebuffer
│   ├── mode_status.*       production sign states
│   ├── mode_char.*         single-character bring-up test
│   ├── mode_sand.*         falling-sand per-pixel test
│   ├── protocol.*          serial command parsing
│   └── sign_app.*          startup + non-blocking scheduler
├── esp32/                  ESP32 port  (Pi 4)
│   ├── src/hub75_esp32.cpp the ONLY board-specific file
│   ├── WIRING.md
│   └── DEBUG_LOG.md
└── stm32/                  STM32 port  (Pi 5)
    ├── src/hub75_stm32.cpp the ONLY board-specific file
    ├── boards/             board manifest with CDC hwids
    └── WIRING.md
```

**Porting to a new board is one file.** Implement four functions from
`hub75.h`: configure pins, run the FM6124 init, refresh, report brightness and
fps. Roughly 150 lines. Everything else describes the panel and the product,
not the microcontroller.

---

## Quick start

```bash
pio run -d esp32 -t upload        # Pi 4 sign
pio run -d stm32 -t upload        # Pi 5 sign  (read stm32/WIRING.md first)
```

The STM32 has real flashing gotchas — a temperature-marginal DFU bootloader
and a one-session write requirement. They are documented in
[`stm32/WIRING.md`](stm32/WIRING.md); read it before the first attempt rather
than after.

## Protocol

Newline-terminated ASCII at 115200. Every command answered `OK` or
`ERR <reason>`, so the Pi can tell "board wedged" from "board disagreed".

| Command | Effect |
|---|---|
| `STATE:clear` | `SAFE`, green, static |
| `STATE:vehicle` | `VEHICLE INCOMING` / `SLOW DOWN`, yellow, flashing |
| `STATE:incident` | `STOP`, red, flashing |
| `STATE:offline` | `NO DATA`, blue |
| `TEXT:line1\|line2` | free text; either line scrolls if wider than 128px |
| `MODE:status\|char\|sand` | switch display mode |
| `CHAR:A` | one large character (bring-up test) |
| `SAND:reset` / `SAND:rate,N` | falling-sand test |
| `BRIGHT:0-255` | panel brightness |
| `FILL:c` / `RECT:x,y,w,h,c` / `CLS` / `DIAG` | diagnostics, `c` = 0-7 |
| `PING` / `INFO` / `HELP` | health and live configuration |

A good `FILL` proves **nothing** about coordinate mapping — a full fill writes
every pixel, so it looks perfect under a correct mapping and a broken one
alike. Use `STATE:clear` or `MODE:sand` to judge mapping.

---

## Design decisions worth knowing

**1-bit colour per channel (8 colours).** Deliberate. A roadside warning sign
wants maximum brightness and contrast, not subtle shades, and 1-bit keeps the
refresh loop short enough to be flicker-free with no modulation timing to get
wrong.

**Precomputed data words.** All 64 `(channel1, channel2)` colour pairs are
resolved to GPIO words at startup, removing six branches per pixel — about
12,000 branches per frame. On STM32 the entire `BSRR` word (set bits low,
reset bits high) is baked in, so the inner loop is one load and one store.

**The board decides when it has no data.** No command for 15 seconds and it
shows `NO DATA` itself. The Pi bridge never sends that state, so a dead cable
produces the same honest sign as a dead API.

**The offline timeout applies only to production mode**, so a bring-up test is
never yanked off the panel mid-observation.

---

## Roadmap

Researched but not implemented, with honest reasoning about whether this
project needs them.

**Binary Code Modulation (BCM)** would give 256 levels per channel instead of
2, by displaying bitplanes for exponentially weighted durations (1:2:4:…:128)
— 8 passes per frame for full colour. The known pitfall is that the
most-significant plane stays lit for half the frame, which reads as flicker at
low brightness; [SmartMatrix](https://deepwiki.com/pixelmatix/SmartMatrix/3.1-hub75-led-panels)
splits high-weight planes into segments spread across the sequence to fix it.
*Not needed for a warning sign, but it is the single biggest feature gap if
this ever becomes a general-purpose library.*

**STM32 DMA → GPIO BSRR** is the significant one for the Pi 5 sign. A timer can
trigger DMA2 to stream precomputed 32-bit words straight into `GPIOx->BSRR`,
driving the panel with [almost no CPU involvement](https://community.st.com/t5/stm32-mcus-products/using-timer-to-trigger-dma-event-to-write-into-gpio-gt-bsrr/td-p/796874).
It must be DMA2 — GPIO sits on AHB1 and DMA1 cannot reach it. Reference
implementations reach [~549Hz with the CPU free](https://github.com/bikefrivolously/led_matrix).
Our precomputed-`BSRR` table is already exactly the data such a stream needs,
so this is a natural next step rather than a rewrite. Held back only because it
cannot be verified without the board in hand.

**Double buffering** would remove tearing when a frame is redrawn mid-refresh.
Not currently visible, because the sign's content changes rarely and 1-bit
pixel writes are atomic.

Prior art worth reading if extending this:
[mrcodetastic/ESP32-HUB75-MatrixPanel-DMA](https://github.com/mrcodetastic/ESP32-HUB75-MatrixPanel-DMA)
(the DMA approach and its scan-mapping presets),
[board707/DMD_STM32](https://registry.platformio.org/libraries/board707/DMD_STM32)
(STM32 HUB75/HUB12/HUB08),
[kostaman/HUB75](https://github.com/kostaman/HUB75) (STM32 HAL).

---

## Reusing this outside Road Sentinel

The core is panel-specific, not project-specific, with one exception:
`mode_status.cpp` hard-codes the Road Sentinel sign states. Everything else —
driver, mapping, framebuffer, text, protocol — works for any 1/8-scan FM6124
panel chain.

If you have panels that behave differently, do not guess at the mapping. Use
the two measurements in `DEBUG_LOG.md`: walk A/B/C through the addresses and
count lit rows, then colour each quarter of the register and photograph where
it lands. Together they take under an hour and settle every unknown, which is
considerably less than eighty configuration attempts.
