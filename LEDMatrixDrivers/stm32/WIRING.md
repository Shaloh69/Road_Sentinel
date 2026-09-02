# STM32 Black Pill → HUB75 wiring (Road Sentinel, Pi 5 sign)

Written for someone who has never wired one of these. The panel is identical to
the ESP32 sign's — only the microcontroller and therefore the pin numbers
differ.

## What you need

| Item | Notes |
|---|---|
| WeAct Black Pill V3.0 | **STM32F411CEU6.** V3.0 boards are F411; chip id 0x431 |
| 2 × P5 outdoor panels | 64×32 each, `P5户外全彩 KLB 6124`, **FM6124** driver IC |
| 16 jumper wires | 12 signal + grounds |
| 5V 8A PSU | **Separate supply for the panels.** Not the board's 5V pin |
| USB-C cable | Black Pill → Raspberry Pi 5, power + serial + DFU flashing |

## The 12 signal wires

HUB75 connector pin numbers in brackets.

| HUB75 | Signal | Black Pill | | HUB75 | Signal | Black Pill |
|---|---|---|---|---|---|---|
| 1 | R1 | **PB0** | | 9 | A | **PB7** |
| 2 | G1 | **PB1** | | 10 | B | **PB8** |
| 3 | B1 | **PB3** | | 11 | C | **PB9** |
| 4 | GND | GND | | 12 | *(D)* | **not connected** |
| 5 | R2 | **PB4** | | 13 | CLK | **PA0** |
| 6 | G2 | **PB5** | | 14 | LAT | **PA1** |
| 7 | B2 | **PB6** | | 15 | OE | **PA2** |
| 8 | GND | GND | | 16 | GND | GND |

Tie at least one GND between the board and the panel, and tie the panel PSU's
ground to the board ground as well.

## Three things that will bite you

**PB2 is skipped on purpose.** PB2 is the BOOT1 strapping pin. Driving it
interferes with entering the DFU bootloader, which is the only way this board
gets flashed without an ST-Link. That is why the data lines run
PB0, PB1, PB3-PB6 rather than PB0-PB5.

**Do not wire pin 12.** These panels are 1/8 scan and have no D line — same as
the ESP32 sign. Driving a fourth address line the panel cannot decode breaks
row addressing regardless of anything else.

**Panel power is separate.** 5V 8A into the panels' own screw terminals. Two
64×32 P5 panels draw 4-8A at full white. Common the grounds, never the 5V.

## Why these particular pins

All six data lines **and** all three address lines are on **GPIOB**, so the
driver moves them with a single `BSRR` write. STM32's `BSRR` is neater than the
ESP32's register pair: the low 16 bits set and the high 16 bits reset, so
setting some lines and clearing the rest is one atomic store.

Control lines (CLK, LAT, OE) sit on GPIOA. Splitting them off costs nothing —
the clock is pulsed separately from the data anyway — and it keeps GPIOB's mask
clean.

Move a data or address line off GPIOB and the driver's masks stop working.

## Flashing — read this before your first attempt

This board's DFU bootloader is temperature-marginal (documented WeAct/QMK
issue, 25MHz crystal). These are not superstitions; they were learned the hard
way and each one cost hours:

1. **Warm the chip first** — hot room, fingertip, hairdryer. Cold, transfers
   over ~2KB drop mid-way and *reads return phantom data*, which mimics a dead
   or counterfeit board.
2. **Enter DFU:** hold BOOT0 → tap NRST → release BOOT0, with USB plugged in.
3. **Flash in ONE session:** `pio run -t upload`. Never split the write across
   multiple CubeProgrammer connections — chunked writes are ACKed and then
   silently lose most blocks.
4. If an operation fails, the bootloader **wedges** and enumerates with empty
   descriptors (`DevID = 0x0000`) until the next BOOT0/NRST re-entry. Discard
   any conclusion drawn from a session that already had a failure in it.

`STM32CubeProgrammer` must be installed; the path is in `platformio.ini`.
`upload_protocol = cubeprogrammer` does not exist in PlatformIO — it warns and
then never uploads — which is why the config goes through `custom`.

## Checking it

The Black Pill enumerates as a USB CDC serial port (VID/PID `0483:5740`, ST's
own Virtual COM Port ID, which Windows binds without Zadig). At 115200:

```
PING          -> PONG          board alive
INFO          -> canvas=128x32 addresses=8 ... fps=125
FILL:4        whole sign red   proves LEDs, power and RGB order
STATE:clear   SAFE in green    proves addressing and mapping
MODE:sand     falling grains   proves per-pixel addressing
```

Expect **~125fps** here versus 250 on the ESP32. The F411 is single-core, so
the refresh runs from a TIM3 interrupt at 1kHz (one address per interrupt)
rather than a dedicated core. 125fps is far above the flicker threshold; the
lower number is a design choice to leave CPU for USB CDC, not a defect.

If no COM port appears after flashing, check `boards/blackpill_f411ce.json` is
present — it is a project-local board manifest whose only addition is the
`hwids` entry that CDC needs.
