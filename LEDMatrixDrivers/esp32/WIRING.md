# ESP32 → HUB75 wiring (Road Sentinel, Pi 4 sign)

Written for someone who has never wired one of these. Every number here is
confirmed against the physical board — several are counterintuitive and each
has already cost real debugging time.

## What you need

| Item | Notes |
|---|---|
| ESP32 dev board | Original 30-pin, **not** S2/S3. Board id `esp32dev` |
| 2 × P5 outdoor panels | 64×32 each, `P5户外全彩 KLB 6124`, **FM6124** driver IC |
| 16 jumper wires | 12 signal + grounds |
| 5V 8A PSU | **Separate supply for the panels.** Not the ESP32's 5V pin |
| USB cable | ESP32 → Raspberry Pi 4, carries both power and the serial link |

## The 12 signal wires

HUB75 connector pin numbers in brackets. The connector is a 16-pin IDC — pin 1
is usually marked with an arrow or a square pad.

| HUB75 | Signal | ESP32 GPIO | | HUB75 | Signal | ESP32 GPIO |
|---|---|---|---|---|---|---|
| 1 | R1 | **25** | | 9 | A | **23** |
| 2 | G1 | **26** | | 10 | B | **19** |
| 3 | B1 | **27** | | 11 | C | **5** |
| 4 | GND | GND | | 12 | *(D)* | **not connected** |
| 5 | R2 | **14** | | 13 | CLK | **16** |
| 6 | G2 | **12** | | 14 | LAT | **4** |
| 7 | B2 | **13** | | 15 | OE | **15** |
| 8 | GND | GND | | 16 | GND | GND |

Tie at least one GND between the ESP32 and the panel, and tie the panel PSU's
ground to the ESP32 ground as well. A signal ground that does not return to the
supply ground is a classic source of flicker and garbage.

## Three things that will bite you

**Do not wire pin 12.** These panels are 1/8 scan and have no D line. An
earlier revision of this project concluded the opposite from a shift-register
probe and wrote it up as *resolved*; that was wrong and is retracted. Driving a
fourth address line the panel cannot decode breaks row addressing no matter
what else is set. Verified by direct measurement: walking A/B/C through
addresses 0-7 lights 8 distinct rows, which is all a 32-row 1/8-scan panel
needs.

**Panel power is separate, and it is not optional.** 5V 8A into the panels' own
screw terminals. Two 64×32 P5 panels draw 4-8A at full white; the 16 signal
wires carry no useful current and anything fed through them browns out. Common
the grounds, never the 5V.

**GPIO 12 is a strapping pin.** On the ESP32, GPIO 12 (MTDI) is sampled at
reset to select flash voltage. It is used here as G2 and works, but if the
board ever refuses to boot with the panel connected, disconnect G2 and retry —
that identifies it immediately.

## Why these particular pins

All twelve are below GPIO 32, which puts them in the ESP32's single 32-bit
`W1TS`/`W1TC` GPIO registers. That lets the driver set all six data lines with
one store instead of six `digitalWrite` calls, which is what makes a
bit-banged refresh fast enough to be flicker-free — measured at 250fps.

Change a pin to one numbered 32 or above and the driver's masks silently stop
working, because those pins live in a different register pair.

## Checking it

Flash, then over serial at 115200:

```
PING          -> PONG          board alive
INFO          -> canvas=128x32 addresses=8 ... fps=250
FILL:4        whole sign red   proves LEDs, power and RGB order
STATE:clear   SAFE in green    proves addressing and mapping
MODE:sand     falling grains   proves per-pixel addressing
```

A good `FILL` proves **nothing** about coordinate mapping — a full fill writes
every pixel, so it looks perfect under a correct mapping and a broken one
alike. Several hardware sessions were lost to exactly that misreading. Use
`STATE:clear` or `MODE:sand` to judge mapping.
