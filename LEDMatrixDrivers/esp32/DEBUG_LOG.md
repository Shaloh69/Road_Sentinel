# ESP32 HUB75 — debug log

The "what did we already rule out" trail. The README is the finished-state
reference; this file is the history, so neither of us re-tries a fix we already
know does not work.

Convention, matching the root `README.md`:
✅ verified on real hardware · 🟡 code-complete, not hardware-verified · ⛔ blocked

---

## 2026-09-03 — RETRACTED: "pin 12 is D, the panels are 1/16 scan"

**This conclusion was wrong and has been reversed by the user, who has the
board in hand.** It is recorded here rather than deleted, because it was
committed (`36535e4`), written into `raspi_scripts/HUB75_PINOUT.md` under a
heading that said *Resolved*, and repeated in `docs/LED_TROUBLESHOOTING.md`.
Anyone reading those files before this correction will have absorbed it.

What was claimed: a `RAWSPAN` probe clocked 256 shift-register positions and
only the last 64 reached the panel, appearing on both panels at once. That was
read as 1:1 clocking → 1/16 scan → a fourth address line must exist → wire
HUB75 pin 12 to GPIO 17.

Why it was wrong: **the probe was run with the driver left at the library
default `SHIFTREG`, on a panel whose controller is FM6124.** An FM6124 that has
never received its register-init sequence is in an undefined state, so what
reached the panel says nothing reliable about register length or scan rate. The
measurement was real; the inference drawn from it was not, because the setup
under measurement was itself misconfigured.

The deeper mistake is worth naming: measuring instead of sweeping presets was
the right instinct, and the previous entry congratulated itself for it. But a
measurement taken through a known-unknown configuration is not more
trustworthy than a sweep — it just *feels* more rigorous. The confidence was
the problem, not the method.

**Ground truth, confirmed by the user against the physical board:**

| Fact | Value |
|---|---|
| Scan rate | **1/8** — A, B, C only, no D line |
| Driver IC | **FM6124** (read off the chip marking) |
| Panels | 2 × 64×32 P5 outdoor, chained side by side → 128×32 |
| Board | ESP32 30-pin dev board (`esp32dev`) |

No D line is wired and none should be. `PIN_D` is `-1` in
`include/config.h`.

---

## 2026-09-03 — FM6124 was never enabled during any of the ~80 sweeps

The single most likely root cause of the entire failure history.

`mxconfig.driver` sat commented out at `src/main.cpp:448` for every one of the
configurations tried — the ~80 hzeller/PioMatter sweeps on the Pi, the five
`setPhysicalPanelScanRate` mappings, the five geometries, and both raw probes.

FM6124 is not a plain shift register. It has internal configuration registers
that must be written with a specific latch sequence at init, and the library
only emits that sequence when `mxconfig.driver = HUB75_I2S_CFG::FM6124` is
set. Without it the controller powers up in an undefined register state, and
**no geometry, scan mapping, pin order or timing change can compensate** —
which is exactly the signature the history shows: many different configurations
failing in similar, unhelpful ways.

This also explains why solid `FILL` always looked perfect while nothing
positional ever rendered. That was previously attributed entirely to coordinate
mapping. Mapping does explain part of it, but an uninitialised FM6124 is the
better explanation for why *no* mapping ever worked.

**Change:** `PANEL_DRIVER = HUB75_I2S_CFG::FM6124` in `include/config.h`,
applied in `display.cpp:begin()`.

---

## 2026-09-03 — Firmware rewritten as modules; 1/8-scan geometry

Split the 508-line `main.cpp` into the layout in the README. The matrix init
was **not** rewritten from scratch — the working parts of the old `#else`
(no-D-line) branch were carried across intact, because that branch already had
the right shape:

```
physical (told to the DMA driver) : 128 × 16, chain 2   -> 256×16
logical  (drawn on by all modes)  : 128 × 32
mapping                           : FOUR_SCAN_32PX_HIGH
```

A 1/8-scan 64×32 panel folds its lower 16 rows into extra columns, so the DMA
layer must be given the internal shape (128×16), not the visible one (64×32).
Configuring 64×32 directly clocks 64 positions into a register that physically
holds 128 — which produces the doubled, tilted output recorded earlier.

### Verified ✅ (over Tailscale to Pi 4, `100.98.53.95`)

| Check | Evidence |
|---|---|
| Compiles clean | `pio run` SUCCESS, 33.5s. RAM 7.9% (25,868 B), Flash 22.1% (290,233 B) |
| Flashes | `pio run -t upload` SUCCESS, hash verified, 39.3s |
| Board boots and answers | `PING` → `PONG` |
| Correct config is live on the board | `INFO` → `canvas=128x32 phys=128x16 chain=2 driver=FM6124 d_line=-1 scan=FOUR_SCAN_32PX_HIGH bright=90` |
| Full command set registered | `HELP` returns all six lines |

The `INFO` command exists precisely so "did my reflash actually take?" is a
one-second question. During bring-up that is worth more than it looks.

### Not verified 🟡

**What the panel physically shows.** Everything above proves the firmware runs
and reports the intended configuration. None of it proves a single LED lights
correctly. That requires someone looking at the sign.

---

## 2026-09-03 — First real hardware observations ✅ (user, looking at the sign)

Four observations, in order. All against `FOUR_SCAN_32PX_HIGH` unless stated.

| # | Test | Observed |
|---|---|---|
| 1 | `MODE:sand` | "filling up with sand but upside down"; later corrected to **bands moving down, not individual pixels** |
| 2 | `DIAG` | green `drawRect(96,0,32,16)` — a small top-right outline — appeared as a **full-width band at the bottom**; letters unrecognisable |
| 3 | Scan sweep, top half filled | **red (`NORMAL_TWO_SCAN`) and green (`NORMAL_ONE_SIXTEEN`) showed nothing.** Blue/amber/white (the three `FOUR_SCAN_*`) each showed a **solid half**, on the **lower** half instead of the upper |
| 4 | Four 8-row stripes, R/G/B/W top-to-bottom | **only blue and red visible** — blue on the top half, red on the bottom half, **clean solid blocks, no tilt** |

### What this establishes

**FM6124 changed the hardware's behaviour.** Before it, every configuration
produced doubled/tilted output. After it, full-width regions render as clean
solid blocks with **no tilt at all** (observation 4). The column folding is
therefore correct now. This is real progress and the flag stays.

**The two `NORMAL_*` mappings are ruled out** — they render nothing at this
geometry (observation 3). Only the `FOUR_SCAN_*` family is viable.

**Banding was a red herring.** `DIAG` banded (observation 2) while a full-width
`RECT` is solid (observations 3, 4). The difference is shape width, not the
panel: narrow logical regions smear, full-width ones do not.

### The remaining fault, stated precisely

Of four 8-row stripes, exactly the two with `(y & 8) == 0` render — logical
rows 0-7 (red) and 16-23 (blue). The two with `(y & 8) != 0` — rows 8-15
(green) and 24-31 (white) — render nowhere.

That bit is exactly what `FOUR_SCAN_32PX_HIGH` keys on. Its mapping is
approximately:

```
(y & 8) == 0 :  x += ((x / 64) + 1) * 64      // shift into the next column block
(y & 8) != 0 :  x += ( x / 64)      * 64
y' = (y / 16) * 8 + (y & 7)
```

So the library expects rows 8-15 and 24-31 to live in a *different column
block* of the same physical rows. On this panel they land nowhere visible.
**The library's assumption about this panel's internal fold does not match the
hardware.** No amount of choosing between the existing presets fixes that if
none of them describes the real fold.

### ⚠️ A claim I made and had to withdraw

On observation 1 I called the sand "the first time positional content has ever
rendered — FM6124 was the missing piece." That was wrong, and the user
corrected it: the sand was moving in *bands*, not individual pixels. I read the
result I was hoping for into an ambiguous description instead of asking what
"upside down" actually looked like.

Worth recording because it is the same failure as the retracted D-line entry
above, one week apart: treating a hoped-for interpretation of thin evidence as
a confirmed result. The panel had genuinely changed behaviour — but "changed"
is not "fixed", and the difference matters when the next person reads this log.

### Next step, not yet done

Per the library maintainer's advice for this exact panel category
(mrcodetastic/ESP32-HUB75-MatrixPanel-DMA discussion #892), run the library's
own **`Pixel_Mapping_Test`** example unmodified against a **single** panel to
measure the real scan mapping, rather than choosing between presets that may
none of them fit. If the true fold is known, a custom mapping is a few lines.

Currently cycling `SCAN:2/3/4` with the four-stripe pattern to check whether
either untested `FOUR_SCAN_*` variant renders all four stripes. 🟡 awaiting
observation.

---

## 2026-09-03 — Pixel_Mapping_Test run; library config space exhausted

Ran the library maintainer's own `Pixel_Mapping_Test` example (the step
recommended in discussion #892), adapted only for panel size, our pins with
D/E at -1, and the confirmed FM6124 flag. Preserved at
`esp32_display/diagnostics/pixel_mapping_test.cpp`.

### The decisive pair of observations ✅

| Test | Result |
|---|---|
| Full-screen fill | **Whole sign lights solid red, both panels, correctly** |
| Single 16px dash at one logical row | **Nothing visible**, any row |

This is the entire problem in two lines. A full fill writes every pixel in the
framebuffer, so every LED lights no matter how scrambled the mapping is — it
looks perfect under a correct mapping and a broken one alike, and is therefore
worthless as a test. Positioned drawing is the only thing that exercises
addressing, and it produces nothing.

It also proves everything upstream is sound: 5V supply, all 12 signal wires,
the FM6124 register-init sequence, the DMA path, and both panels in the chain.
The fault is confined to row addressing.

### Configurations ruled out, with their specific failure modes

| Configuration | Result |
|---|---|
| `NORMAL_TWO_SCAN` | nothing renders |
| `NORMAL_ONE_SIXTEEN` | nothing renders |
| `FOUR_SCAN_32PX_HIGH` | only `(y & 8) == 0` rows render, vertically swapped, clean |
| `FOUR_SCAN_64PX_HIGH` | identical to the above |
| `FOUR_SCAN_16PX_HIGH` | all four row groups render, but sheared with dim bleed; one stripe appeared as 3 bands on one panel and 2 on the other |
| Maintainer's custom `pxbase` mapping | positioned draws invisible entirely |
| `line_decoder = TYPE595` | **ruled out** — see below |

### TYPE595 ruled out ❌

Worth recording because the hypothesis was well-motivated and wrong. A 595-type
panel clocks A/B/C into a shift register to generate row selects, which would
have reconciled the two facts that otherwise conflict: this panel has no D
line, yet is 32 rows tall.

Enabling it made positioned draws appear as **multiple replicated red lines
across both panels** — a change from invisible, which briefly looked like
progress. But it also **broke the full-screen fill**, which works correctly
under the default binary decoder. A panel that fills correctly under binary
addressing and incorrectly under 595 addressing is not a 595 panel. Reverted
the same session; left commented out in the diagnostic so it is not retried.

### Where this leaves it 🟡

The library's configuration space is exhausted. Every built-in scan mapping,
both line decoders, the maintainer's parameterised custom mapping, the
confirmed driver IC and the geometry the maintainer's own example prescribes
have all been tried, and the panel still renders nothing positional.

The remaining signature — one logical pixel producing either nothing or
several physical lines — is **addressing multiplicity**, not coordinate
scrambling. Remapping relocates pixels; it cannot stop one from appearing in
several places or none. So further mapping work is not the answer, and per the
session's own working agreement this is a check-in rather than more sweeping.

Options, in the order I would try them:

1. **Ask upstream.** The example's README says plainly: "Create an issue and
   you will be helped!" The evidence here is unusually clean — driver IC
   confirmed off the chip, full fill working, and a specific failure mode for
   each of five mappings. That is a far better report than most.
2. **Try a different stack.** ESPHome's HUB75 component, or a panel-specific
   driver. The Home Assistant thread already noted this panel class is finicky
   across multiple firmware stacks, so this is not obviously better, but it is
   independent evidence.
3. **Measure the scan rate physically** rather than inferring it — drive one
   address value and count how many rows light. This is the one hardware fact
   still taken on report rather than measured, and both previous wrong
   conclusions in this log came from inferring it.
4. **Substitute the panel.** Indoor P5 1/16-scan panels are well-supported by
   this library and inexpensive. For a thesis with a deadline this is worth
   weighing honestly against more debugging: the firmware, protocol, Pi bridge
   and web integration are all complete and would work unchanged against a
   panel the library supports.

### Rig gotcha

`upload_speed = 921600` **fails reproducibly** (`Failed to leave compressed
flash mode ... C800: Not enough data`) while the panel is displaying a
full-brightness frame — the current draw browns the board out at the end of the
write. `230400` is reliable. This cost two failed flashes before it was
noticed.

## 2026-09-03 - SOLVED (confirmed on the physical sign, with photos)

The sign renders legible text. SAFE in green, VEHICLE INCOMING / SLOW DOWN in
flashing yellow, STOP in flashing red, and the falling-sand test showing
individual coloured grains - all confirmed by the user looking at the panel.

### What fixed it: replacing the library with a hand-written driver

The decisive observation was an asymmetry visible for a while without being
acted on: **a bit-banged test drove this panel correctly on the first attempt,
while the library failed under every configuration it offers.** Same panel,
same wires, same FM6124 init, minutes apart.

The reason is one unverifiable assumption. The library hands you a DMA
framebuffer whose column index is *assumed* to equal the shift-register clock
position. On this panel it does not, and no API in the library lets you correct
for it. Chasing that with scan-mapping presets could never work, because the
presets adjust a mapping sitting on top of the broken assumption.

`src/hub75.cpp` emits the clock pulses itself, so position `p` **is** the p-th
pulse. There is nothing left to assume.

### The measured panel layout, now encoded in code

| Property | Value |
|---|---|
| Scan | 1/8 - A/B/C only, no D line |
| Rows per address | 4, spaced 8 apart |
| Address 0 | the BOTTOM row (row order inverted) |
| Register | 256 positions per channel |
| Panel split | each panel owns 128 consecutive positions |
| Within a panel | first 64 = upper line, second 64 = the line 8 rows below |
| Channels | R1/G1/B1 drive rows 7-a and 15-a; R2/G2/B2 drive 23-a and 31-a |
| Chain | reversed - first-clocked data travels furthest |

Orientation took two further corrections, each read off the panel rather than
reasoned about:

1. **SWAP_PANELS** - centred text split to both outer edges with a gap in the
   middle, the signature of two 64px halves trading places.
2. **FLIP_X** - text then became *readable but rotated 180 degrees*. Readable is
   the diagnostic word: a purely vertical error mirrors glyphs and leaves them
   unreadable, so readable-but-inverted means both axes were wrong together,
   which against this base mapping resolves to X only.

### Verified

| Check | Evidence |
|---|---|
| Compiles, flashes | `pio run -t upload` SUCCESS. RAM 11.6%, Flash 21.2% |
| Refresh rate | `INFO` reports **fps=250**, measured on the board |
| Legible text | photos of SAFE, STOP, VEHICLE INCOMING / SLOW DOWN |
| Per-pixel addressing | falling sand shows individual coloured grains |
| Colour correctness | green, yellow and red all render as intended |

### What this cost, and the lesson

Roughly eighty configuration attempts on the Pi, then five scan mappings, two
line decoders, five geometries and a custom mapping on the ESP32 - none of
which could have worked, because all were adjusting a layer above the faulty
assumption.

Two wrong conclusions were recorded as settled along the way (the D-line
retraction above, and a premature breakthrough call on banded sand output).
Both came from inferring structure instead of measuring it, then writing the
inference down with more confidence than the evidence carried.

What actually worked was the cheapest thing available the whole time: drive the
pins directly, light one known thing, and look at the panel. The bit-banged
address walk and the quarter-coloured register test together took under an hour
and produced every number in the table above.

## Test-design notes worth keeping

Carried forward from `docs/LED_TROUBLESHOOTING.md` because they were learned
expensively and still apply:

- **A solid fill proves nothing about mapping.** `fillScreen` writes every
  pixel, so it looks identical under every scan mapping, correct or not.
  Several hardware sessions were spent on exactly that non-test.
- **Do not draw four colours at once.** Where rows alias, the colours merge and
  the result is unreadable. One colour, one region, held on screen.
- **A test that only reveals the happy path is worse than no test**, because it
  consumes a scarce hardware observation and returns nothing.

This is why falling sand is the bring-up test rather than a static pattern:
grains are individually addressable pixels with predictable motion, so a wrong
mapping is not merely "wrong-looking" but *diagnostic* —

| Symptom | Points at |
|---|---|
| Correct on panel 1, panel 2 dark | chain length |
| Grains teleport half a canvas sideways | `PHYS_W` folding |
| Each grain doubled, 16 rows apart | address lines |
| Falls in 4-row bands that never connect | scan mapping |
| Red and blue swapped | RGB pin order (not a mapping fault) |

## Geometries already tried and failed (all WITHOUT the FM6124 flag)

Worth re-testing only if FM6124 alone does not fix it — and worth remembering
that every row here is contaminated by the missing driver init.

| Physical config | Chain | Result |
|---|---|---|
| 64×32 | 1 | renders, doubled + tilted |
| 64×32 | 2 | renders, doubled + tilted |
| 128×16 | 1 | panel dark |
| 128×16 | 2 | panel dark |
| 256×16 (raw, no remap) | 2 | renders; two bands (expected), still tilted |

All five `setPhysicalPanelScanRate` mappings swept at 64×32, none correct.

## Access

Pi 4 is reachable over Tailscale at `100.98.53.95` as `roadsentinel`, key-based,
**no browser re-auth needed** — contrary to `docs/DEPLOYMENT.md:184`, which says
tailnet ACL forbids SSH. That is stale. The LAN address `192.168.1.18` is *not*
reachable from off-site, so Tailscale is the only path.

The ESP32 is on `/dev/ttyUSB0` on the Pi. Build and flash there:

```bash
ssh roadsentinel@100.98.53.95
cd ~/esp32_display && ~/.pio-venv/bin/pio run -t upload --upload-port /dev/ttyUSB0
```

Stop `roadsentinel-esp32-bridge` first if it is running — two processes cannot
hold the same port, and the failure mode is a silent board with no error.
