#pragma once
/*
 * Road Sentinel — ESP32 HUB75 display: all hardware facts in one place.
 *
 * Every magic number about the panel lives here. If a value is wrong, it is
 * wrong once, in one file, rather than in six.
 *
 * ─────────────────────────────────────────────────────────────────────────
 * CONFIRMED HARDWARE FACTS (2026-09-03, user-confirmed against the physical
 * board and the chip marking). Do not "fix" these from generic HUB75
 * assumptions — every one of them has already cost debugging time.
 * ─────────────────────────────────────────────────────────────────────────
 */

#include <ESP32-HUB75-MatrixPanel-I2S-DMA.h>

// ── Panel geometry ─────────────────────────────────────────────────────────
// Two 64x32 P5 outdoor panels, daisy-chained side by side (panel 1 OUT ->
// panel 2 IN), read as one banner.
#define PANEL_W       64
#define PANEL_H       32
#define PANEL_CHAIN   2

// The canvas everything draws on. All mode code uses these two and nothing
// else — no code outside display.cpp should know the physical layout.
#define CANVAS_W      (PANEL_W * PANEL_CHAIN)   // 128
#define CANVAS_H      PANEL_H                   // 32

// ── Scan rate: 1/8, and why the physical description is not 64x32 ──────────
//
// These panels are 1/8 scan: they expose A, B, C and NO D line. Three address
// lines = 8 row addresses. Each address drives two rows at once (the upper
// half through R1/G1/B1, the lower through R2/G2/B2), so one scan pass paints
// 8 x 2 = 16 rows.
//
// The panel is 32 rows tall, but only 16 are addressable per pass. The other
// 16 rows are not missing — they are folded into extra COLUMNS in the panel's
// internal shift register. So a 64x32 1/8-scan panel is internally wired
// closer to 128x16, and that is the shape the DMA layer must be told about:
//
//     physical (what the DMA driver clocks out) : 128 x 16, chained x2 = 256x16
//     logical  (what this firmware draws on)    : 128 x 32
//
// VirtualMatrixPanel + FOUR_SCAN_32PX_HIGH is the translation between them.
//
// This is the single most confusing thing about this panel and the reason a
// naive 64x32 config renders content doubled and tilted: at 64x32 the driver
// clocks 64 positions per row into a register that physically holds 128.
#define PHYS_W        (PANEL_W * 2)    // 128 — width doubles...
#define PHYS_H        (PANEL_H / 2)    // 16  — ...as height halves

// ── GPIO map — physically wired, confirmed against the board ───────────────
// HUB75 connector pin numbers in brackets. Changing any of these requires
// moving an actual wire; do not edit to chase a rendering bug.
#define PIN_R1   25   // [1]
#define PIN_G1   26   // [2]
#define PIN_B1   27   // [3]
#define PIN_R2   14   // [5]
#define PIN_G2   12   // [6]
#define PIN_B2   13   // [7]
#define PIN_A    23   // [9]
#define PIN_B    19   // [10]
#define PIN_C     5   // [11]
#define PIN_CLK  16   // [13]
#define PIN_LAT   4   // [14]
#define PIN_OE   15   // [15]

// D [12] and E are deliberately absent.
//
// D: this panel is 1/8 scan and has no D line. Earlier revisions of this repo
// concluded the opposite ("pin 12 is D, not NC") from a shift-register probe
// and recorded it as resolved in HUB75_PINOUT.md. That conclusion was WRONG
// and has been retracted — see DEBUG_LOG.md 2026-09-03. Passing a real GPIO
// here makes the driver emit a fourth address line the panel cannot decode.
//
// E: only 1/32-scan (64-row) panels need it. These are 32-row.
#define PIN_D    -1
#define PIN_E    -1

// ── Driver IC ──────────────────────────────────────────────────────────────
// FM6124, read off the physical chip marking (and consistent with the panel
// label `P5户外全彩 KLB 6124`).
//
// This MATTERS far more than it looks. FM6124 is not a plain shift register:
// it has internal configuration registers that must be written with a
// specific latch sequence before it will display anything correctly. The
// library only emits that sequence when this flag is set. Left at the default
// SHIFTREG, an FM6124 panel powers up in an undefined register state, and NO
// geometry, scan-mapping or pin change can compensate — which is why roughly
// eighty configurations were swept without ever producing readable text.
#define PANEL_DRIVER  HUB75_I2S_CFG::FM6124

// Shifts pixel data half a clock relative to the clock edge. If output is
// offset by exactly one pixel column, this is the knob — and it is the ONLY
// other knob worth touching before asking for a hardware check.
#define PANEL_CLKPHASE  false

// I2S clock. FM6124 panels are generally happy at the library default; 10MHz
// is quieter on long ribbon cables if ghosting appears.
#define PANEL_I2SSPEED  HUB75_I2S_CFG::HZ_10M

// ── Timings ────────────────────────────────────────────────────────────────
#define DEFAULT_BRIGHTNESS   90

// If the Pi goes quiet this long, the board shows "NO DATA" on its own.
// A sign that keeps displaying ROAD CLEAR because its data source died is
// worse than one that admits it does not know. The bridge deliberately never
// sends STATE:offline, so a dead cable and a dead API look the same.
#define COMMAND_TIMEOUT_MS   15000
#define FLASH_INTERVAL_MS    500
#define SCROLL_INTERVAL_MS   40     // ~25 px/sec, readable at driving speed
#define SAND_INTERVAL_MS     33     // ~30 fps

// ── Colours — Night Watch design tokens ────────────────────────────────────
// These are the EXACT hex values from client/web/hero.ts, so the physical
// sign and the web dashboard use one palette. Do not eyeball new ones.
//   amber  #F2B33D  primary-400    green #3DDC97  success-400
//   red    #E5484D  danger-400     blue  #5B9DF5  secondary-400
#define RGB_AMBER    242, 179,  61
#define RGB_GREEN     61, 220, 151
#define RGB_RED      229,  72,  77
#define RGB_BLUE      91, 157, 245
#define RGB_WHITE    255, 255, 255
#define RGB_DIMRED   110,   0,   0
