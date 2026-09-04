#pragma once
/*
 * Road Sentinel — HUB75 sign: all hardware facts in one place.
 *
 * Every magic number about the panel lives here, so a wrong value is wrong
 * once rather than in six files.
 *
 * ─────────────────────────────────────────────────────────────────────────
 * CONFIRMED HARDWARE FACTS. Do not "correct" these from generic HUB75
 * assumptions — several are counterintuitive and each has already cost real
 * debugging time. The scan structure in particular was measured directly off
 * the hardware (see hub75.h and DEBUG_LOG.md), after being inferred wrongly
 * twice from specifications and silkscreens.
 * ─────────────────────────────────────────────────────────────────────────
 */

// ── Panel geometry ─────────────────────────────────────────────────────────
// Two 64x32 P5 outdoor panels, daisy-chained (panel 1 OUT -> panel 2 IN),
// mounted side by side to read as one banner.
#define PANEL_W       64
#define PANEL_H       32
#define PANEL_CHAIN   2

// The logical canvas everything draws on: x 0..127 left to right,
// y 0..31 top to bottom.
#define CANVAS_W      (PANEL_W * PANEL_CHAIN)   // 128
#define CANVAS_H      PANEL_H                   // 32

// ── Scan structure — MEASURED, not assumed ─────────────────────────────────
// 1/8 scan: A, B, C only, no D line. Three address lines = 8 addresses, and
// each address lights FOUR rows spaced 8 apart (8 x 4 = 32 rows exactly).
#define SCAN_ADDRESSES 8

// Clock positions per address, per colour channel. Each panel owns 128
// consecutive positions (64 for its upper line, 64 for the line 8 rows down),
// and there are two panels.
#define REGISTER_LEN  (PANEL_W * 2 * PANEL_CHAIN)   // 256

// ── GPIO map — per board ───────────────────────────────────────────────────
// The panel is identical on both boards; only the wiring differs. Both maps
// were chosen so that all six data lines plus the three address lines sit on
// ONE port, which lets the refresh loop move them with a single register
// write. That is what makes a bit-banged driver fast enough to be
// flicker-free, and it is the main constraint when picking pins.
//
// HUB75 connector pin in brackets.

#if defined(ARDUINO_ARCH_ESP32)

  // ESP32 dev board — drives BOTH signs (Pi 4 and Pi 5). The two boards run
  // identical firmware; nothing here is per-installation.
  //
  // Every pin is below GPIO 32 so the single 32-bit W1TS/W1TC registers cover
  // all of them.
  //
  // Proven good by direct measurement: a bit-banged walk of addresses 0..7 lit
  // 8 distinct rows, and a quarter-coloured register test placed all four
  // quarters correctly. That exercises every pin here.
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

#elif defined(ARDUINO_ARCH_STM32)

  // WeAct Black Pill V3.0, STM32F411CEU6.
  //
  // REFERENCE PORT — not used in the Road Sentinel deployment. Both signs run
  // on ESP32. Kept because it demonstrates that a new board is genuinely one
  // file, which is the whole claim of this library's structure. Never
  // hardware-verified: it compiles and nothing more.
  //
  // Data and address lines are all on GPIOB so one BSRR write moves them
  // together. PB2 is deliberately skipped: it is the BOOT1 strapping pin, and
  // driving it interferes with entering the DFU bootloader.
  //
  // Control lines sit on GPIOA. Splitting them off costs nothing — the clock
  // is pulsed separately from the data anyway — and it keeps GPIOB's mask
  // clean.
  #define PIN_R1   PB0   // [1]
  #define PIN_G1   PB1   // [2]
  #define PIN_B1   PB3   // [3]   PB2 skipped: BOOT1
  #define PIN_R2   PB4   // [5]
  #define PIN_G2   PB5   // [6]
  #define PIN_B2   PB6   // [7]
  #define PIN_A    PB7   // [9]
  #define PIN_B    PB8   // [10]
  #define PIN_C    PB9   // [11]
  #define PIN_CLK  PA0   // [13]
  #define PIN_LAT  PA1   // [14]
  #define PIN_OE   PA2   // [15]

#else
  #error "Unsupported board - add a pin map in panel_config.h"
#endif

// D [12] and E are deliberately absent on BOTH boards and must stay that way.
//
// D: these panels are 1/8 scan and have no D line. An earlier revision of this
// project concluded the opposite from a shift-register probe and recorded it
// as resolved; that was retracted (see DEBUG_LOG.md). Driving a fourth address
// line the panel cannot decode breaks row addressing no matter what else is
// set.
//
// E: only 1/32-scan 64-row panels need it. These are 32-row.

// ── Orientation ────────────────────────────────────────────────────────────
// Applied to the finished logical canvas, so they are independent of the scan
// mapping and safe to change without touching hub75.cpp. Set from what the
// sign actually showed, not from theory.
//
// Settled empirically, in this order:
//
//   SWAP_PANELS  centred text split to both outer edges with a gap in the
//                middle, so the two 64px halves had traded places.
//   then         text became READABLE but rotated 180 degrees. Readable is the
//                important word: a purely vertical error mirrors letters and
//                leaves them unreadable, so readable-but-inverted means both
//                axes are wrong together.
//   FLIP_X       correcting a 180 rotation means flipping both axes, which
//                against the base mapping works out to X only.
//
// If the sign is ever remounted the other way up, toggle both FLIP_X and
// FLIP_Y together — that is the 180 rotation. Toggling just one mirrors the
// text and makes it unreadable.
// SWAP_HALVES  which data channel feeds which 16-row half. Observed as each
//              glyph having its top half on the bottom and vice versa: the
//              canvas rows 0-15 and 16-31 had traded places, which is what a
//              reversed R1/R2 channel assignment does.
#define SWAP_HALVES   1
#define FLIP_Y        0
#define FLIP_X        1
#define SWAP_PANELS   1

// ── Timings ────────────────────────────────────────────────────────────────
#define DEFAULT_BRIGHTNESS   140

// If the Pi goes quiet this long the board says so itself. A sign that keeps
// showing SAFE because its data source died is worse than one admitting it
// does not know. The bridge deliberately never sends the offline state, so a
// dead cable and a dead API produce the same honest result.
#define COMMAND_TIMEOUT_MS   15000
#define FLASH_INTERVAL_MS    450
#define SCROLL_INTERVAL_MS   40     // ~25 px/sec, readable at driving speed
#define SAND_INTERVAL_MS     33     // ~30 fps

// The shape the falling sand resolves into as it settles.
#define SAND_SHAPE_TEXT      "FAK YU!!"
#define SAND_SHAPE_SIZE      2
