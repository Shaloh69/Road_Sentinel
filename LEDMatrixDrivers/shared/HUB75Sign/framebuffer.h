#pragma once
/*
 * The pixel store. Portable — no register access, no board knowledge.
 *
 * One byte per logical pixel holding a 3-bit colour. 128x32 = 4KB, which is
 * nothing on either an ESP32 (320KB) or an STM32F411 (128KB), and far easier
 * to reason about than a packed layout.
 *
 * Colour is 1 bit per channel, giving eight saturated colours. That is a
 * deliberate choice rather than a shortcut: a roadside warning sign wants
 * maximum brightness and contrast, not subtle shades, and 1-bit colour keeps
 * the refresh loop short enough to run flicker-free without binary code
 * modulation timing to get wrong.
 */

#include <Arduino.h>
#include "panel_config.h"

#define HC_OFF     0x0
#define HC_BLUE    0x1
#define HC_GREEN   0x2
#define HC_CYAN    0x3
#define HC_RED     0x4
#define HC_MAGENTA 0x5
#define HC_YELLOW  0x6
#define HC_WHITE   0x7

namespace fbuf {

// Raw store. The refresh loop indexes this directly through the map tables,
// so it is exposed rather than hidden behind accessors — a function call per
// pixel would cost real brightness at ~2000 pixels per frame.
extern uint8_t pixels[CANVAS_W * CANVAS_H];

// Coordinates are LOGICAL: x 0..127 left to right, y 0..31 top to bottom,
// across the whole two-panel sign. Out-of-range writes are dropped rather
// than wrapping, so a drawing bug shows as missing pixels, not corruption
// somewhere unrelated.
inline void set(int16_t x, int16_t y, uint8_t colour) {
  if (x < 0 || x >= CANVAS_W || y < 0 || y >= CANVAS_H) return;
  pixels[y * CANVAS_W + x] = colour & 0x7;
}

inline uint8_t get(int16_t x, int16_t y) {
  if (x < 0 || x >= CANVAS_W || y < 0 || y >= CANVAS_H) return 0;
  return pixels[y * CANVAS_W + x];
}

inline void clear(uint8_t colour = HC_OFF) {
  memset(pixels, colour & 0x7, sizeof(pixels));
}

} // namespace fbuf
