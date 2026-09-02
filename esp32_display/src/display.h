#pragma once
/*
 * Matrix initialisation and shared drawing helpers.
 *
 * This is the only file that knows the panel's physical layout. Everything
 * else draws on a plain 128x32 canvas through `gfx()` and stays ignorant of
 * scan rates, chaining and address lines.
 */

#include <ESP32-HUB75-MatrixPanel-I2S-DMA.h>
#include <ESP32-VirtualMatrixPanel-I2S-DMA.h>
#include "config.h"

namespace display {

// Bring up the panel. Returns false if the DMA allocation failed, which in
// practice means the ESP32 is out of DMA-capable memory.
bool begin();

// The LOGICAL 128x32 canvas. All mode code draws here.
//
// Drawing through the virtual panel rather than the DMA object is not
// optional on a 1/8-scan panel: the virtual panel is what remaps logical
// (x,y) onto the panel's folded internal layout. Draw straight to the DMA
// object and a solid fill still looks perfect — every LED gets written
// either way — while anything positional lands in the wrong place. That
// asymmetry is exactly what made this hard to diagnose.
VirtualMatrixPanel *gfx();

// The RAW physical surface, 256x16, no remapping. Diagnostics only.
MatrixPanel_I2S_DMA *raw();

void setBrightness(uint8_t b);
uint8_t brightness();

// Palette, resolved to RGB565 once at init.
extern uint16_t C_BLACK, C_WHITE, C_RED, C_GREEN, C_AMBER, C_BLUE, C_DIMRED;

// Width in pixels that `s` would occupy at the given GFX text size.
int textWidth(const String &s, uint8_t size);

// Draw `s` horizontally centred on the canvas at row `y`.
void drawCentered(const String &s, int y, uint16_t colour, uint8_t size);

// Draw `s` with its left edge at `x`, which may be negative or off-canvas —
// used by the scrollers, where the text deliberately starts off-screen.
void drawAt(const String &s, int x, int y, uint16_t colour, uint8_t size);

} // namespace display
