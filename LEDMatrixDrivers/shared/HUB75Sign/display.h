#pragma once
/*
 * The drawing surface: Adafruit_GFX on top of our own HUB75 driver.
 *
 * Adafruit_GFX is kept deliberately. It is a pure software graphics library —
 * fonts, text, shapes, clipping — and has nothing to do with driving a panel.
 * It was never the source of any problem here; the HUB75 DMA library was, and
 * that one is gone. Hand-rolling a font table would have added risk for no
 * benefit.
 *
 * Everything above this file draws on a plain 128x32 canvas and knows nothing
 * about addresses, shift registers or chain order.
 */

#include <Adafruit_GFX.h>
#include "panel_config.h"
#include "hub75.h"

class Canvas : public Adafruit_GFX {
  public:
    Canvas() : Adafruit_GFX(CANVAS_W, CANVAS_H) {}
    void drawPixel(int16_t x, int16_t y, uint16_t colour) override;
    void fillScreen(uint16_t colour) override;
};

namespace display {

void begin();

// The 128x32 canvas. All mode code draws here.
Canvas *gfx();

void    setBrightness(uint8_t b);
uint8_t brightness();

// Width in pixels that `s` occupies at the given GFX text size.
int textWidth(const String &s, uint8_t size);

// Horizontally centred at row `y`.
void drawCentered(const String &s, int y, uint16_t colour, uint8_t size);

// Left edge at `x`, which may be negative or off-canvas — the scrollers rely
// on drawing partly outside the visible area.
void drawAt(const String &s, int x, int y, uint16_t colour, uint8_t size);

} // namespace display
