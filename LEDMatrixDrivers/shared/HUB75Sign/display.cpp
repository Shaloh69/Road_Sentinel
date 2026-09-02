#include "display.h"

// Colours pass through Adafruit_GFX's uint16_t argument unchanged — they are
// the driver's 3-bit codes, not RGB565. No conversion happens anywhere, which
// removes a whole class of "why is my red white" confusion.
void Canvas::drawPixel(int16_t x, int16_t y, uint16_t colour) {
  hub75::setPixel(x, y, (uint8_t)(colour & 0x7));
}

void Canvas::fillScreen(uint16_t colour) {
  hub75::clear((uint8_t)(colour & 0x7));
}

namespace display {

static Canvas canvas;

void begin() {
  hub75::begin();
}

Canvas *gfx() { return &canvas; }

void setBrightness(uint8_t b) { hub75::setBrightness(b); }
uint8_t brightness()          { return hub75::brightness(); }

int textWidth(const String &s, uint8_t size) {
  // The built-in GFX font is a fixed 5x7 glyph in a 6x8 cell, so width is
  // exact arithmetic. Avoids getTextBounds, which depends on cursor position.
  return (int)s.length() * 6 * size;
}

void drawAt(const String &s, int x, int y, uint16_t colour, uint8_t size) {
  canvas.setTextSize(size);
  canvas.setTextColor(colour);
  canvas.setTextWrap(false);      // scrollers draw deliberately off-canvas
  canvas.setCursor(x, y);
  canvas.print(s);
}

void drawCentered(const String &s, int y, uint16_t colour, uint8_t size) {
  int x = (CANVAS_W - textWidth(s, size)) / 2;
  if (x < 0) x = 0;
  drawAt(s, x, y, colour, size);
}

} // namespace display
