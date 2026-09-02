#include "mode_char.h"
#include "display.h"

namespace char_mode {

static char cur = 'A';

void setChar(char c) { cur = c; render(); }
char current()       { return cur; }

void render() {
  display::gfx()->fillScreen(display::C_BLACK);

  // Size 4 is the largest that fits a 32px-tall canvas: the GFX base font is
  // 7px tall in an 8px box, so 4x = 28px drawn in 32. Centred by arithmetic
  // rather than getTextBounds, which is cursor-sensitive.
  const uint8_t size = 4;
  String s(cur);
  int x = (CANVAS_W - display::textWidth(s, size)) / 2;
  int y = (CANVAS_H - 8 * size) / 2;
  display::drawAt(s, x, y, display::C_WHITE, size);

  // Corner ticks: they mark the true extents of the logical canvas, so a
  // character that looks centred but sits on a shifted or clipped canvas is
  // still detectable. Without them a uniformly-offset mapping looks fine.
  display::gfx()->drawPixel(0, 0, display::C_AMBER);
  display::gfx()->drawPixel(CANVAS_W - 1, 0, display::C_AMBER);
  display::gfx()->drawPixel(0, CANVAS_H - 1, display::C_AMBER);
  display::gfx()->drawPixel(CANVAS_W - 1, CANVAS_H - 1, display::C_AMBER);
}

} // namespace char_mode
