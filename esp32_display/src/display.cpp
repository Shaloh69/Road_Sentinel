#include "display.h"

namespace display {

static MatrixPanel_I2S_DMA *dma  = nullptr;
static VirtualMatrixPanel  *virt = nullptr;
static uint8_t             bright = DEFAULT_BRIGHTNESS;

uint16_t C_BLACK, C_WHITE, C_RED, C_GREEN, C_AMBER, C_BLUE, C_DIMRED;

bool begin() {
  HUB75_I2S_CFG::i2s_pins pins = {
    PIN_R1, PIN_G1, PIN_B1,
    PIN_R2, PIN_G2, PIN_B2,
    PIN_A,  PIN_B,  PIN_C,
    PIN_D,            // -1: 1/8 scan, this panel has no D line
    PIN_E,            // -1: only 1/32-scan 64-row panels need E
    PIN_CLK, PIN_LAT, PIN_OE
  };

  // PHYS_W x PHYS_H, not PANEL_W x PANEL_H. See the long note in config.h:
  // a 1/8-scan 64x32 panel is internally arranged as 128x16, and the DMA
  // layer must be told the internal shape, not the visible one.
  HUB75_I2S_CFG mxconfig(PHYS_W, PHYS_H, PANEL_CHAIN, pins);

  // The FM6124 register-init sequence. Without this the panel never leaves
  // its power-on register state and no amount of geometry tuning helps.
  mxconfig.driver     = PANEL_DRIVER;
  mxconfig.clkphase   = PANEL_CLKPHASE;
  mxconfig.i2sspeed   = PANEL_I2SSPEED;

  dma = new MatrixPanel_I2S_DMA(mxconfig);
  if (!dma->begin()) return false;

  dma->setBrightness8(bright);
  dma->clearScreen();

  // 1 row of 2 modules, each 64x32 -> a 128x32 logical canvas laid over the
  // 256x16 physical surface above.
  virt = new VirtualMatrixPanel(*dma, 1, PANEL_CHAIN, PANEL_W, PANEL_H);

  // The mapping that unfolds the panel's internal 128x16 arrangement back
  // into the visible 64x32. "FOUR_SCAN_32PX_HIGH" is the library's name for
  // the 1/8-scan-on-a-32px-tall-panel case, despite the confusing name — it
  // refers to the internal four-row grouping, not to a 1/4 scan rate.
  virt->setPhysicalPanelScanRate(FOUR_SCAN_32PX_HIGH);

  C_BLACK  = virt->color565(0, 0, 0);
  C_WHITE  = virt->color565(RGB_WHITE);
  C_RED    = virt->color565(RGB_RED);
  C_GREEN  = virt->color565(RGB_GREEN);
  C_AMBER  = virt->color565(RGB_AMBER);
  C_BLUE   = virt->color565(RGB_BLUE);
  C_DIMRED = virt->color565(RGB_DIMRED);
  return true;
}

VirtualMatrixPanel *gfx()  { return virt; }
MatrixPanel_I2S_DMA *raw() { return dma; }

void setBrightness(uint8_t b) {
  bright = b;
  if (dma) dma->setBrightness8(b);
}
uint8_t brightness() { return bright; }

int textWidth(const String &s, uint8_t size) {
  // The built-in GFX font is a fixed 5x7 cell drawn in a 6x8 box, so width is
  // exact arithmetic — no need to call getTextBounds and no dependence on the
  // cursor's current position, which getTextBounds is sensitive to.
  return (int)s.length() * 6 * size;
}

void drawAt(const String &s, int x, int y, uint16_t colour, uint8_t size) {
  virt->setTextSize(size);
  virt->setTextColor(colour);
  virt->setTextWrap(false);        // must not wrap: scrollers draw off-canvas
  virt->setCursor(x, y);
  virt->print(s);
}

void drawCentered(const String &s, int y, uint16_t colour, uint8_t size) {
  int x = (CANVAS_W - textWidth(s, size)) / 2;
  if (x < 0) x = 0;
  drawAt(s, x, y, colour, size);
}

} // namespace display
