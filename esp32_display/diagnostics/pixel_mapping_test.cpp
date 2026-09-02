/*
 * Pixel_Mapping_Test — the library maintainer's own scan-mapping diagnostic,
 * adapted for Road Sentinel's panel. NOT production firmware.
 *
 * This is a MEASUREMENT, not another configuration guess. It fills the panel
 * one pixel at a time, left to right and top to bottom, slowly enough to
 * watch. How the panel actually fills reveals the mapping directly:
 *
 *   fills cleanly L->R, top->bottom   -> mapping is correct
 *   fills in short segments of N px   -> N is the panel's PIXEL BASE
 *   rows appear in the wrong order    -> the row-block expressions are swapped
 *   a block fills R->L                -> that block needs the reversed formula
 *
 * The pixel base is the value this whole exercise has been missing. The
 * production firmware assumes pxbase == panelResX == 64. If this panel's real
 * base is 8 or 16, every mapping tried so far is wrong in the same way, which
 * fits the observed symptoms: one colour band repeating several times within a
 * panel, shear across the width, and the two chained panels disagreeing.
 *
 * Deviations from the stock example, and why each is justified:
 *   - PANEL_RES_X/Y set to 64x32 (our panels, not the example's 32x16)
 *   - NUM_COLS = 1: the maintainer's step 1 says test a SINGLE panel first.
 *     Only the first panel in the chain will light. That is expected.
 *   - explicit pins with D = -1 and E = -1. The library's defaults are D=17,
 *     E=32; leaving them would drive address lines this 1/8-scan panel does
 *     not have, which is the exact fault this project already spent a week on.
 *   - mxconfig.driver = FM6124, a confirmed hardware fact. Leaving it at the
 *     default SHIFTREG would repeat the mistake that invalidated the earlier
 *     shift-register probe: measuring through a knowingly wrong configuration.
 *
 * Everything else is the stock example, unmodified.
 */

#include <Arduino.h>
#include "ESP32-VirtualMatrixPanel-I2S-DMA.h"

class CustomPxBasePanel : public VirtualMatrixPanel {
  public:
    using VirtualMatrixPanel::VirtualMatrixPanel;
  protected:
    VirtualCoords getCoords(int16_t x, int16_t y);
};

inline VirtualCoords CustomPxBasePanel::getCoords(int16_t x, int16_t y) {
  coords = VirtualMatrixPanel::getCoords(x, y);
  if (coords.x == -1 || coords.y == -1) return coords;

  // THE VALUE UNDER TEST. Stock example uses panelResX (64 for our panels).
  // If the panel fills in short segments, change this to that segment length.
  uint8_t pxbase = panelResX;

  if (panelResY == 32) {
    if ((coords.y & 8) == 0)
      coords.x += ((coords.x / pxbase) + 1) * pxbase;   // 1st, 3rd block of 8 rows
    else
      coords.x += (coords.x / pxbase) * pxbase;         // 2nd, 4th block of 8 rows
    coords.y = (coords.y >> 4) * 8 + (coords.y & 0b00000111);
  }
  else if (panelResY == 16) {
    if ((coords.y & 4) == 0)
      coords.x += ((coords.x / pxbase) + 1) * pxbase;
    else
      coords.x += (coords.x / pxbase) * pxbase;
    coords.y = (coords.y >> 3) * 4 + (coords.y & 0b00000011);
  }
  else {
    uint8_t half_height = panelResY / 2;
    if ((coords.y % half_height) < half_height / 2)
      coords.x += (coords.x / pxbase + 1) * pxbase;
    else
      coords.x += (coords.x / pxbase) * pxbase;
    coords.y = (coords.y / half_height) * (half_height / 2)
             + (coords.y % (half_height / 2));
  }
  return coords;
}

#define PANEL_RES_X 64
#define PANEL_RES_Y 32
#define NUM_ROWS    1
#define NUM_COLS    1      // single panel, per the example's step 1

#define VIRTUAL_MATRIX_CHAIN_TYPE CHAIN_BOTTOM_RIGHT_UP

MatrixPanel_I2S_DMA *dma_display   = nullptr;
CustomPxBasePanel   *FourScanPanel = nullptr;

void setup() {
  Serial.begin(115200);
  delay(300);
  Serial.println("Pixel_Mapping_Test — Road Sentinel panel");

  HUB75_I2S_CFG::i2s_pins pins = {
    25, 26, 27,    // R1, G1, B1
    14, 12, 13,    // R2, G2, B2
    23, 19,  5,    // A,  B,  C
    -1,            // D — 1/8 scan, this panel has no D line
    -1,            // E — 32-row panel, no E
    16,  4, 15     // CLK, LAT, OE
  };

  HUB75_I2S_CFG mxconfig(
    PANEL_RES_X * 2,        // DO NOT CHANGE (example's instruction)
    PANEL_RES_Y / 2,        // DO NOT CHANGE
    NUM_ROWS * NUM_COLS,    // DO NOT CHANGE
    pins
  );

  mxconfig.clkphase = false;
  mxconfig.driver   = HUB75_I2S_CFG::FM6124;   // confirmed chip marking

  // THE VARIABLE UNDER TEST, and the only untried one that fits the evidence.
  //
  // A TYPE595 panel does not use A/B/C as binary address bits. It clocks them
  // into a 595 shift register that generates the row-select lines, so three
  // pins can address 16 rows. That reconciles two facts that otherwise
  // conflict: this panel has no D line, yet it is 32 rows tall.
  //
  // It also explains the exact symptom pair seen on the hardware — a
  // full-screen fill lights every LED perfectly (it writes the whole
  // framebuffer, so addressing never matters), while every positioned draw
  // fails under every binary-addressing mapping in the library.
  // RULED OUT 2026-09-03. Enabling TYPE595 broke the full-screen fill that
  // works correctly with the default binary decoder, and made positioned
  // draws appear as replicated lines across both panels. A panel that fills
  // correctly under binary addressing and incorrectly under 595 addressing is
  // not a 595-type panel. Left here, disabled, so it is not retried.
  // mxconfig.line_decoder = HUB75_I2S_CFG::TYPE595;

  dma_display = new MatrixPanel_I2S_DMA(mxconfig);
  dma_display->setBrightness8(180);
  if (!dma_display->begin())
    Serial.println("****** !KABOOM! I2S memory allocation failed ***********");

  dma_display->clearScreen();
  delay(500);

  FourScanPanel = new CustomPxBasePanel((*dma_display), NUM_ROWS, NUM_COLS,
                                        PANEL_RES_X, PANEL_RES_Y,
                                        VIRTUAL_MATRIX_CHAIN_TYPE);
  Serial.printf("filling %dx%d one pixel at a time\n",
                FourScanPanel->width(), FourScanPanel->height());
}

void loop() {
  // Phase 0 — anchor. A full red screen proves power, pins, driver init and
  // the DMA path all work. If this is dark, nothing below means anything and
  // the problem is upstream of any mapping question.
  dma_display->fillScreenRGB888(200, 0, 0);
  Serial.println("PHASE 0: full red - panel alive check");
  delay(3000);

  // Phases 1-4 — one short DASH per 8-row block, not a single pixel.
  //
  // A single lit LED is invisible across a room on an outdoor P5 panel, which
  // made the previous version of this test useless. A 16px dash is still
  // localised enough that duplicates are countable, but bright enough to see.
  //
  // The question each phase answers: does ONE logical dash produce ONE
  // physical dash? If it appears in two or four places, one address is driving
  // several rows and no coordinate remapping can fix that.
  const int probes[4] = {0, 8, 16, 24};
  for (int p = 0; p < 4; p++) {
    dma_display->clearScreen();
    delay(300);
    for (int x = 8; x < 24; x++)
      FourScanPanel->drawPixel(x, probes[p], FourScanPanel->color565(255, 0, 0));
    Serial.printf("PHASE %d: one 16px dash at logical row %d\n", p + 1, probes[p]);
    delay(6000);
  }
}
