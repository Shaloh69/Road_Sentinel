#include <Arduino.h>
/*
 * Road Sentinel — ESP32 HUB75 display driver
 *
 * The ESP32 does one job: drive the LED panel. It has no WiFi, no clock, no
 * knowledge of the system. The Raspberry Pi stays the only networked device
 * and tells this board what to show over a USB serial line.
 *
 * Why serial rather than WiFi on the ESP32: one fewer thing to provision at
 * the installation site, one fewer credential to rotate, and a wired link
 * cannot drop out the way site WiFi does. The Pi already has the detection
 * pipeline and network access, so putting the intelligence there and keeping
 * this board dumb is both simpler and easier to debug — if the panel is wrong,
 * it is this file; if the content is wrong, it is the Pi.
 *
 * Hardware
 *   ESP32 dev board -> HUB75 panel, direct GPIO (see raspi_scripts/HUB75_PINOUT.md)
 *   Panel power is SEPARATE: 5V 8A into the panels' own terminals, with the
 *   supply ground tied to ESP32 ground. The 16 data wires carry no useful power.
 *
 * Library
 *   ESP32-HUB75-MatrixPanel-DMA by mrcodetastic (Arduino Library Manager)
 *
 * Protocol — newline-terminated ASCII, 115200 baud
 *   STATE:clear            road clear (green)
 *   STATE:vehicle          vehicle incoming (amber, flashing)
 *   STATE:incident         incident ahead (red, flashing)
 *   STATE:offline          no data from the Pi (dim blue)
 *   TEXT:line1|line2       arbitrary two-line message, white on black
 *   FILL:r,g,b             solid full-screen colour (panel diagnostics)
 *   RECT:x,y,w,h,r,g,b     one filled rectangle, no clear (region probing)
 *   CLS                    clear to black
 *   DIAG                   primitives test pattern
 *   SCAN:0-4               swap physical scan mapping at runtime
 *                          0 TWO_SCAN 1 ONE_SIXTEEN 2 FOUR_32 3 FOUR_16 4 FOUR_64
 *   BRIGHT:0-255           panel brightness
 *   PING                   -> replies PONG, for the Pi's health check
 *
 * Every command is acknowledged with "OK" or "ERR <reason>" so the Pi can tell
 * the difference between "board is wedged" and "board disagreed with me".
 */

#include <ESP32-HUB75-MatrixPanel-I2S-DMA.h>
#include <ESP32-VirtualMatrixPanel-I2S-DMA.h>

// ── Panel geometry ─────────────────────────────────────────────────────────
// Counted on the hardware: 64 wide x 32 tall per panel, two chained -> 128x32.
#define PANEL_W     64
#define PANEL_H     32
#define PANEL_CHAIN 2

MatrixPanel_I2S_DMA *dma_display = nullptr;

// These panels are 1/8 scan outdoor units, whose internal pixel order does not
// match the straightforward row-by-row layout the DMA class assumes. The
// library's own docs are explicit that a 64x32 1/8-scan panel "requires custom
// co-ordinate remapping logic" — VirtualMatrixPanel is that layer.
//
// This is why solid colours looked perfect while nothing positional rendered:
// fillScreen writes the whole framebuffer, so every physical LED lights no
// matter how the mapping is scrambled, but drawPixel/fillRect/text all depend
// on the mapping being right.
//
// ALL drawing goes through `display`. Only setup/brightness touch dma_display.
VirtualMatrixPanel *display = nullptr;

// ── Colours ────────────────────────────────────────────────────────────────
uint16_t C_BLACK, C_WHITE, C_RED, C_GREEN, C_AMBER, C_BLUE, C_DIMRED;

// ── State ──────────────────────────────────────────────────────────────────
enum DisplayState { ST_BOOT, ST_CLEAR, ST_VEHICLE, ST_INCIDENT, ST_OFFLINE,
                    ST_TEXT, ST_FILL, ST_DIAG };
DisplayState state = ST_BOOT;

String  textLine1 = "";
String  textLine2 = "";
uint8_t brightness = 90;

// Solid-colour fill, for panel diagnostics. A flat full-screen colour is the
// clearest possible test signal: any banding, wrong hue, or dead region shows
// up immediately with no text rendering to confuse the picture.
uint8_t fillR = 0, fillG = 0, fillB = 0;

bool     flashOn      = false;
uint32_t lastFlash    = 0;
uint32_t lastCommand  = 0;

// If the Pi goes quiet for this long, fall back to OFFLINE rather than showing
// stale information. A sign that confidently displays "ROAD CLEAR" because its
// data source died is worse than one that admits it does not know.
const uint32_t COMMAND_TIMEOUT_MS = 15000;
const uint32_t FLASH_INTERVAL_MS  = 500;

// ── Drawing helpers ────────────────────────────────────────────────────────

void drawCentered(const String &s, int y, uint16_t colour, uint8_t size) {
  display->setTextSize(size);
  display->setTextColor(colour);
  int16_t x1, y1;
  uint16_t w, h;
  display->getTextBounds(s, 0, 0, &x1, &y1, &w, &h);
  int x = ((PANEL_W * PANEL_CHAIN) - (int)w) / 2 - x1;
  if (x < 0) x = 0;
  display->setCursor(x, y);
  display->print(s);
}

void drawTwoLines(const String &big, const String &small,
                  uint16_t bg, uint16_t fgBig, uint16_t fgSmall) {
  display->fillScreen(bg);
  drawCentered(big, 2, fgBig, 2);
  drawCentered(small, 20, fgSmall, 1);
}

// ── Screens ────────────────────────────────────────────────────────────────

void renderClear() {
  display->fillScreen(C_BLACK);
  drawCentered("ROAD CLEAR", 12, C_GREEN, 2);
}

void renderVehicle() {
  // Flashing so it reads as "act now" rather than ambient information.
  uint16_t bg = flashOn ? C_AMBER : C_BLACK;
  uint16_t fg = flashOn ? C_BLACK : C_AMBER;
  drawTwoLines("VEHICLE", "SLOW DOWN", bg, fg, fg);
}

void renderIncident() {
  uint16_t bg = flashOn ? C_RED : C_DIMRED;
  drawTwoLines("INCIDENT", "AHEAD - SLOW DOWN", bg, C_WHITE, C_WHITE);
}

void renderOffline() {
  display->fillScreen(C_BLACK);
  drawCentered("-- NO DATA --", 12, C_BLUE, 1);
}

void renderText() {
  display->fillScreen(C_BLACK);
  if (textLine2.length() > 0) {
    drawCentered(textLine1, 2, C_WHITE, 2);
    drawCentered(textLine2, 20, C_AMBER, 1);
  } else {
    drawCentered(textLine1, 12, C_WHITE, 2);
  }
}

void renderFill() {
  display->fillScreen(display->color565(fillR, fillG, fillB));
}

void renderDiag() {
  display->fillScreen(C_BLACK);
  // 1. Solid block, top-left quadrant — tests fillRect coordinates
  display->fillRect(0, 0, 32, 16, C_RED);
  // 2. Outlined box, top-right — tests drawRect
  display->drawRect(96, 0, 32, 16, C_GREEN);
  // 3. Diagonal line across the whole panel — tests drawLine
  display->drawLine(0, 0, 127, 31, C_BLUE);
  // 4. Individual pixels along the bottom edge — the most primitive test
  for (int x = 0; x < 128; x += 4) display->drawPixel(x, 31, C_WHITE);
  // 5. A single large character — tests GFX text specifically
  display->setTextSize(2);
  display->setTextColor(C_WHITE);
  display->setCursor(50, 8);
  display->print("A");
  // 6. A small character, in case size 2 is the problem
  display->setTextSize(1);
  display->setTextColor(C_AMBER);
  display->setCursor(50, 24);
  display->print("b");
}

void renderBoot() {
  display->fillScreen(C_BLACK);
  drawCentered("ROAD SENTINEL", 8, C_AMBER, 1);
  drawCentered("waiting for Pi", 20, C_BLUE, 1);
}

void render() {
  switch (state) {
    case ST_CLEAR:    renderClear();    break;
    case ST_VEHICLE:  renderVehicle();  break;
    case ST_INCIDENT: renderIncident(); break;
    case ST_OFFLINE:  renderOffline();  break;
    case ST_TEXT:     renderText();     break;
    case ST_FILL:     renderFill();     break;
    case ST_DIAG:     renderDiag();     break;
    default:          renderBoot();     break;
  }
}

// ── Serial protocol ────────────────────────────────────────────────────────

void handleCommand(String cmd) {
  cmd.trim();
  if (cmd.length() == 0) return;

  lastCommand = millis();

  if (cmd == "PING") {
    Serial.println("PONG");
    return;
  }

  if (cmd.startsWith("STATE:")) {
    String v = cmd.substring(6);
    v.trim();
    if      (v == "clear")    state = ST_CLEAR;
    else if (v == "vehicle")  state = ST_VEHICLE;
    else if (v == "incident") state = ST_INCIDENT;
    else if (v == "offline")  state = ST_OFFLINE;
    else { Serial.println("ERR unknown state"); return; }
    render();
    Serial.println("OK");
    return;
  }

  if (cmd.startsWith("TEXT:")) {
    String payload = cmd.substring(5);
    int bar = payload.indexOf('|');
    if (bar >= 0) {
      textLine1 = payload.substring(0, bar);
      textLine2 = payload.substring(bar + 1);
    } else {
      textLine1 = payload;
      textLine2 = "";
    }
    state = ST_TEXT;
    render();
    Serial.println("OK");
    return;
  }

  if (cmd.startsWith("RECT:")) {
    // RECT:x,y,w,h,r,g,b — draw one filled rectangle without clearing first,
    // so regions can be probed one at a time to map what is addressable.
    String v = cmd.substring(5);
    int vals[7]; int idx = 0; int start = 0;
    for (int i = 0; i <= v.length() && idx < 7; i++) {
      if (i == v.length() || v.charAt(i) == ',') {
        vals[idx++] = v.substring(start, i).toInt();
        start = i + 1;
      }
    }
    if (idx < 7) { Serial.println("ERR expected RECT:x,y,w,h,r,g,b"); return; }
    display->fillRect(vals[0], vals[1], vals[2], vals[3],
                          display->color565(vals[4], vals[5], vals[6]));
    state = ST_FILL;   // keep loop() from redrawing over it
    Serial.println("OK");
    return;
  }

  if (cmd.startsWith("RECTPX:")) {
    // Same rectangle as RECT, but drawn with drawPixel only — bypassing the
    // library's optimised fillRect/drawFastHLine path. If this renders and
    // RECT does not, those overrides are the fault, not our coordinates.
    String v = cmd.substring(7);
    int vals[7]; int idx = 0; int start = 0;
    for (int i = 0; i <= v.length() && idx < 7; i++) {
      if (i == v.length() || v.charAt(i) == ',') {
        vals[idx++] = v.substring(start, i).toInt();
        start = i + 1;
      }
    }
    if (idx < 7) { Serial.println("ERR expected RECTPX:x,y,w,h,r,g,b"); return; }
    uint16_t c = display->color565(vals[4], vals[5], vals[6]);
    for (int yy = vals[1]; yy < vals[1] + vals[3]; yy++)
      for (int xx = vals[0]; xx < vals[0] + vals[2]; xx++)
        display->drawPixel(xx, yy, c);
    state = ST_FILL;
    Serial.println("OK");
    return;
  }

  if (cmd.startsWith("SCAN:")) {
    // Swap the physical scan mapping at runtime. Which mapping a given panel
    // needs is not derivable from the datasheet — the library's own docs call
    // these user-contributed and say results vary by panel — so being able to
    // sweep them without reflashing turns a 20-second cycle into a 1-second one.
    int n = cmd.substring(5).toInt();
    switch (n) {
      case 0: display->setPhysicalPanelScanRate(NORMAL_TWO_SCAN);     break;
      case 1: display->setPhysicalPanelScanRate(NORMAL_ONE_SIXTEEN);  break;
      case 2: display->setPhysicalPanelScanRate(FOUR_SCAN_32PX_HIGH); break;
      case 3: display->setPhysicalPanelScanRate(FOUR_SCAN_16PX_HIGH); break;
      case 4: display->setPhysicalPanelScanRate(FOUR_SCAN_64PX_HIGH); break;
      default: Serial.println("ERR scan 0-4"); return;
    }
    render();
    Serial.println("OK");
    return;
  }

  if (cmd.startsWith("RAWROW:")) {
    // RAWROW:y,r,g,b - one full-width row in RAW physical coordinates.
    String v = cmd.substring(7);
    int vals[4]; int idx = 0; int start = 0;
    for (int i = 0; i <= v.length() && idx < 4; i++) {
      if (i == v.length() || v.charAt(i) == ',') {
        vals[idx++] = v.substring(start, i).toInt(); start = i + 1;
      }
    }
    if (idx < 4) { Serial.println("ERR expected RAWROW:y,r,g,b"); return; }
    uint16_t c = dma_display->color565(vals[1], vals[2], vals[3]);
    for (int x = 0; x < PANEL_W * 2 * PANEL_CHAIN; x++)
      dma_display->drawPixel(x, vals[0], c);
    state = ST_FILL;
    Serial.println("OK");
    return;
  }

  if (cmd.startsWith("RAWSPAN:")) {
    // RAWSPAN:x0,x1,y,r,g,b - a RAW physical span, no remapping layer.
    //
    // The point of this over RAWROW: a full raw row fills every shift-register
    // position at once, so it lights several physical segments and there is no
    // way to tell which position drove which segment. Lighting one quarter at
    // a time makes that mapping directly observable, which is the one fact
    // needed to build a correct panel mapping and the one fact no amount of
    // preset-sweeping has produced.
    String v = cmd.substring(8);
    int vals[6]; int idx = 0; int start = 0;
    for (int i = 0; i <= v.length() && idx < 6; i++) {
      if (i == v.length() || v.charAt(i) == ',') {
        vals[idx++] = v.substring(start, i).toInt(); start = i + 1;
      }
    }
    if (idx < 6) { Serial.println("ERR expected RAWSPAN:x0,x1,y,r,g,b"); return; }
    uint16_t c = dma_display->color565(vals[3], vals[4], vals[5]);
    for (int x = vals[0]; x <= vals[1]; x++)
      dma_display->drawPixel(x, vals[2], c);
    state = ST_FILL;
    Serial.println("OK");
    return;
  }

  if (cmd == "RAWCLS") {
    dma_display->fillScreen(0);
    state = ST_FILL;
    Serial.println("OK");
    return;
  }

  if (cmd == "CLS") {
    display->fillScreen(C_BLACK);
    state = ST_FILL;
    fillR = fillG = fillB = 0;
    Serial.println("OK");
    return;
  }

  if (cmd == "DIAG") {
    state = ST_DIAG;
    render();
    Serial.println("OK");
    return;
  }

  if (cmd.startsWith("FILL:")) {
    // FILL:r,g,b  — each 0-255
    String v = cmd.substring(5);
    int c1 = v.indexOf(',');
    int c2 = v.indexOf(',', c1 + 1);
    if (c1 < 0 || c2 < 0) { Serial.println("ERR expected FILL:r,g,b"); return; }
    int r = v.substring(0, c1).toInt();
    int g = v.substring(c1 + 1, c2).toInt();
    int b = v.substring(c2 + 1).toInt();
    fillR = (uint8_t)constrain(r, 0, 255);
    fillG = (uint8_t)constrain(g, 0, 255);
    fillB = (uint8_t)constrain(b, 0, 255);
    state = ST_FILL;
    render();
    Serial.println("OK");
    return;
  }

  if (cmd.startsWith("BRIGHT:")) {
    int v = cmd.substring(7).toInt();
    if (v < 0)   v = 0;
    if (v > 255) v = 255;
    brightness = (uint8_t)v;
    dma_display->setBrightness8(brightness);
    render();
    Serial.println("OK");
    return;
  }

  Serial.println("ERR unknown command");
}

// ── Setup / loop ───────────────────────────────────────────────────────────

void setup() {
  Serial.begin(115200);
  Serial.setTimeout(50);

  HUB75_I2S_CFG::i2s_pins pins = {
    25, 26, 27,   // R1, G1, B1
    14, 12, 13,   // R2, G2, B2
    23, 19,  5,   // A,  B,  C
    -1,           // D  — this panel's pin 12 is NC (no D line). See
                  //      raspi_scripts/HUB75_PINOUT.md. Set to 17 if a
                  //      future panel does use D.
    -1,           // E  — only needed for 1/32-scan (64-row) panels
    16,  4, 15    // CLK, LAT, OE
  };

  // 1/8 scan: each 64-wide panel shifts 128 positions, two chained = 256 per
  // row, over 16 addressable rows. Same 4096 pixels, different shape.
  HUB75_I2S_CFG mxconfig(PANEL_W * 2, PANEL_H / 2, PANEL_CHAIN, pins);

  // If output looks shifted by a pixel or doubled, these are the two knobs
  // worth trying before anything else.
  mxconfig.clkphase = false;
  // mxconfig.driver = HUB75_I2S_CFG::FM6124;   // uncomment if the panel needs it

  dma_display = new MatrixPanel_I2S_DMA(mxconfig);
  dma_display->begin();
  dma_display->setBrightness8(brightness);
  dma_display->clearScreen();

  // The virtual panel presents the LOGICAL geometry (64x32 per panel) on top
  // of the 128x16 physical description above; the scan mapping does the
  // translation between the two.
  display = new VirtualMatrixPanel((*dma_display), 1, PANEL_CHAIN,
                                   PANEL_W, PANEL_H);
  // The remapping that makes coordinates land correctly on a 1/8-scan panel.
  // If output is still wrong, the other candidates are FOUR_SCAN_16PX_HIGH
  // and FOUR_SCAN_64PX_HIGH — these are user-contributed mappings and which
  // one fits varies by panel.
  display->setPhysicalPanelScanRate(FOUR_SCAN_32PX_HIGH);

  C_BLACK  = display->color565(0, 0, 0);
  C_WHITE  = display->color565(255, 255, 255);
  C_RED    = display->color565(229, 72, 77);
  C_DIMRED = display->color565(110, 0, 0);
  C_GREEN  = display->color565(61, 220, 151);
  C_AMBER  = display->color565(242, 179, 61);
  C_BLUE   = display->color565(91, 157, 245);

  lastCommand = millis();
  render();
  Serial.println("READY");
}

void loop() {
  // Commands are newline-terminated; anything else is ignored.
  while (Serial.available()) {
    String line = Serial.readStringUntil('\n');
    handleCommand(line);
  }

  uint32_t now = millis();

  // Flashing states need a periodic redraw.
  if ((state == ST_VEHICLE || state == ST_INCIDENT) &&
      now - lastFlash >= FLASH_INTERVAL_MS) {
    lastFlash = now;
    flashOn = !flashOn;
    render();
  }

  // Pi has gone quiet — say so rather than keep showing stale state.
  if (state != ST_OFFLINE && state != ST_BOOT &&
      now - lastCommand > COMMAND_TIMEOUT_MS) {
    state = ST_OFFLINE;
    render();
  }

  delay(10);
}
