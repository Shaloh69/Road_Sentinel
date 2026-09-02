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
 *   BRIGHT:0-255           panel brightness
 *   PING                   -> replies PONG, for the Pi's health check
 *
 * Every command is acknowledged with "OK" or "ERR <reason>" so the Pi can tell
 * the difference between "board is wedged" and "board disagreed with me".
 */

#include <ESP32-HUB75-MatrixPanel-I2S-DMA.h>

// ── Panel geometry ─────────────────────────────────────────────────────────
// Counted on the hardware: 64 wide x 32 tall per panel, two chained -> 128x32.
#define PANEL_W     64
#define PANEL_H     32
#define PANEL_CHAIN 2

MatrixPanel_I2S_DMA *dma_display = nullptr;

// ── Colours ────────────────────────────────────────────────────────────────
uint16_t C_BLACK, C_WHITE, C_RED, C_GREEN, C_AMBER, C_BLUE, C_DIMRED;

// ── State ──────────────────────────────────────────────────────────────────
enum DisplayState { ST_BOOT, ST_CLEAR, ST_VEHICLE, ST_INCIDENT, ST_OFFLINE, ST_TEXT };
DisplayState state = ST_BOOT;

String  textLine1 = "";
String  textLine2 = "";
uint8_t brightness = 90;

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
  dma_display->setTextSize(size);
  dma_display->setTextColor(colour);
  int16_t x1, y1;
  uint16_t w, h;
  dma_display->getTextBounds(s, 0, 0, &x1, &y1, &w, &h);
  int x = ((PANEL_W * PANEL_CHAIN) - (int)w) / 2 - x1;
  if (x < 0) x = 0;
  dma_display->setCursor(x, y);
  dma_display->print(s);
}

void drawTwoLines(const String &big, const String &small,
                  uint16_t bg, uint16_t fgBig, uint16_t fgSmall) {
  dma_display->fillScreen(bg);
  drawCentered(big, 2, fgBig, 2);
  drawCentered(small, 20, fgSmall, 1);
}

// ── Screens ────────────────────────────────────────────────────────────────

void renderClear() {
  dma_display->fillScreen(C_BLACK);
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
  dma_display->fillScreen(C_BLACK);
  drawCentered("-- NO DATA --", 12, C_BLUE, 1);
}

void renderText() {
  dma_display->fillScreen(C_BLACK);
  if (textLine2.length() > 0) {
    drawCentered(textLine1, 2, C_WHITE, 2);
    drawCentered(textLine2, 20, C_AMBER, 1);
  } else {
    drawCentered(textLine1, 12, C_WHITE, 2);
  }
}

void renderBoot() {
  dma_display->fillScreen(C_BLACK);
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

  HUB75_I2S_CFG mxconfig(PANEL_W, PANEL_H, PANEL_CHAIN, pins);

  // If output looks shifted by a pixel or doubled, these are the two knobs
  // worth trying before anything else.
  mxconfig.clkphase = false;
  // mxconfig.driver = HUB75_I2S_CFG::FM6124;   // uncomment if the panel needs it

  dma_display = new MatrixPanel_I2S_DMA(mxconfig);
  dma_display->begin();
  dma_display->setBrightness8(brightness);
  dma_display->clearScreen();

  C_BLACK  = dma_display->color565(0, 0, 0);
  C_WHITE  = dma_display->color565(255, 255, 255);
  C_RED    = dma_display->color565(229, 72, 77);
  C_DIMRED = dma_display->color565(110, 0, 0);
  C_GREEN  = dma_display->color565(61, 220, 151);
  C_AMBER  = dma_display->color565(242, 179, 61);
  C_BLUE   = dma_display->color565(91, 157, 245);

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
