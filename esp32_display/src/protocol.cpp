#include "protocol.h"
#include "app.h"
#include "display.h"
#include "mode_status.h"
#include "mode_char.h"
#include "mode_sand.h"

namespace protocol {

// Line accumulator. Fixed size rather than a String so a garbled link cannot
// grow the heap without bound.
static char    buf[128];
static uint8_t len = 0;

// Split "a,b,c" into up to `max` ints. Returns how many were parsed.
static int parseInts(const String &s, int *out, int max) {
  int n = 0, start = 0;
  for (int i = 0; i <= (int)s.length() && n < max; i++) {
    if (i == (int)s.length() || s.charAt(i) == ',') {
      out[n++] = s.substring(start, i).toInt();
      start = i + 1;
    }
  }
  return n;
}

static void handle(String cmd) {
  cmd.trim();
  if (cmd.length() == 0) return;

  app::markCommand();

  // ── Health ──────────────────────────────────────────────────────────────
  if (cmd == "PING") { Serial.println("PONG"); return; }

  if (cmd == "INFO") {
    // Reports what is actually running, so "did my reflash take?" is a
    // one-second question rather than a guess. During hardware bring-up that
    // distinction matters more than it sounds.
    Serial.printf("canvas=%dx%d phys=%dx%d chain=%d driver=FM6124 "
                  "d_line=%d scan=FOUR_SCAN_32PX_HIGH bright=%d mode=%d\n",
                  CANVAS_W, CANVAS_H, PHYS_W, PHYS_H, PANEL_CHAIN,
                  PIN_D, display::brightness(), (int)app::mode());
    return;
  }

  if (cmd == "HELP" || cmd == "?") {
    Serial.println("STATE:clear|vehicle|incident|offline");
    Serial.println("TEXT:line1|line2   MODE:status|char|sand");
    Serial.println("CHAR:A   SAND:reset   SAND:rate,N");
    Serial.println("FILL:r,g,b   RECT:x,y,w,h,r,g,b   CLS   DIAG   SCAN:0-4");
    Serial.println("RAWSPAN:x0,x1,y,r,g,b   RAWCLS");
    Serial.println("BRIGHT:0-255   INFO   PING");
    return;
  }

  // ── Mode A: production status ───────────────────────────────────────────
  if (cmd.startsWith("STATE:")) {
    String v = cmd.substring(6); v.trim();
    status_mode::State s;
    if      (v == "clear")    s = status_mode::ST_CLEAR;
    else if (v == "vehicle")  s = status_mode::ST_VEHICLE;
    else if (v == "incident") s = status_mode::ST_INCIDENT;
    else if (v == "offline")  s = status_mode::ST_OFFLINE;
    else { Serial.println("ERR unknown state"); return; }
    app::setMode(app::MODE_STATUS);
    status_mode::setState(s);
    Serial.println("OK");
    return;
  }

  if (cmd.startsWith("TEXT:")) {
    String payload = cmd.substring(5);
    int bar = payload.indexOf('|');
    if (bar >= 0) status_mode::setText(payload.substring(0, bar),
                                       payload.substring(bar + 1));
    else          status_mode::setText(payload, "");
    app::setMode(app::MODE_STATUS);
    status_mode::setState(status_mode::ST_TEXT);
    Serial.println("OK");
    return;
  }

  // ── Mode switching ──────────────────────────────────────────────────────
  if (cmd.startsWith("MODE:")) {
    String v = cmd.substring(5); v.trim();
    if (v == "status") { app::setMode(app::MODE_STATUS);
                         status_mode::render(); }
    else if (v == "char") { app::setMode(app::MODE_CHAR);
                            char_mode::render(); }
    else if (v == "sand") { app::setMode(app::MODE_SAND);
                            sand::reset(); }
    else { Serial.println("ERR mode status|char|sand"); return; }
    Serial.println("OK");
    return;
  }

  // ── Mode B ──────────────────────────────────────────────────────────────
  if (cmd.startsWith("CHAR:")) {
    String v = cmd.substring(5);
    if (v.length() < 1) { Serial.println("ERR expected CHAR:<one char>"); return; }
    app::setMode(app::MODE_CHAR);
    char_mode::setChar(v.charAt(0));
    Serial.println("OK");
    return;
  }

  // ── Mode C ──────────────────────────────────────────────────────────────
  if (cmd.startsWith("SAND:")) {
    String v = cmd.substring(5); v.trim();
    if (v == "reset") {
      app::setMode(app::MODE_SAND);
      sand::reset();
      Serial.println("OK");
      return;
    }
    if (v.startsWith("rate,")) {
      sand::setRate((uint8_t)v.substring(5).toInt());
      Serial.println("OK");
      return;
    }
    Serial.println("ERR expected SAND:reset or SAND:rate,N");
    return;
  }

  // ── Diagnostics ─────────────────────────────────────────────────────────
  if (cmd.startsWith("FILL:")) {
    // A flat colour proves the LEDs, the power rail and the RGB pin order.
    // It proves NOTHING about coordinate mapping — every pixel is written
    // either way — so never read a good FILL as "the panel works".
    int v[3];
    if (parseInts(cmd.substring(5), v, 3) < 3) {
      Serial.println("ERR expected FILL:r,g,b"); return;
    }
    app::setMode(app::MODE_DIAG);
    display::gfx()->fillScreen(display::gfx()->color565(
        constrain(v[0], 0, 255), constrain(v[1], 0, 255), constrain(v[2], 0, 255)));
    Serial.println("OK");
    return;
  }

  if (cmd == "CLS") {
    app::setMode(app::MODE_DIAG);
    display::gfx()->fillScreen(display::C_BLACK);
    Serial.println("OK");
    return;
  }

  if (cmd.startsWith("RECT:")) {
    // RECT:x,y,w,h,r,g,b — one filled rectangle in LOGICAL coordinates,
    // without clearing first, so regions can be probed one at a time.
    //
    // This is the workhorse of scan-mapping calibration: a rectangle covering
    // a known fraction of the canvas either appears as that same contiguous
    // fraction on the panel, or it does not. Unlike a full-screen fill it
    // cannot look correct under a wrong mapping, and unlike text it has no
    // second way to fail.
    int v[7];
    if (parseInts(cmd.substring(5), v, 7) < 7) {
      Serial.println("ERR expected RECT:x,y,w,h,r,g,b"); return;
    }
    app::setMode(app::MODE_DIAG);
    display::gfx()->fillRect(v[0], v[1], v[2], v[3],
                             display::gfx()->color565(v[4], v[5], v[6]));
    Serial.println("OK");
    return;
  }

  if (cmd == "DIAG") {
    app::setMode(app::MODE_DIAG);
    auto *g = display::gfx();
    g->fillScreen(display::C_BLACK);
    g->fillRect(0, 0, 32, 16, display::C_RED);        // fillRect coords
    g->drawRect(96, 0, 32, 16, display::C_GREEN);     // drawRect outline
    g->drawLine(0, 0, CANVAS_W - 1, CANVAS_H - 1, display::C_BLUE);
    for (int x = 0; x < CANVAS_W; x += 4)             // bare drawPixel
      g->drawPixel(x, CANVAS_H - 1, display::C_WHITE);
    display::drawAt("A", 50, 8, display::C_WHITE, 2);
    display::drawAt("b", 50, 24, display::C_AMBER, 1);
    Serial.println("OK");
    return;
  }

  if (cmd.startsWith("SCAN:")) {
    // Runtime scan-mapping swap. Kept because this panel category is known to
    // be fiddly across firmware stacks, and a serial command costs one second
    // where a reflash costs twenty. If a value other than 2 turns out to be
    // correct, change FOUR_SCAN_32PX_HIGH in display.cpp and log why.
    int n = cmd.substring(5).toInt();
    switch (n) {
      case 0: display::gfx()->setPhysicalPanelScanRate(NORMAL_TWO_SCAN);     break;
      case 1: display::gfx()->setPhysicalPanelScanRate(NORMAL_ONE_SIXTEEN);  break;
      case 2: display::gfx()->setPhysicalPanelScanRate(FOUR_SCAN_32PX_HIGH); break;
      case 3: display::gfx()->setPhysicalPanelScanRate(FOUR_SCAN_16PX_HIGH); break;
      case 4: display::gfx()->setPhysicalPanelScanRate(FOUR_SCAN_64PX_HIGH); break;
      default: Serial.println("ERR scan 0-4"); return;
    }
    Serial.println("OK");
    return;
  }

  if (cmd.startsWith("RAWSPAN:")) {
    // A span in RAW physical coordinates (256x16), bypassing the remap layer
    // entirely. Lighting one region at a time makes the correspondence
    // between shift-register position and physical segment directly readable
    // off the panel — the one fact preset-sweeping cannot produce.
    int v[6];
    if (parseInts(cmd.substring(8), v, 6) < 6) {
      Serial.println("ERR expected RAWSPAN:x0,x1,y,r,g,b"); return;
    }
    app::setMode(app::MODE_DIAG);
    uint16_t c = display::raw()->color565(v[3], v[4], v[5]);
    for (int x = v[0]; x <= v[1]; x++) display::raw()->drawPixel(x, v[2], c);
    Serial.println("OK");
    return;
  }

  if (cmd == "RAWCLS") {
    app::setMode(app::MODE_DIAG);
    display::raw()->fillScreen(0);
    Serial.println("OK");
    return;
  }

  if (cmd.startsWith("BRIGHT:")) {
    int v = constrain(cmd.substring(7).toInt(), 0, 255);
    display::setBrightness((uint8_t)v);
    Serial.println("OK");
    return;
  }

  Serial.println("ERR unknown command");
}

void poll() {
  while (Serial.available()) {
    char c = (char)Serial.read();
    if (c == '\n' || c == '\r') {
      if (len > 0) {
        buf[len] = '\0';
        handle(String(buf));
        len = 0;
      }
      continue;
    }
    if (len < sizeof(buf) - 1) {
      buf[len++] = c;
    } else {
      // Overlong line: drop it rather than truncating into a command that
      // happens to parse. Silent truncation would be worse than a clear error.
      len = 0;
      Serial.println("ERR line too long");
    }
  }
}

} // namespace protocol
