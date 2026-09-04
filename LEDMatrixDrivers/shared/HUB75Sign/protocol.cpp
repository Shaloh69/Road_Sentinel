#include "protocol.h"
#include "app.h"
#include "display.h"
#include "mode_status.h"
#include "mode_char.h"
#include "mode_sand.h"
#include "hub75.h"

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
    Serial.printf("canvas=%dx%d addresses=%d reglen=%d driver=own-bitbang "
                  "fm6124=yes d_line=none bright=%d fps=%lu mode=%d\n",
                  CANVAS_W, CANVAS_H, SCAN_ADDRESSES, REGISTER_LEN,
                  display::brightness(), (unsigned long)hub75::framesPerSecond(),
                  (int)app::mode());
    return;
  }

  if (cmd == "HELP" || cmd == "?") {
    Serial.println("STATE:clear|vehicle|speeding|crash|stopped|congestion|incident|offline");
    Serial.println("TEXT:line1|line2   MODE:status|char|sand");
    Serial.println("CHAR:A   SAND:reset   SAND:rate,N");
    Serial.println("FILL:c   RECT:x,y,w,h,c   CLS   DIAG   (c = 0-7)");
    Serial.println("BRIGHT:0-255   INFO   PING");
    return;
  }

  // ── Mode A: production status ───────────────────────────────────────────
  if (cmd.startsWith("STATE:")) {
    String v = cmd.substring(6); v.trim();
    status_mode::State s;
    // The vocabulary mirrors the server's incidents.incident_type ENUM, so
    // the Pi never has to decide what a given incident should look like — it
    // forwards what happened and the sign owns the presentation.
    if      (v == "clear")      s = status_mode::ST_SAFE;
    else if (v == "vehicle")    s = status_mode::ST_VEHICLE;
    else if (v == "speeding")   s = status_mode::ST_SPEEDING;
    else if (v == "crash")      s = status_mode::ST_CRASH;
    else if (v == "stopped")    s = status_mode::ST_STOPPED;
    else if (v == "congestion") s = status_mode::ST_CONGESTION;
    else if (v == "incident")   s = status_mode::ST_STOP;
    else if (v == "offline")    s = status_mode::ST_OFFLINE;
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
    // FILL:c  where c is 0-7: 0 off, 1 blue, 2 green, 3 cyan,
    //                        4 red, 5 magenta, 6 yellow, 7 white.
    //
    // A flat fill proves the LEDs, the power rail and the RGB pin order. It
    // proves NOTHING about coordinate mapping — every pixel is written either
    // way — so never read a good FILL as "the panel works". Several hardware
    // sessions were lost to exactly that misreading.
    int c = cmd.substring(5).toInt();
    if (c < 0 || c > 7) { Serial.println("ERR colour 0-7"); return; }
    app::setMode(app::MODE_DIAG);
    display::gfx()->fillScreen((uint16_t)c);
    Serial.println("OK");
    return;
  }

  if (cmd == "CLS") {
    app::setMode(app::MODE_DIAG);
    display::gfx()->fillScreen(HC_OFF);
    Serial.println("OK");
    return;
  }

  if (cmd.startsWith("RECT:")) {
    // RECT:x,y,w,h,c — one filled rectangle in canvas coordinates, drawn
    // without clearing first so regions can be probed one at a time.
    //
    // Unlike a full-screen fill this cannot look right under a wrong mapping,
    // and unlike text it has only one way to fail — which makes it the useful
    // diagnostic of the three.
    int v[5];
    if (parseInts(cmd.substring(5), v, 5) < 5) {
      Serial.println("ERR expected RECT:x,y,w,h,c"); return;
    }
    app::setMode(app::MODE_DIAG);
    display::gfx()->fillRect(v[0], v[1], v[2], v[3], (uint16_t)(v[4] & 0x7));
    Serial.println("OK");
    return;
  }

  if (cmd == "DIAG") {
    app::setMode(app::MODE_DIAG);
    auto *g = display::gfx();
    g->fillScreen(HC_OFF);
    g->fillRect(0, 0, 32, 16, HC_RED);                       // fillRect coords
    g->drawRect(96, 0, 32, 16, HC_GREEN);                    // drawRect outline
    g->drawLine(0, 0, CANVAS_W - 1, CANVAS_H - 1, HC_BLUE);  // drawLine
    for (int x = 0; x < CANVAS_W; x += 4)                    // bare drawPixel
      g->drawPixel(x, CANVAS_H - 1, HC_WHITE);
    display::drawAt("A", 50, 8,  HC_WHITE,  2);              // scaled glyph
    display::drawAt("b", 50, 24, HC_YELLOW, 1);              // base glyph
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
