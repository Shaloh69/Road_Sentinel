#include "mode_status.h"
#include "display.h"

namespace status_mode {

static State    cur        = ST_BOOT;
static String   line1      = "";
static String   line2      = "";
static bool     flashOn    = false;
static uint32_t lastFlash  = 0;
static uint32_t lastScroll = 0;

// Horizontal scroll offsets, in pixels, for the two text lines. Only used
// when a line is wider than the canvas; a line that fits is drawn centred and
// still, because a scrolling sign is harder to read than a static one and
// most production messages fit.
static int  off1 = 0, off2 = 0;
static bool scroll1 = false, scroll2 = false;

static void resetScroll() {
  scroll1 = display::textWidth(line1, 2) > CANVAS_W;
  scroll2 = display::textWidth(line2, 1) > CANVAS_W;
  off1 = scroll1 ? CANVAS_W : 0;    // start off the right edge
  off2 = scroll2 ? CANVAS_W : 0;
}

void setText(const String &l1, const String &l2) {
  line1 = l1;
  line2 = l2;
  resetScroll();
}

void setState(State s) {
  cur = s;
  flashOn = false;
  render();
}

State state() { return cur; }

// ── Screens ────────────────────────────────────────────────────────────────

static void renderClear() {
  // The common case: large, static, green. No flash and no scroll — an
  // ambient "nothing to worry about" reading, which is what a driver should
  // be able to take in at a glance without tracking moving text.
  display::gfx()->fillScreen(display::C_BLACK);
  display::drawCentered("ROAD CLEAR", 9, display::C_GREEN, 2);
}

static void renderVehicle() {
  // Inverted flash rather than a simple blink: reversing foreground and
  // background keeps the text legible in both phases, where blinking to black
  // leaves the sign blank half the time.
  uint16_t bg = flashOn ? display::C_AMBER : display::C_BLACK;
  uint16_t fg = flashOn ? display::C_BLACK : display::C_AMBER;
  display::gfx()->fillScreen(bg);
  display::drawCentered("VEHICLE",   2, fg, 2);
  display::drawCentered("SLOW DOWN", 21, fg, 1);
}

static void renderIncident() {
  // Red is reserved system-wide for confirmed incidents (client/web/DESIGN.md),
  // so it must not appear in any other state. Flashing between full and dim
  // red rather than to black keeps the red signal continuously present.
  uint16_t bg = flashOn ? display::C_RED : display::C_DIMRED;
  display::gfx()->fillScreen(bg);
  display::drawCentered("INCIDENT", 2, display::C_WHITE, 2);
  display::drawCentered("AHEAD - SLOW DOWN", 21, display::C_WHITE, 1);
}

static void renderOffline() {
  display::gfx()->fillScreen(display::C_BLACK);
  display::drawCentered("-- NO DATA --", 12, display::C_BLUE, 1);
}

static void renderBoot() {
  display::gfx()->fillScreen(display::C_BLACK);
  display::drawCentered("ROAD SENTINEL", 6, display::C_AMBER, 1);
  display::drawCentered("waiting for Pi", 20, display::C_BLUE, 1);
}

static void renderText() {
  display::gfx()->fillScreen(display::C_BLACK);
  if (line2.length() > 0) {
    if (scroll1) display::drawAt(line1, off1, 2, display::C_WHITE, 2);
    else         display::drawCentered(line1, 2, display::C_WHITE, 2);
    if (scroll2) display::drawAt(line2, off2, 21, display::C_AMBER, 1);
    else         display::drawCentered(line2, 21, display::C_AMBER, 1);
  } else {
    if (scroll1) display::drawAt(line1, off1, 9, display::C_WHITE, 2);
    else         display::drawCentered(line1, 9, display::C_WHITE, 2);
  }
}

void render() {
  switch (cur) {
    case ST_CLEAR:    renderClear();    break;
    case ST_VEHICLE:  renderVehicle();  break;
    case ST_INCIDENT: renderIncident(); break;
    case ST_OFFLINE:  renderOffline();  break;
    case ST_TEXT:     renderText();     break;
    default:          renderBoot();     break;
  }
}

void tick(uint32_t now) {
  // Flashing states redraw on the flash interval.
  if ((cur == ST_VEHICLE || cur == ST_INCIDENT) &&
      now - lastFlash >= FLASH_INTERVAL_MS) {
    lastFlash = now;
    flashOn = !flashOn;
    render();
    return;
  }

  // Scrolling text redraws far more often, but only while something actually
  // scrolls — a static screen costs nothing, which keeps the board responsive
  // to incoming serial commands.
  if (cur == ST_TEXT && (scroll1 || scroll2) &&
      now - lastScroll >= SCROLL_INTERVAL_MS) {
    lastScroll = now;
    if (scroll1 && --off1 < -display::textWidth(line1, 2)) off1 = CANVAS_W;
    if (scroll2 && --off2 < -display::textWidth(line2, 1)) off2 = CANVAS_W;
    render();
  }
}

} // namespace status_mode
