#include "mode_status.h"
#include "display.h"

namespace status_mode {

static State    cur        = ST_BOOT;
static String   line1      = "";
static String   line2      = "";
static bool     flashOn    = false;
static uint32_t lastFlash  = 0;
static uint32_t lastScroll = 0;

static int  off1 = 0, off2 = 0;
static bool scroll1 = false, scroll2 = false;

static void resetScroll() {
  scroll1 = display::textWidth(line1, 2) > CANVAS_W;
  scroll2 = display::textWidth(line2, 1) > CANVAS_W;
  off1 = scroll1 ? CANVAS_W : 0;
  off2 = scroll2 ? CANVAS_W : 0;
}

void setText(const String &l1, const String &l2) {
  line1 = l1; line2 = l2;
  resetScroll();
}

void setState(State s) {
  cur = s;
  flashOn = false;
  render();
}

State state() { return cur; }

// ── Screens ────────────────────────────────────────────────────────────────
//
// Sizing is worked out against the real 128x32 canvas rather than guessed. The
// built-in GFX font is a 5x7 glyph in a 6x8 cell, so a character at size N is
// 6N wide and 8N tall:
//
//   "SAFE"             size 3 ->  4 x 18 =  72px wide, 24 tall
//   "STOP"             size 4 ->  4 x 24 =  96px wide, 32 tall
//   "SLOW DOWN"        size 2 ->  9 x 12 = 108px wide, 16 tall
//   "VEHICLE INCOMING" size 1 -> 16 x  6 =  96px wide,  8 tall
//
// All fit inside 128 wide, so nothing needs to scroll — which matters,
// because static text is markedly easier to read from a moving vehicle than
// text that slides.

static void renderSafe() {
  // The common case. Large, green, static, no flash — it should read as
  // ambient reassurance at a glance, not compete for attention.
  display::gfx()->fillScreen(HC_OFF);
  display::drawCentered("SAFE", 4, HC_GREEN, 3);
}

static void renderVehicle() {
  // Inverted flash rather than blinking to black: reversing foreground and
  // background keeps the words legible in BOTH phases, where blinking off
  // leaves the sign blank half the time — exactly when a driver may look.
  uint16_t bg = flashOn ? HC_YELLOW : HC_OFF;
  uint16_t fg = flashOn ? HC_OFF    : HC_YELLOW;
  display::gfx()->fillScreen(bg);
  display::drawCentered("VEHICLE INCOMING", 2, fg, 1);
  display::drawCentered("SLOW DOWN",       14, fg, 2);
}

// Two lines of size 2, stacked to fill the 32px canvas exactly (16px each).
// Every phrase below was checked against the 128px width at size 2: the GFX
// font is 6px per character per size step, so the longest word used here
// ("STOPPED", 7 chars) is 84px and clears it comfortably.
static void renderTwoLine(const char *top, const char *bottom,
                          uint16_t colour, bool invert) {
  uint16_t bg = invert ? colour : HC_OFF;
  uint16_t fg = invert ? HC_OFF : colour;

  display::gfx()->fillScreen(bg);
  display::drawCentered(top, 1, fg, 2);
  display::drawCentered(bottom, 17, fg, 2);
}

static void renderSpeeding() {
  // Single hazard on THIS approach: tell the driver in front of the sign to
  // slow down. No "vehicle incoming" — nothing is coming the other way.
  uint16_t bg = flashOn ? HC_YELLOW : HC_OFF;
  uint16_t fg = flashOn ? HC_OFF : HC_YELLOW;

  display::gfx()->fillScreen(bg);
  display::drawCentered("SLOW", 1, fg, 2);
  display::drawCentered("DOWN", 17, fg, 2);
}

static void renderCrash() {
  renderTwoLine("CRASH", "AHEAD", HC_RED, flashOn);
}

static void renderStopped() {
  renderTwoLine("STOPPED", "VEHICLE", HC_RED, flashOn);
}

static void renderCongestion() {
  renderTwoLine("TRAFFIC", "AHEAD", HC_YELLOW, flashOn);
}

static void renderStop() {
  // Red is reserved system-wide for confirmed incidents, so it appears in no
  // other state. Flashing red-on-black at the largest size the panel can hold.
  uint16_t bg = flashOn ? HC_RED : HC_OFF;
  uint16_t fg = flashOn ? HC_OFF : HC_RED;
  display::gfx()->fillScreen(bg);
  display::drawCentered("STOP", 2, fg, 4);
}

static void renderOffline() {
  display::gfx()->fillScreen(HC_OFF);
  display::drawCentered("NO DATA", 12, HC_BLUE, 1);
}

static void renderBoot() {
  display::gfx()->fillScreen(HC_OFF);
  display::drawCentered("ROAD SENTINEL", 6, HC_YELLOW, 1);
  display::drawCentered("waiting for Pi", 20, HC_BLUE, 1);
}

static void renderText() {
  display::gfx()->fillScreen(HC_OFF);
  if (line2.length() > 0) {
    if (scroll1) display::drawAt(line1, off1, 2, HC_WHITE, 2);
    else         display::drawCentered(line1, 2, HC_WHITE, 2);
    if (scroll2) display::drawAt(line2, off2, 21, HC_YELLOW, 1);
    else         display::drawCentered(line2, 21, HC_YELLOW, 1);
  } else {
    if (scroll1) display::drawAt(line1, off1, 9, HC_WHITE, 2);
    else         display::drawCentered(line1, 9, HC_WHITE, 2);
  }
}

void render() {
  switch (cur) {
    case ST_SAFE:       renderSafe();       break;
    case ST_VEHICLE:    renderVehicle();    break;
    case ST_SPEEDING:   renderSpeeding();   break;
    case ST_CRASH:      renderCrash();      break;
    case ST_STOPPED:    renderStopped();    break;
    case ST_CONGESTION: renderCongestion(); break;
    case ST_STOP:       renderStop();       break;
    case ST_OFFLINE: renderOffline(); break;
    case ST_TEXT:    renderText();    break;
    default:         renderBoot();    break;
  }
}

void tick(uint32_t now) {
  // Every alerting state flashes; only SAFE and the informational screens are
  // static. Listed explicitly rather than "not SAFE" so adding a calm state
  // later cannot make it blink by accident.
  if ((cur == ST_VEHICLE || cur == ST_STOP || cur == ST_SPEEDING ||
       cur == ST_CRASH || cur == ST_STOPPED || cur == ST_CONGESTION) &&
      now - lastFlash >= FLASH_INTERVAL_MS) {
    lastFlash = now;
    flashOn = !flashOn;
    render();
    return;
  }

  // Only redraws while something actually scrolls; a static screen costs
  // nothing, which keeps the board responsive to incoming commands.
  if (cur == ST_TEXT && (scroll1 || scroll2) &&
      now - lastScroll >= SCROLL_INTERVAL_MS) {
    lastScroll = now;
    if (scroll1 && --off1 < -display::textWidth(line1, 2)) off1 = CANVAS_W;
    if (scroll2 && --off2 < -display::textWidth(line2, 1)) off2 = CANVAS_W;
    render();
  }
}

} // namespace status_mode
