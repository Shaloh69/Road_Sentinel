#include "mode_sand.h"
#include "display.h"

namespace sand {

// ── Settled grains ─────────────────────────────────────────────────────────
// One byte per cell holding a 1-based palette index, 0 for empty.
static uint8_t grid[CANVAS_H][CANVAS_W];

static const uint8_t PALETTE_N = 4;
static uint16_t      palette[PALETTE_N];
static bool          paletteReady = false;

// ── Target shape ───────────────────────────────────────────────────────────
// Grains are aimed at cells of a shape rather than piling at random, so the
// heap resolves into that shape as it fills. Targets are filled bottom-up per
// column, which is what lets a grain fall straight down without passing
// through one that has already settled.
#define MAX_PER_COL 20
static uint8_t colTargets[CANVAS_W][MAX_PER_COL];  // rows, bottom-most first
static uint8_t colCount[CANVAS_W];
static uint8_t colTaken[CANVAS_W];

// ── Grains in flight ───────────────────────────────────────────────────────
#define MAX_ACTIVE 24
struct Grain { int16_t x, y, targetY; uint8_t colour; bool live; };
static Grain active[MAX_ACTIVE];

static uint8_t  emitRate = 3;
static uint16_t grains   = 0;

// ── Phases ─────────────────────────────────────────────────────────────────
// The animation is a one-shot sequence, not a loop. It builds the shape, holds
// it long enough to read, tears it down grain by grain, and then reports DONE
// so the caller can hand the panel back to its normal status duty.
//
// One-shot matters: this runs on a roadside sign. An attract animation that
// looped forever would need something else to remember to stop it, and the
// thing most likely to be forgotten is the thing that returns a safety device
// to doing its job.
enum Phase { P_BUILD, P_HOLD, P_COLLAPSE, P_DONE };
static Phase    phase      = P_BUILD;
static uint16_t holdFrames = 0;

// 3 seconds at SAND_INTERVAL_MS (~30fps).
static const uint16_t HOLD_FRAMES = 3000 / SAND_INTERVAL_MS;

// Grains released per frame during teardown. One gives a distinctly
// one-at-a-time cascade rather than the shape simply dropping out.
static const uint8_t COLLAPSE_RATE = 1;

static void ensurePalette() {
  if (paletteReady) return;
  // Four clearly distinct hues. Doubles as a colour check: red and blue
  // appearing swapped is an RGB pin-order fault, which is a different problem
  // from a mapping fault and worth telling apart at a glance.
  palette[0] = HC_YELLOW;
  palette[1] = HC_GREEN;
  palette[2] = HC_BLUE;
  palette[3] = HC_WHITE;
  paletteReady = true;
}

static inline void put(int x, int y, uint8_t idx) {
  grid[y][x] = idx;
  display::gfx()->drawPixel(x, y, palette[idx - 1]);
}

static inline void erase(int x, int y) {
  display::gfx()->drawPixel(x, y, HC_OFF);
}

// Render the target shape once and read it back out of the framebuffer, so the
// font stays the single source of glyph data rather than a second hand-made
// copy that could drift from it.
static void buildTargets() {
  memset(colTargets, 0, sizeof(colTargets));
  memset(colCount,   0, sizeof(colCount));
  memset(colTaken,   0, sizeof(colTaken));

  display::gfx()->fillScreen(HC_OFF);
  display::drawCentered(SAND_SHAPE_TEXT, (CANVAS_H - 8 * SAND_SHAPE_SIZE) / 2,
                        HC_WHITE, SAND_SHAPE_SIZE);

  // Walk each column bottom-up so colTargets[x][0] is the lowest cell. Filling
  // in that order means a falling grain always stops above the grains already
  // settled beneath it.
  for (int x = 0; x < CANVAS_W; x++) {
    for (int y = CANVAS_H - 1; y >= 0; y--) {
      if (hub75::getPixel(x, y) && colCount[x] < MAX_PER_COL)
        colTargets[x][colCount[x]++] = (uint8_t)y;
    }
  }
  display::gfx()->fillScreen(HC_OFF);
}

void reset() {
  ensurePalette();
  memset(grid, 0, sizeof(grid));
  for (int i = 0; i < MAX_ACTIVE; i++) active[i].live = false;
  grains = 0;
  holdFrames = 0;
  phase = P_BUILD;
  buildTargets();
}

void setRate(uint8_t r) { emitRate = constrain(r, 1, 16); }
uint16_t grainCount()   { return grains; }
bool     isDone()       { return phase == P_DONE; }

// Pick a column that still has an unclaimed target. Random rather than
// left-to-right so the shape fills unevenly, which reads as falling sand
// instead of a progress bar.
static int pickColumn() {
  for (int tries = 0; tries < 24; tries++) {
    int x = random(CANVAS_W);
    if (colTaken[x] < colCount[x]) return x;
  }
  for (int x = 0; x < CANVAS_W; x++)
    if (colTaken[x] < colCount[x]) return x;
  return -1;
}

static bool anyLive() {
  for (int i = 0; i < MAX_ACTIVE; i++) if (active[i].live) return true;
  return false;
}

// Detach one settled grain and let it fall away. Chosen from the BOTTOM up so
// the shape erodes from underneath and the rest appears to drop after it,
// rather than dissolving from the top down which reads as pixels simply
// switching off.
static bool releaseOne() {
  for (int y = CANVAS_H - 1; y >= 0; y--) {
    for (int pass = 0; pass < CANVAS_W; pass++) {
      int x = random(CANVAS_W);
      if (grid[y][x] == 0) continue;
      int slot = -1;
      for (int i = 0; i < MAX_ACTIVE; i++) if (!active[i].live) { slot = i; break; }
      if (slot < 0) return false;                 // no room; try next frame
      // targetY beyond the canvas means "fall off the bottom and vanish".
      active[slot] = { (int16_t)x, (int16_t)y, (int16_t)CANVAS_H,
                       grid[y][x], true };
      grid[y][x] = 0;
      erase(x, y);
      return true;
    }
  }
  return false;                                    // nothing left to release
}

void step() {
  ensurePalette();

  switch (phase) {
    case P_BUILD:
      if (!anyLive() && pickColumn() < 0) { phase = P_HOLD; holdFrames = 0; }
      break;

    case P_HOLD:
      // Shape complete and readable. Nothing moves for three seconds.
      if (++holdFrames >= HOLD_FRAMES) phase = P_COLLAPSE;
      return;

    case P_COLLAPSE:
      for (uint8_t n = 0; n < COLLAPSE_RATE; n++) releaseOne();
      if (!anyLive() && grains == 0) { phase = P_DONE; return; }
      break;

    case P_DONE:
      return;
  }

  // ── Emit (build phase only) ─────────────────────────────────────────────
  if (phase == P_BUILD) {
    for (uint8_t n = 0; n < emitRate; n++) {
      int slot = -1;
      for (int i = 0; i < MAX_ACTIVE; i++) if (!active[i].live) { slot = i; break; }
      if (slot < 0) break;
      int x = pickColumn();
      if (x < 0) break;
      active[slot] = { (int16_t)x, 0, (int16_t)colTargets[x][colTaken[x]],
                       (uint8_t)random(1, PALETTE_N + 1), true };
      colTaken[x]++;
    }
  }

  // ── Fall ────────────────────────────────────────────────────────────────
  // One row per frame, drawn as it moves, so grains are visibly individual
  // pixels in motion. That is what makes this a per-pixel test rather than
  // decoration: a pixel landing anywhere other than asked would break the
  // straight vertical line of the fall.
  for (int i = 0; i < MAX_ACTIVE; i++) {
    Grain &g = active[i];
    if (!g.live) continue;

    erase(g.x, g.y);
    g.y++;

    if (g.y >= CANVAS_H) {          // fell off the bottom (collapse phase)
      g.live = false;
      if (grains) grains--;
      continue;
    }
    if (g.y >= g.targetY) {         // reached its place in the shape
      put(g.x, g.y, g.colour);
      g.live = false;
      grains++;
      continue;
    }
    display::gfx()->drawPixel(g.x, g.y, palette[g.colour - 1]);
  }
}

} // namespace sand
