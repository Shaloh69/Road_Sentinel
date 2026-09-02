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
// Grains do not pile at random: each one is aimed at a cell of a shape, so the
// pile resolves into that shape as it fills. Targets are filled bottom-up per
// column, which is what lets a grain fall straight down without passing
// through one that has already settled.
#define MAX_PER_COL 20
static uint8_t colTargets[CANVAS_W][MAX_PER_COL];  // rows, bottom-most first
static uint8_t colCount[CANVAS_W];                 // how many targets exist
static uint8_t colTaken[CANVAS_W];                 // how many are spoken for

// ── Grains in flight ───────────────────────────────────────────────────────
#define MAX_ACTIVE 22
struct Grain { int16_t x, y, targetY; uint8_t colour; bool live; };
static Grain active[MAX_ACTIVE];

static uint8_t  emitRate   = 3;
static uint16_t grains     = 0;
static uint16_t holdFrames = 0;

// Once the shape is complete there is nothing left to watch, so hold it for a
// couple of seconds and start over rather than freezing on a static image
// that reads as a crash.
static const uint16_t HOLD_FRAMES = 70;   // ~2.3s at 30fps

static void ensurePalette() {
  if (paletteReady) return;
  // Four clearly distinct hues. This doubles as a colour check: if red and
  // blue look swapped that is an RGB pin-order fault, which is a different
  // problem from a mapping fault and worth telling apart at a glance.
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
// font is the single source of glyph data rather than a second hand-made copy.
static void buildTargets() {
  memset(colTargets, 0, sizeof(colTargets));
  memset(colCount,   0, sizeof(colCount));
  memset(colTaken,   0, sizeof(colTaken));

  display::gfx()->fillScreen(HC_OFF);
  display::drawCentered(SAND_SHAPE_TEXT, (CANVAS_H - 8 * SAND_SHAPE_SIZE) / 2,
                        HC_WHITE, SAND_SHAPE_SIZE);

  // Walk each column from the bottom up, so colTargets[x][0] is the lowest
  // cell. Filling in that order means a falling grain always stops above the
  // grains already settled beneath it.
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
  buildTargets();
}

void setRate(uint8_t r) { emitRate = constrain(r, 1, 16); }
uint16_t grainCount()   { return grains; }

// Pick a column that still has an unclaimed target. Random rather than
// left-to-right so the shape fills in unevenly, which reads as falling sand
// instead of a progress bar.
static int pickColumn() {
  for (int tries = 0; tries < 24; tries++) {
    int x = random(CANVAS_W);
    if (colTaken[x] < colCount[x]) return x;
  }
  for (int x = 0; x < CANVAS_W; x++)
    if (colTaken[x] < colCount[x]) return x;
  return -1;                                   // shape complete
}

void step() {
  ensurePalette();

  // Shape finished and everything has landed: hold, then start again.
  bool anyLive = false;
  for (int i = 0; i < MAX_ACTIVE; i++) if (active[i].live) { anyLive = true; break; }
  if (!anyLive && pickColumn() < 0) {
    if (++holdFrames >= HOLD_FRAMES) reset();
    return;
  }

  // ── Emit ────────────────────────────────────────────────────────────────
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

  // ── Fall ────────────────────────────────────────────────────────────────
  // One row per frame, drawn as it goes, so grains are visibly individual
  // pixels in motion. That is what makes this a per-pixel test and not just
  // decoration: if any pixel landed somewhere other than asked, the fall would
  // not read as a straight vertical line.
  for (int i = 0; i < MAX_ACTIVE; i++) {
    Grain &g = active[i];
    if (!g.live) continue;

    if (g.y > 0) erase(g.x, g.y - 1 < 0 ? 0 : g.y);
    if (grid[g.y][g.x] == 0) erase(g.x, g.y);

    if (g.y >= g.targetY) {
      put(g.x, g.y, g.colour);
      g.live = false;
      grains++;
      continue;
    }

    g.y++;
    if (grid[g.y][g.x] == 0)
      display::gfx()->drawPixel(g.x, g.y, palette[g.colour - 1]);
  }
}

} // namespace sand
