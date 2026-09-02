#include "mode_sand.h"
#include "display.h"

namespace sand {

// One byte per pixel: 0 = empty, otherwise a 1-based index into PALETTE.
// 128 x 32 = 4096 bytes, which is cheap on an ESP32 and much faster to
// simulate than reading pixels back out of the DMA framebuffer.
static uint8_t grid[CANVAS_H][CANVAS_W];

static const uint8_t PALETTE_N = 4;
static uint16_t      palette[PALETTE_N];
static bool          paletteReady = false;

static uint8_t  emitRate  = 3;    // grains added per frame
static uint16_t grains    = 0;
static uint16_t idleFrames = 0;   // frames with nothing moving

// Once the pile stops moving there is nothing left to observe, so start over
// rather than leaving a static heap that looks like a crash.
static const uint16_t IDLE_FRAMES_BEFORE_RESET = 60;   // ~2s at 30fps

static void ensurePalette() {
  if (paletteReady) return;
  // Night Watch tokens. Four distinct hues also make this a colour test:
  // if red and blue are swapped the RGB pin order is wrong, which is a
  // different fault from a mapping error and worth telling apart at a glance.
  palette[0] = display::C_AMBER;
  palette[1] = display::C_GREEN;
  palette[2] = display::C_BLUE;
  palette[3] = display::C_WHITE;
  paletteReady = true;
}

static inline void put(int x, int y, uint8_t colourIdx) {
  grid[y][x] = colourIdx;
  display::gfx()->drawPixel(x, y, palette[colourIdx - 1]);
}

static inline void clearCell(int x, int y) {
  grid[y][x] = 0;
  display::gfx()->drawPixel(x, y, display::C_BLACK);
}

void reset() {
  ensurePalette();
  memset(grid, 0, sizeof(grid));
  display::gfx()->fillScreen(display::C_BLACK);
  grains = 0;
  idleFrames = 0;
}

void setRate(uint8_t r) { emitRate = constrain(r, 1, 16); }
uint16_t grainCount()   { return grains; }

void step() {
  ensurePalette();
  bool moved = false;

  // ── Emit ────────────────────────────────────────────────────────────────
  for (uint8_t i = 0; i < emitRate; i++) {
    int x = random(CANVAS_W);
    if (grid[0][x] == 0) {
      put(x, 0, (uint8_t)random(1, PALETTE_N + 1));
      grains++;
      moved = true;
    }
  }

  // ── Physics ─────────────────────────────────────────────────────────────
  // Scan bottom-up so a grain that falls into an already-processed row is not
  // moved twice in one frame — that bug makes sand appear to teleport to the
  // floor instantly, which would mask exactly the mapping faults this mode
  // exists to reveal.
  for (int y = CANVAS_H - 2; y >= 0; y--) {
    // Alternate the horizontal scan direction each row. A fixed direction
    // biases diagonal slides one way and the pile leans, which reads as a
    // mapping problem when it is only an artefact of the loop order.
    bool leftToRight = (y & 1) == 0;
    for (int i = 0; i < CANVAS_W; i++) {
      int x = leftToRight ? i : (CANVAS_W - 1 - i);
      uint8_t c = grid[y][x];
      if (c == 0) continue;

      if (grid[y + 1][x] == 0) {                 // straight down
        clearCell(x, y);
        put(x, y + 1, c);
        moved = true;
        continue;
      }

      // Blocked below: try to slide diagonally. Random preference so the pile
      // is symmetric rather than always shedding to one side.
      int first  = random(2) ? -1 : 1;
      int second = -first;
      for (int t = 0; t < 2; t++) {
        int dx = (t == 0) ? first : second;
        int nx = x + dx;
        if (nx < 0 || nx >= CANVAS_W) continue;
        if (grid[y + 1][nx] != 0) continue;
        clearCell(x, y);
        put(nx, y + 1, c);
        moved = true;
        break;
      }
    }
  }

  idleFrames = moved ? 0 : (idleFrames + 1);
  if (idleFrames >= IDLE_FRAMES_BEFORE_RESET) reset();
}

} // namespace sand
