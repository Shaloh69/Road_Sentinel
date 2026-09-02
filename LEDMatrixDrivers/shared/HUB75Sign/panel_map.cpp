#include "panel_map.h"

namespace panelmap {

uint16_t ch1[SCAN_ADDRESSES][REGISTER_LEN];
uint16_t ch2[SCAN_ADDRESSES][REGISTER_LEN];

void build() {
  for (int a = 0; a < SCAN_ADDRESSES; a++) {
    for (int p = 0; p < REGISTER_LEN; p++) {

      // Which panel the first-clocked half of the register drives. Measured
      // from where centred text landed, not assumed: with SWAP_PANELS = 1,
      // positions 0..127 feed the LEFT panel.
      int q, xBase;
      if (p < PANEL_W * 2) { q = p;               xBase = SWAP_PANELS ? 0 : PANEL_W; }
      else                 { q = p - PANEL_W * 2; xBase = SWAP_PANELS ? PANEL_W : 0; }

      int subRow = q / PANEL_W;    // 0 = upper line, 1 = the line 8 rows below
      int col    = q % PANEL_W;

      int lx  = xBase + col;
      int ly1 = (7  - a) + subRow * 8;   // upper half rows:  7-a and 15-a
      int ly2 = (23 - a) + subRow * 8;   // lower half rows: 23-a and 31-a

      // MEASURED: R1/G1/B1 drives the LOWER 16 rows on this panel, not the
      // upper. Observed as every glyph having its top half on the bottom.
      // Kept as a flag rather than folded into the formulas above so the scan
      // structure and the channel assignment stay separately checkable.
      if (SWAP_HALVES) { int tmp = ly1; ly1 = ly2; ly2 = tmp; }

      // Orientation last, so it is independent of the scan mapping and can be
      // changed if the sign is ever remounted.
      if (FLIP_X) lx = CANVAS_W - 1 - lx;
      if (FLIP_Y) { ly1 = CANVAS_H - 1 - ly1; ly2 = CANVAS_H - 1 - ly2; }

      ch1[a][p] = (uint16_t)(ly1 * CANVAS_W + lx);
      ch2[a][p] = (uint16_t)(ly2 * CANVAS_W + lx);
    }
  }
}

} // namespace panelmap
