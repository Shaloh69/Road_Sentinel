#pragma once
/*
 * The measured panel mapping. Portable — pure arithmetic, identical on every
 * board, because it describes the PANEL and not the microcontroller.
 *
 * ─────────────────────────────────────────────────────────────────────────
 * HOW THIS WAS DETERMINED
 *
 * Not from a datasheet, and not by choosing among a library's presets. Two
 * bit-banged measurements on the physical hardware produced every number
 * here, after this project twice reached a confident WRONG conclusion by
 * inferring the scan structure instead of measuring it.
 *
 * 1. Walking A/B/C through addresses 0..7 with an all-on row:
 *      - 8 distinct positions => all three address lines work: 1/8 scan
 *      - each address lights FOUR rows, spaced 8 apart
 *      - the group moves BOTTOM -> TOP as the address rises, so address 0 is
 *        the bottom row: the row order is inverted versus every convention
 *      => address a lights rows { 7-a, 15-a, 23-a, 31-a }
 *
 * 2. Colouring each quarter of the 256-position register and photographing it:
 *      - positions   0-63  -> one panel, upper of its two lines
 *      - positions  64-127 -> same panel, lower line, 8 rows down
 *      - positions 128-191 -> other panel, upper line
 *      - positions 192-255 -> other panel, lower line
 *      => each panel owns 128 consecutive positions, split 64 upper / 64 lower
 *      - driving R2/G2/B2 rather than R1/G1/B1 gave the same picture 16 rows
 *        away => the two channels feed the two 16-row halves of the canvas
 *
 * Orientation (SWAP_PANELS / FLIP_X / FLIP_Y / SWAP_HALVES in panel_config.h)
 * was then settled by reading the sign, one flag at a time. See DEBUG_LOG.md.
 * ─────────────────────────────────────────────────────────────────────────
 *
 * The tables are built once at startup rather than computed in the refresh
 * loop: that loop runs ~2000 times per frame and any arithmetic inside it
 * costs brightness directly.
 */

#include <Arduino.h>
#include "panel_config.h"

namespace panelmap {

// For each address and clock position, the framebuffer index feeding each of
// the two data channels.
extern uint16_t ch1[SCAN_ADDRESSES][REGISTER_LEN];
extern uint16_t ch2[SCAN_ADDRESSES][REGISTER_LEN];

void build();

} // namespace panelmap
