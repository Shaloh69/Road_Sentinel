#pragma once
/*
 * Mode C — falling-sand pixel simulation across the full 128x32 canvas.
 *
 * This is the panel bring-up test, not decoration, and it is deliberately a
 * better test than a static pattern:
 *
 *   - Every grain is one individually addressable pixel, so it exercises the
 *     coordinate mapping at single-pixel resolution. A solid fill writes the
 *     whole framebuffer and therefore looks perfect under EVERY scan mapping,
 *     correct or not — it is the least informative test possible here, and
 *     several earlier hardware sessions were spent on exactly that mistake.
 *   - The motion is predictable, so a wrong mapping is legible rather than
 *     merely "wrong-looking". Grains must fall straight DOWN and pile up from
 *     the bottom. If they drift sideways, jump between the two panels, fall
 *     upward, or pile in mid-air, the mapping is broken — and HOW it breaks
 *     names the fault:
 *
 *       grains fall correctly on panel 1, panel 2 dark   -> chain length
 *       grains teleport half-canvas sideways             -> PHYS_W folding
 *       two grains appear per grain, 16 rows apart       -> address lines
 *       grains fall in 4-row bands that do not connect   -> scan mapping
 *
 * That mapping from symptom to cause is the whole reason this mode exists.
 */

#include <Arduino.h>

namespace sand {
void reset();
void step();                 // advance one frame; call on the millis() tick
void setRate(uint8_t grains_per_frame);
uint16_t grainCount();

// True once the one-shot sequence has finished (build -> hold -> collapse).
// The caller hands the panel back to status duty on this, so the animation
// cannot outlive itself on a roadside sign.
bool isDone();
}
