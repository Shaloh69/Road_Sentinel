#pragma once
/*
 * Mode B — single large character, for hardware bring-up.
 *
 * Deliberately the simplest thing that renders glyph shapes. Between a solid
 * fill (which proves only that the LEDs light) and full text (which fails for
 * many different reasons at once), one big character isolates whether glyph
 * rendering lands on the right pixels: a human knows instantly what an "A"
 * should look like, so a wrong mapping is obvious without measuring anything.
 */

#include <Arduino.h>

namespace char_mode {
void setChar(char c);
char current();
void render();
}
