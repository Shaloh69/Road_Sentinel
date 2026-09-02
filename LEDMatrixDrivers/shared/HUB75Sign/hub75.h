#pragma once
/*
 * The driver interface every board must implement.
 *
 * ─────────────────────────────────────────────────────────────────────────
 * WHY THIS IS HAND-WRITTEN AND NOT A LIBRARY
 *
 * ESP32-HUB75-MatrixPanel-I2S-DMA was tried exhaustively against these
 * panels: all five built-in scan mappings, both line decoders, the
 * maintainer's own parameterised custom mapping, five geometries. None
 * rendered legible text. A bit-banged test written in an afternoon drove the
 * same panel correctly on the first try.
 *
 * The cause is a single unverifiable assumption. That library exposes a DMA
 * framebuffer whose column index is *assumed* to equal the shift-register
 * clock position. On these panels it does not, and nothing in its API lets
 * you correct for it — so every scan-mapping preset was adjusting a layer
 * sitting on top of a broken foundation.
 *
 * Here the driver emits the clock pulses itself, so position `p` IS the p-th
 * pulse. There is nothing left to assume. Full trail in DEBUG_LOG.md.
 * ─────────────────────────────────────────────────────────────────────────
 *
 * WHAT A PORT MUST PROVIDE
 *
 * Only the four functions below. Everything else — the framebuffer, the
 * panel mapping, text rendering, the display modes and the serial protocol —
 * is portable and shared, because it describes the PANEL and the PRODUCT
 * rather than the microcontroller.
 *
 * A port is therefore roughly 150 lines: configure the pins, write the
 * FM6124 init sequence, and run a refresh loop that walks the address lines
 * shifting `panelmap::ch1/ch2` out of `fbuf::pixels`.
 *
 * Existing ports:
 *   esp32/src/hub75_esp32.cpp   ESP32 dev board  (Road Sentinel Pi 4 sign)
 *   stm32/src/hub75_stm32.cpp   WeAct Black Pill (Road Sentinel Pi 5 sign)
 */

#include <Arduino.h>
#include "panel_config.h"
#include "framebuffer.h"
#include "panel_map.h"

namespace hub75 {

// Configure pins, initialise the panel controller, build the map and start
// refreshing. Must not return until the display is live.
void begin();

// 0-255. Implemented as the lit window per address (an OE duty cycle), so it
// is a true brightness control and does not distort colour.
void setBrightness(uint8_t b);
uint8_t brightness();

// Refresh rate actually achieved, for the INFO command. Worth reporting: a
// driver that silently drops to 30Hz looks like flicker rather than like a
// bug, and that is a miserable thing to diagnose from a photograph.
uint32_t framesPerSecond();

// ── Convenience wrappers, so callers need only include this header ─────────
inline void    setPixel(int16_t x, int16_t y, uint8_t c) { fbuf::set(x, y, c); }
inline uint8_t getPixel(int16_t x, int16_t y)            { return fbuf::get(x, y); }
inline void    clear(uint8_t c = HC_OFF)                 { fbuf::clear(c); }

} // namespace hub75
