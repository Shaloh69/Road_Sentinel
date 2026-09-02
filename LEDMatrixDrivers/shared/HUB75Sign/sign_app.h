#pragma once
/*
 * Startup and the main scheduler. Portable: it describes the PRODUCT, not the
 * microcontroller, so both boards share it verbatim and each keeps only a
 * ten-line main.cpp shim.
 */

#include <Arduino.h>

namespace signapp {

void begin();   // call from setup()
void tick();    // call from loop(); never blocks

}
