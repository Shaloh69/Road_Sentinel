#pragma once
/*
 * Serial command protocol — newline-terminated ASCII at 115200.
 *
 * Wire compatibility with raspi_scripts/esp32_display_bridge.py is a hard
 * requirement: STATE:, TEXT:, BRIGHT: and PING must keep working exactly as
 * they do today. Everything else is additive.
 *
 * Every command is answered with "OK" or "ERR <reason>", so the Pi can tell
 * "board is wedged" apart from "board rejected that". PING answers "PONG".
 */

#include <Arduino.h>

namespace protocol {

// Drains whatever bytes are waiting and dispatches any complete lines.
// Never blocks: it returns immediately when the buffer holds a partial line,
// so an animation running at 30fps is not stalled by a half-received command.
void poll();

}
