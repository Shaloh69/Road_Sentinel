#pragma once
/*
 * The one piece of global state: which mode owns the panel right now.
 *
 * Kept in its own header so protocol.cpp can switch modes without including
 * main.cpp, and so the mode implementations stay unaware of each other.
 */

#include <Arduino.h>

namespace app {

enum Mode {
  MODE_STATUS,   // A — production status screens
  MODE_CHAR,     // B — single-character bring-up test
  MODE_SAND,     // C — falling-sand pixel test
  MODE_DIAG      // static diagnostic output (FILL / DIAG / RAW*)
};

void setMode(Mode m);
Mode mode();

// Records that a command arrived, which resets the offline timeout.
void markCommand();

// Milliseconds since the last command of any kind.
uint32_t sinceCommand();

}
