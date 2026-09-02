/*
 * Road Sentinel — ESP32 HUB75 LED sign firmware
 *
 * The ESP32 does exactly one job: draw on the panel. It has no WiFi, no clock
 * and no knowledge of the system. The Raspberry Pi stays the only networked
 * device and tells this board what to show over USB serial.
 *
 * Why the intelligence lives on the Pi: one fewer thing to provision on site,
 * one fewer credential to rotate, and a wired link cannot drop the way site
 * WiFi does. It also localises faults — if the sign shows the WRONG THING the
 * bug is in the Pi bridge; if it shows it WRONGLY the bug is in this firmware.
 *
 * Layout
 *   include/config.h   every hardware fact, in one place
 *   display.cpp        matrix init + the 128x32 logical canvas
 *   mode_status.cpp    A — production status screens
 *   mode_char.cpp      B — single-character bring-up test
 *   mode_sand.cpp      C — falling-sand pixel test
 *   protocol.cpp       serial command parsing
 *   main.cpp           this file: setup, and a non-blocking scheduler
 *
 * See README.md for the wiring table and protocol spec, and DEBUG_LOG.md for
 * what has already been ruled out on real hardware.
 */

#include <Arduino.h>
#include "config.h"
#include "app.h"
#include "display.h"
#include "protocol.h"
#include "mode_status.h"
#include "mode_char.h"
#include "mode_sand.h"

namespace app {

static Mode     current     = MODE_STATUS;
static uint32_t lastCommand = 0;

void setMode(Mode m)  { current = m; }
Mode mode()           { return current; }
void markCommand()    { lastCommand = millis(); }

uint32_t sinceCommand() { return millis() - lastCommand; }

} // namespace app

static uint32_t lastSandStep = 0;

void setup() {
  Serial.begin(115200);

  if (!display::begin()) {
    // Nothing can be drawn, so the serial line is the only way to say so.
    // Repeat rather than print once: the Pi bridge opens the port after the
    // board has already booted and would miss a single startup message.
    while (true) {
      Serial.println("ERR display init failed (DMA alloc)");
      delay(2000);
    }
  }

  app::markCommand();
  app::setMode(app::MODE_STATUS);
  status_mode::setState(status_mode::ST_BOOT);

  Serial.println("READY");
}

void loop() {
  // Commands first, always. Nothing below blocks, so a state change is acted
  // on within one loop iteration even mid-animation.
  protocol::poll();

  uint32_t now = millis();

  switch (app::mode()) {
    case app::MODE_STATUS:
      status_mode::tick(now);

      // Offline fallback applies ONLY to the production mode. A bring-up test
      // must not be yanked off the panel 15 seconds in just because nobody is
      // polling — that would make the sand test unusable for the exact
      // unattended hardware observation it exists for.
      if (status_mode::state() != status_mode::ST_OFFLINE &&
          status_mode::state() != status_mode::ST_BOOT &&
          app::sinceCommand() > COMMAND_TIMEOUT_MS) {
        status_mode::setState(status_mode::ST_OFFLINE);
      }
      break;

    case app::MODE_SAND:
      if (now - lastSandStep >= SAND_INTERVAL_MS) {
        lastSandStep = now;
        sand::step();
      }
      break;

    case app::MODE_CHAR:
    case app::MODE_DIAG:
      // Static output — drawn once when set, nothing to advance.
      break;
  }
}
