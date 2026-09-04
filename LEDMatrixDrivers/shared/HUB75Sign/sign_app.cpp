/*
 * Road Sentinel LED sign — startup and the main scheduler.
 *
 * Portable. This file describes the PRODUCT, not the microcontroller, so both
 * the ESP32 and STM32 builds share it verbatim.
 *
 * The board does exactly one job: draw on the panel. It has no WiFi, no clock
 * and no knowledge of the system. The Raspberry Pi stays the only networked
 * device and tells this board what to show over USB serial.
 *
 * Why the intelligence lives on the Pi: one fewer thing to provision on site,
 * one fewer credential to rotate, and a wired link cannot drop the way site
 * WiFi does. It also localises faults — if the sign shows the WRONG THING the
 * bug is in the Pi bridge; if it shows it WRONGLY the bug is in this firmware.
 *
 * Layout
 *   panel_config.h     every hardware fact, in one place
 *   framebuffer.h      the 3-bit pixel store
 *   panel_map.cpp      the measured panel mapping
 *   hub75_<board>.cpp  the driver — hand-written, one file per board
 *   display.cpp        Adafruit_GFX bound to the framebuffer
 *   mode_status.cpp    A — production status screens
 *   mode_char.cpp      B — single-character bring-up test
 *   mode_sand.cpp      C — falling-sand pixel test
 *   protocol.cpp       serial command parsing
 *   sign_app.cpp       this file: startup and the scheduler
 *
 * See WIRING.md for the pinout and README.md for the protocol spec.
 */

#include <Arduino.h>
#include "sign_app.h"
#include "panel_config.h"
#include "app.h"
#include "display.h"
#include "hub75.h"
#include "protocol.h"
#include "mode_status.h"
#include "mode_char.h"
#include "mode_sand.h"

// Mode ownership is global rather than inside signapp, because protocol.cpp
// switches modes and should not need to know about the app wrapper.
namespace app {

static Mode     current     = MODE_STATUS;
static uint32_t lastCommand = 0;

void setMode(Mode m)  { current = m; }
Mode mode()           { return current; }
void markCommand()    { lastCommand = millis(); }
uint32_t sinceCommand() { return millis() - lastCommand; }

} // namespace app

namespace signapp {

static uint32_t lastSandStep = 0;

void begin() {
  Serial.begin(115200);

  // The hand-written driver has no allocation that can fail: the framebuffer
  // is static and the refresh is started unconditionally. Nothing to check,
  // which is one fewer failure mode than the DMA library had.
  display::begin();

  app::markCommand();
  app::setMode(app::MODE_STATUS);
  status_mode::setState(status_mode::ST_BOOT);

  Serial.println("READY");
}

void tick() {
  // Commands first, always. Nothing below blocks, so a state change is acted
  // on within one iteration even mid-animation.
  protocol::poll();

  uint32_t now = millis();

  switch (app::mode()) {
    case app::MODE_STATUS:
      status_mode::tick(now);

      // The offline fallback applies ONLY to the production mode. A bring-up
      // test must not be yanked off the panel 15 seconds in just because
      // nobody is polling — that would make the sand test useless for the
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

        // The sequence is one-shot. When it finishes, return to status duty
        // by itself rather than waiting to be told — nothing external has to
        // remember to put a safety sign back to work.
        if (sand::isDone()) {
          app::setMode(app::MODE_STATUS);
          status_mode::render();
        }
      }
      break;

    case app::MODE_CHAR:
    case app::MODE_DIAG:
      // Static output — drawn once when set, nothing to advance.
      break;
  }
}

} // namespace signapp
