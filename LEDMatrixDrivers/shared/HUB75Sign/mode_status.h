#pragma once
/*
 * Mode A — system status. The actual production use case.
 *
 * The four states match the rest of Road Sentinel exactly and are NOT
 * reinvented here:
 *
 *   clear / vehicle / incident   come from Node's /api/public/status, which
 *     emits "clear" | "vehicle_incoming" | "incident"
 *     (server/node-service/src/routes/public-status.ts:47). The Pi bridge
 *     shortens vehicle_incoming -> vehicle on the wire. Using the server's
 *     computed state rather than recomputing it here is what stops the
 *     physical sign and the public web page from ever disagreeing.
 *
 *   offline   is decided by this board alone, on a command timeout. The
 *     bridge never sends it. That is deliberate: a dead serial cable then
 *     produces the same honest sign as a dead API, instead of the panel
 *     confidently holding a stale ROAD CLEAR while the system is down.
 */

#include <Arduino.h>

namespace status_mode {

// Named for what the sign SHOWS, so reading the code tells you what a driver
// sees. The wire protocol keeps the server's vocabulary (clear / vehicle /
// incident) because that comes from /api/public/status and must not drift.
//
//   ST_SAFE     -> "SAFE"                       green,  static
//   ST_VEHICLE  -> "VEHICLE INCOMING/SLOW DOWN" yellow, flashing
//   ST_STOP     -> "STOP"                       red,    flashing
// Named for what the sign SHOWS. The incident types come straight from the
// `incidents.incident_type` ENUM in migrate.ts, so the sign can say what is
// actually wrong instead of collapsing every incident into a generic STOP.
//
//   ST_SAFE        "SAFE"                        green,  static
//   ST_VEHICLE     "SLOW DOWN / VEHICLE INCOMING" yellow, flashing  (both approaches)
//   ST_SPEEDING    "SLOW DOWN"                   yellow, flashing
//   ST_CRASH       "CRASH / AHEAD"               red,    flashing
//   ST_STOPPED     "STOPPED / VEHICLE"           red,    flashing
//   ST_CONGESTION  "TRAFFIC / AHEAD"             yellow, flashing
//   ST_STOP        "STOP"                        red,    flashing  (generic fallback)
enum State {
  ST_BOOT,
  ST_SAFE,
  ST_VEHICLE,
  ST_SPEEDING,
  ST_CRASH,
  ST_STOPPED,
  ST_CONGESTION,
  ST_STOP,
  // Lane-hazard and advisory screens. A 2-lane blind curve: one camera per
  // lane, one sign per lane, and the danger is what the driver cannot see
  // around the bend.
  ST_WRONGWAY,     // "WRONG / WAY"       red     - head-on risk
  ST_TRUCK,        // "TRUCK / AHEAD"     yellow  - wide vehicle in the curve
  ST_BUS,          // "BUS / AHEAD"       yellow
  ST_NOOVERTAKE,   // "NO / OVERTAKING"   yellow  - both lanes occupied
  ST_KEEPRIGHT,    // "KEEP / RIGHT"      yellow  - lane discipline
  ST_BLINDCURVE,   // "BLIND / CURVE"     amber   - standing advisory, static
  ST_OFFLINE,
  ST_TEXT,
};

void  setState(State s);
State state();

// Two-line free text. Either line scrolls independently if it is too wide for
// the 128px canvas — this is how a speed readout or vehicle count gets shown
// without truncating it.
void setText(const String &l1, const String &l2);

void render();                 // full redraw
void tick(uint32_t now);       // flash + scroll advance; redraws only if due

}
