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

enum State { ST_BOOT, ST_CLEAR, ST_VEHICLE, ST_INCIDENT, ST_OFFLINE, ST_TEXT };

void  setState(State s);
State state();

// Two-line free text. Either line scrolls independently if it is too wide for
// the 128px canvas — this is how a speed readout or vehicle count gets shown
// without truncating it.
void setText(const String &l1, const String &l2);

void render();                 // full redraw
void tick(uint32_t now);       // flash + scroll advance; redraws only if due

}
