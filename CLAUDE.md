# Road Sentinel — working agreement

AI blind-curve warning system for Barangay Busay, Cebu. Thesis project.
Dual camera → YOLO detection → MySQL → Next.js dashboard + roadside LED signs.

## Hard rules

**IMPORTANT: never commit secrets.** `.env`, `*.env.txt`, `authorized_keys`,
tunnel URLs, and DB passwords stay out of git. Check `git status` before every
commit. `.gitignore` already covers `.env`, `*.pt`, `datasets/`, `__pycache__`.

**YOU MUST verify before claiming something works.** Run it, read the output,
paste the evidence. "Should work" is not a result. If it cannot be verified
without hardware access, say so explicitly rather than implying success.

**IMPORTANT: never fabricate hardware verification.** Anything needing a
physical Pi, camera, or LED panel is unverified until someone looks at it and
says what they saw. Mark those 🟡, never ✅.

**Never push on a failing audit.** Commit locally, say what is open.

**Do not run crash/incident model training.** It is a GPU job the user runs
deliberately. Never start it in the background.

## Architecture, briefly

| Piece | Stack | Port |
|---|---|---|
| `server/ai-service` | FastAPI, YOLO26 via ultralytics | 8000 |
| `server/node-service` | Express + Socket.IO + MySQL | 3001 |
| `client/web` | Next.js 15 + HeroUI, "Night Watch" design system | 3000 |
| `raspi_scripts` | camera_sender, led_sign_bridge, pi_agent | — |
| `LEDMatrixDrivers` | HUB75 sign firmware: shared core + ESP32 port (both signs); STM32 port is unused reference | — |

`migrate.ts` is the authoritative DB schema. `mysql_schema.sql` is a generated
reference — do not edit it by hand and expect it to take effect.

## Things that have bitten us

Each of these cost real debugging time. They are here so they cost it once.

**MySQL has no `ADD COLUMN IF NOT EXISTS`.** That is MariaDB/Postgres syntax.
In MySQL it is a hard parse error, not a no-op. Catch `ER_DUP_FIELDNAME`
instead. Only ever surfaces against a genuinely fresh database.

**`Stop-ScheduledTask` does not stop Node.** The task spawns
npm → nodemon → node; stopping it leaves the child chain alive holding the old
environment, so an edited `.env` silently does nothing. Kill node explicitly.

**CORS fails invisibly.** When a Cloudflare quick-tunnel URL rotates, Node's
`CORS_ORIGIN` no longer matches and responses come back with no
`Access-Control-Allow-Origin`. `curl` shows a healthy 200; only the browser
fails. Run `rewire_tunnels.ps1` after any tunnel restart.

**Do not write files with `echo pw | sudo -S tee <<EOF`.** The heredoc and the
password share stdin, so `tee` writes the password instead of the content.
Write the file first, then `sudo cp`.

**A library's abstraction can hide an unverifiable assumption.**
`ESP32-HUB75-MatrixPanel-DMA` exposes a DMA framebuffer whose column index is
*assumed* to equal the shift-register clock position. On our panels it is not,
and its API cannot correct for it — so all five scan-mapping presets were
adjusting a layer above a broken foundation and none could ever converge. Cost:
~80 Pi configurations plus a week on the ESP32. A hand-written driver, where
position `p` IS the p-th clock pulse, worked immediately. When a config sweep
fails uniformly, suspect the layer beneath it.

**Panel facts are measured, never inferred.** This project reached two
confident WRONG conclusions from specs and silkscreens ("pin 12 is D", "1/16
scan"), wrote both into docs as *resolved*, and lost days. Two bit-banged
tests — walk A/B/C and count lit rows, then colour each quarter of the shift
register and photograph where it lands — settled everything in under an hour.
See `LEDMatrixDrivers/esp32/DEBUG_LOG.md`.

**A test where everything lights tells you nothing.** A full-screen fill writes
every pixel, so it looks identical under a correct mapping and a broken one.
Several hardware sessions were spent on exactly that non-test. Positioned
content is the only thing that exercises addressing.

**Quick-tunnel URLs rotate on every restart.** Pi→server traffic uses the
stable Tailscale address, never a tunnel URL.

## Working style

Prefer the smallest change that fixes the actual cause. When a fix is not
working, stop adding parameters and ask what evidence would distinguish the
competing explanations — sweeping a config space that has already failed
twelve times is a sign the model of the problem is wrong, not that the sweep
needs to be wider.

State uncertainty plainly. If a theory is disproven — including one confidently
argued earlier — say so directly rather than quietly moving on.

Cite file:line when describing code. Keep commit messages explanatory: what
broke, why, how it was verified.

## Docs

`docs/` holds the audit (`documentation.md`), the phase-by-phase revamp record
(`Summarization.md`), and deployment (`DEPLOYMENT.md`). `client/web/DESIGN.md`
covers the design system. `raspi_scripts/` holds hardware notes.

Update the relevant doc in the same commit as the change it describes.
