# Road Sentinel — Web Dashboard

Next.js 15 + HeroUI v2 dashboard for the Road Sentinel traffic-monitoring system: live dual-camera feeds, analytics, incidents, camera calibration, an authenticated admin terminal, and a public no-login status page.

Styled with **"Night Watch"** — see [`DESIGN.md`](./DESIGN.md) for the design system's tokens, font pairing, and rationale.

## Running locally

```bash
npm install
echo NEXT_PUBLIC_API_URL=http://localhost:3001 > .env.local
npm run dev                        # http://localhost:3000
```

Needs the Node service (`server/node-service`) running for real data — see the repo root [`README.md`](../../README.md) for the full stack setup, or run `start.bat` from the repo root to bring everything up (MySQL, AI service, Node service, this client) in one step.

## Structure

- `app/` — Next.js App Router pages (dashboard, monitor, analytics, incidents, cameras, history, reports, settings, admin, status)
- `components/` — shared UI (video feed, stat cards, alert cards, sidebar, calibration tool, animated background, page transitions)
- `lib/` — client-side helpers (socket connections, CSV/PDF export, localStorage settings)
- `hero.ts` / `styles/globals.css` — the Night Watch design tokens
- `config/` — site metadata and font definitions

## Further docs

- [`DESIGN.md`](./DESIGN.md) — design system rationale
- [`../../docs/documentation.md`](../../docs/documentation.md) — ground-truth codebase audit
- [`../../docs/Summarization.md`](../../docs/Summarization.md) — full revamp record
