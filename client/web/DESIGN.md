# Design system — "Night Watch" (Phase 3 revamp)

## Before

The client shipped as an unrenamed HeroUI starter (`package.json` still said
`"next-app-template"`) with four unstyled template pages (`/about`, `/blog`,
`/pricing`, `/docs`) wired to nothing, and every real page hand-rolled its
own `bg-white/10 backdrop-blur-md` glass panels over a purple/burgundy
gradient (`#1B1931` / `#44174E` / `#862249` / `#A34054` / `#ED9E59`) with
hex values pasted directly into `className` on a per-file basis — there was
no shared token file, so "brand color" meant eleven slightly different
copies of `#ED9E59`.

## After

One token system (`hero.ts` for HeroUI's native component colors,
`styles/globals.css`'s `@theme` block for Tailwind utilities) named
**"Night Watch"**: a near-black operations-monitoring base (`#0B0E14`)
suited to a 24/7 live-view screen, with **amber** (`#F2B33D`) as the single
brand accent and **red reserved exclusively for critical incident
severity** — everywhere else (nav highlights, buttons, focus rings) uses
amber, teal-green (success), warning-orange, or info-blue instead, so red
never gets diluted into ordinary chrome and still reads as "something is
actually wrong" the moment it appears. Severity chips use a deliberate
four-step ramp — info (low) → amber (medium) → warning-orange (high) →
danger-red (critical) — instead of reusing brand-adjacent colors for
"critical."

## Rationale

A red-primary UI is a poor fit for an incident-safety dashboard: if red is
the dominant color the moment the app loads, it stops meaning "critical
incident." Night Watch keeps the road-safety identity (amber reads as
"traffic signal caution," not "everything is fine") while isolating red as
a single, trustworthy signal — the same reasoning behind RAG (red/amber/
green) conventions in operations tooling generally.

## Typography

**Space Grotesk** (headings) + **IBM Plex Sans** (body) + **IBM Plex Mono**
(stats, timestamps, IDs, FPS/speed readouts, terminal output). Plex Sans and
Plex Mono are a matched superfamily built specifically for data-dense
technical tooling, so body copy and monospace numbers read as one system
rather than two unrelated fonts bolted together; Space Grotesk gives
headings actual geometric character instead of leaning on Inter/system-font
defaults for everything, which the "AI slop" checklist this phase used as a
grounding reference explicitly calls out as a tell.

## What changed structurally

- Deleted the four unwired starter pages (`/about`, `/blog`, `/pricing`,
  `/docs`) and the dead, never-imported `components/navbar.tsx` — the
  sidebar is the only real navigation and always has been.
- Rebuilt `components/animated-background.tsx` as a slow, low-amplitude
  amber/info "aurora" glow (down from four saturated purple/burgundy blob
  orbs) that respects `prefers-reduced-motion`.
- Added `components/page-transition.tsx` — a Framer Motion
  `AnimatePresence` wrapper at the root layout level, covering every route,
  also reduced-motion aware.
- Wired `@heroui/toast`'s native `ToastProvider`/`addToast` (already a
  project dependency — no new library needed) to real events: a new
  incident arriving over the dashboard's live socket (critical incidents
  persist until dismissed instead of auto-timing out), WebSocket
  connect/reconnect/disconnect, settings saved/reset, and report
  download success/failure.
