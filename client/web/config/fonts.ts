import { Space_Grotesk, IBM_Plex_Sans, IBM_Plex_Mono } from "next/font/google";

// Font pairing (Phase 3 design overhaul): Space Grotesk for headings gives
// the dashboard actual technical/HUD character instead of another
// Inter-only interface; IBM Plex Sans/Mono are a matched superfamily built
// for data-dense tooling, so body text and monospace stats (FPS, speeds,
// timestamps, terminal output) read as one deliberate system.
export const fontHeading = Space_Grotesk({
  subsets: ["latin"],
  weight: ["500", "600", "700"],
  variable: "--font-heading-raw",
  display: "swap",
});

export const fontSans = IBM_Plex_Sans({
  subsets: ["latin"],
  weight: ["400", "500", "600"],
  variable: "--font-sans-raw",
  display: "swap",
});

export const fontMono = IBM_Plex_Mono({
  subsets: ["latin"],
  weight: ["400", "500"],
  variable: "--font-mono-raw",
  display: "swap",
});
