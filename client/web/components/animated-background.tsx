"use client";

// "Night Watch" background (Phase 3): near-black operations base with two
// very low-opacity amber/info glows, a faint HUD grid, and noise to break
// up flat color banding. Deliberately restrained — this sits behind a
// live-view screen meant to be looked at for hours, not a marketing page,
// so the motion is slow and low-amplitude and fully respects
// prefers-reduced-motion (handled in globals.css via .animate-aurora).
export const AnimatedBackground = () => {
  return (
    <div className="fixed inset-0 -z-10 overflow-hidden bg-bg">
      <div className="absolute top-[-10%] left-[-5%] w-[50rem] h-[50rem] bg-brand rounded-full mix-blend-screen filter blur-[140px] opacity-[0.06] animate-aurora" />
      <div className="absolute bottom-[-15%] right-[-10%] w-[45rem] h-[45rem] bg-info rounded-full mix-blend-screen filter blur-[140px] opacity-[0.05] animate-aurora animation-delay-4000" />

      <div className="absolute inset-0 bg-grid-pattern" />
      <div className="absolute inset-0 bg-noise opacity-[0.03]" />
    </div>
  );
};
