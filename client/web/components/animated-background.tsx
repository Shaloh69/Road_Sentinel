"use client";

export const AnimatedBackground = () => {
  return (
    <div className="fixed inset-0 -z-10 overflow-hidden">
      {/* Base gradient background */}
      <div className="absolute inset-0 bg-gradient-to-br from-[#1B1931] via-[#44174E] to-[#862249]" />

      {/* Animated gradient orbs */}
      <div className="absolute top-0 left-0 w-full h-full">
        {/* Orb 1 - Large purple */}
        <div
          className="absolute top-[10%] left-[10%] w-96 h-96 bg-[#44174E] rounded-full mix-blend-multiply filter blur-3xl opacity-70 animate-blob"
        />

        {/* Orb 2 - Medium burgundy */}
        <div
          className="absolute top-[30%] right-[20%] w-80 h-80 bg-[#862249] rounded-full mix-blend-multiply filter blur-3xl opacity-70 animate-blob animation-delay-2000"
        />

        {/* Orb 3 - Small peach */}
        <div
          className="absolute bottom-[20%] left-[30%] w-72 h-72 bg-[#ED9E59] rounded-full mix-blend-multiply filter blur-3xl opacity-50 animate-blob animation-delay-4000"
        />

        {/* Orb 4 - Medium rose */}
        <div
          className="absolute bottom-[10%] right-[10%] w-80 h-80 bg-[#A34054] rounded-full mix-blend-multiply filter blur-3xl opacity-60 animate-blob animation-delay-3000"
        />
      </div>

      {/* Animated grid overlay */}
      <div className="absolute inset-0 bg-grid-pattern opacity-5" />

      {/* Subtle noise texture */}
      <div className="absolute inset-0 bg-noise opacity-5" />
    </div>
  );
};
