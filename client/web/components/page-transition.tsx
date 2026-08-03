"use client";

import type { ReactNode } from "react";

import { AnimatePresence, motion, useReducedMotion } from "framer-motion";
import { usePathname } from "next/navigation";

// Route-level page transition (Phase 3 design deliverable #5). HeroUI's own
// components moved to native CSS transitions, but cross-route transitions
// aren't something the App Router or HeroUI handle themselves, so this is
// the one deliberate Framer Motion usage in the app.
export function PageTransition({ children }: { children: ReactNode }) {
  const pathname = usePathname();
  const reduceMotion = useReducedMotion();

  if (reduceMotion) {
    return <div className="page-transition-container">{children}</div>;
  }

  return (
    <AnimatePresence initial={false} mode="wait">
      <motion.div
        key={pathname}
        animate={{ opacity: 1, y: 0 }}
        className="page-transition-container"
        exit={{ opacity: 0, y: -8 }}
        initial={{ opacity: 0, y: 8 }}
        transition={{ duration: 0.22, ease: [0.4, 0, 0.2, 1] }}
      >
        {children}
      </motion.div>
    </AnimatePresence>
  );
}
