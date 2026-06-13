"use client";

import { MotionConfig } from "framer-motion";
import type { ReactNode } from "react";

/** Opts every framer-motion animation into the user's reduced-motion preference
    (disables transforms/scale; keeps non-vestibular opacity). Pairs with the
    CSS @media (prefers-reduced-motion) block that stills the CSS animations. */
export default function MotionProvider({ children }: { children: ReactNode }) {
  return <MotionConfig reducedMotion="user">{children}</MotionConfig>;
}
