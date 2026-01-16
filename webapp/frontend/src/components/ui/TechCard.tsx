import React, { useRef, useState } from "react";
import { motion, useMotionTemplate, useMotionValue, useSpring } from "framer-motion";
import { cn } from "@/lib/utils";

interface TechCardProps {
  children: React.ReactNode;
  className?: string;
  noPadding?: boolean;
  highlight?: boolean; // If true, adds a constant glow
}

export function TechCard({ children, className, noPadding = false, highlight = false }: TechCardProps) {
  const ref = useRef<HTMLDivElement>(null);
  
  // 3D Tilt Effect State
  const x = useMotionValue(0);
  const y = useMotionValue(0);
  
  // Smooth spring physics for the tilt
  const mouseX = useSpring(x, { stiffness: 500, damping: 100 });
  const mouseY = useSpring(y, { stiffness: 500, damping: 100 });

  function onMouseMove({ currentTarget, clientX, clientY }: React.MouseEvent) {
    const { left, top, width, height } = currentTarget.getBoundingClientRect();
    const xPct = (clientX - left) / width - 0.5;
    const yPct = (clientY - top) / height - 0.5;
    
    x.set(xPct);
    y.set(yPct);
  }

  function onMouseLeave() {
    x.set(0);
    y.set(0);
  }

  // Dynamic transforms based on mouse position
  const rotateX = useMotionTemplate`${mouseY}deg`; // Invert Y for tilt
  const rotateY = useMotionTemplate`${mouseX}deg`; // Invert X for tilt
  
  return (
    <motion.div
      ref={ref}
      onMouseMove={onMouseMove}
      onMouseLeave={onMouseLeave}
      style={{
        transformStyle: "preserve-3d",
        // rotateX: rotateX, // Too dizzying for a dashboard, let's keep it subtle 2D highlight
        // rotateY: rotateY
      }}
      className={cn(
        "relative group rounded border border-slate-800 bg-slate-900/60 backdrop-blur-md overflow-hidden transition-all duration-500",
        highlight ? "shadow-[0_0_15px_rgba(6,182,212,0.15)] border-cyan-500/30" : "hover:border-slate-600",
        className
      )}
    >
      {/* 1. Mouse Spotlight Effect */}
      <motion.div
        style={{
          background: useMotionTemplate`
            radial-gradient(
              650px circle at ${mouseX}px ${mouseY}px,
              rgba(6, 182, 212, 0.05),
              transparent 80%
            )
          `,
        }}
        className="pointer-events-none absolute -inset-px opacity-0 transition duration-300 group-hover:opacity-100 z-0"
      />

      {/* 2. Technical Corner Markers (SVG) */}
      <div className="absolute top-0 left-0 w-2 h-2 border-t border-l border-cyan-500/50 rounded-tl-sm opacity-50" />
      <div className="absolute top-0 right-0 w-2 h-2 border-t border-r border-cyan-500/50 rounded-tr-sm opacity-50" />
      <div className="absolute bottom-0 left-0 w-2 h-2 border-b border-l border-cyan-500/50 rounded-bl-sm opacity-50" />
      <div className="absolute bottom-0 right-0 w-2 h-2 border-b border-r border-cyan-500/50 rounded-br-sm opacity-50" />

      {/* Content */}
      <div className={cn("relative z-10 h-full", noPadding ? "" : "p-4")}>
        {children}
      </div>
    </motion.div>
  );
}
