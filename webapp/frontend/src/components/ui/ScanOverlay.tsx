import React from "react";
import { cn } from "@/lib/utils";
import { Scan } from "lucide-react";

interface ScanOverlayProps {
  active?: boolean;
  scanning?: boolean;
  label?: string;
}

export function ScanOverlay({ active = true, scanning = false, label = "READY" }: ScanOverlayProps) {
  if (!active) return null;

  return (
    <div className="absolute inset-0 pointer-events-none z-20 overflow-hidden">
        {/* Corner Brackets - Large */}
        <svg className="absolute inset-0 w-full h-full text-cyan-500/40 p-2">
            <path d="M20,2 L2,2 L2,20" fill="none" stroke="currentColor" strokeWidth="1" />
            <path d="M2,80 L2,98 L20,98" transform="translate(0, -100%) rotate(-90) translate(0, 100%)" fill="none" stroke="currentColor" strokeWidth="1" style={{transformOrigin: 'bottom left'}} /> 
            {/* Actually CSS is easier for corners than responsive SVG path math here */}
        </svg>

        {/* CSS Corners */}
        <div className="absolute top-2 left-2 w-8 h-8 border-t-2 border-l-2 border-cyan-400/60 rounded-tl" />
        <div className="absolute top-2 right-2 w-8 h-8 border-t-2 border-r-2 border-cyan-400/60 rounded-tr" />
        <div className="absolute bottom-2 left-2 w-8 h-8 border-b-2 border-l-2 border-cyan-400/60 rounded-bl" />
        <div className="absolute bottom-2 right-2 w-8 h-8 border-b-2 border-r-2 border-cyan-400/60 rounded-br" />

        {/* Center Reticule */}
        <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-32 h-32 border border-cyan-500/20 rounded-full flex items-center justify-center">
            <div className="w-1 h-1 bg-cyan-500 rounded-full animate-pulse" />
            <div className="absolute w-full h-px bg-cyan-500/20" />
            <div className="absolute h-full w-px bg-cyan-500/20" />
        </div>

        {/* Scanning Line (if active) */}
        {scanning && (
            <div className="absolute inset-x-0 top-0 h-1/2 border-b-2 border-cyan-500/40 bg-gradient-to-t from-cyan-500/10 to-transparent animate-[scanline_2s_linear_infinite]" />
        )}

        {/* Label Status */}
        <div className="absolute bottom-4 left-1/2 -translate-x-1/2 px-2 py-0.5 bg-slate-900/80 border border-cyan-500/30 text-[10px] font-mono text-cyan-400 rounded uppercase tracking-widest flex items-center gap-2">
            {scanning && <span className="w-1.5 h-1.5 rounded-full bg-cyan-500 animate-ping" />}
            {scanning ? "ACQUIRING..." : label}
        </div>
    </div>
  );
}
