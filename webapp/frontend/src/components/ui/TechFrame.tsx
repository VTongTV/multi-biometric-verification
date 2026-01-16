import React from "react";
import { cn } from "@/lib/utils";

interface TechFrameProps {
  children: React.ReactNode;
  label?: string;
  className?: string;
  color?: 'blue' | 'indigo' | 'violet' | 'emerald' | 'cyan';
  corner?: 'tl' | 'tr' | 'bl' | 'br';
}

export function TechFrame({ children, label, className, color = 'cyan', corner = 'tr' }: TechFrameProps) {
  
  const colorStyles = {
      blue: "border-blue-300 bg-blue-50/20",
      indigo: "border-indigo-300 bg-indigo-50/20",
      violet: "border-violet-300 bg-violet-50/20",
      emerald: "border-emerald-300 bg-emerald-50/20",
      cyan: "border-cyan-300 bg-cyan-50/20",
  };

  const labelStyles = {
      blue: "text-blue-700 border-blue-300 bg-blue-50",
      indigo: "text-indigo-700 border-indigo-300 bg-indigo-50",
      violet: "text-violet-700 border-violet-300 bg-violet-50",
      emerald: "text-emerald-700 border-emerald-300 bg-emerald-50",
      cyan: "text-cyan-700 border-cyan-300 bg-cyan-50",
  };

  const cornerColor = {
      blue: "border-blue-500",
      indigo: "border-indigo-500",
      violet: "border-violet-500",
      emerald: "border-emerald-500",
      cyan: "border-cyan-500",
  };

  return (
    <div className={cn(
        "relative border transition-all duration-300 group", 
        colorStyles[color],
        className
    )}>
      {/* Technical Corner Brackets */}
      <div className={cn("absolute -top-[1px] -left-[1px] w-3 h-3 border-t-2 border-l-2 z-10", cornerColor[color])} />
      <div className={cn("absolute -top-[1px] -right-[1px] w-3 h-3 border-t-2 border-r-2 z-10", cornerColor[color])} />
      <div className={cn("absolute -bottom-[1px] -left-[1px] w-3 h-3 border-b-2 border-l-2 z-10", cornerColor[color])} />
      <div className={cn("absolute -bottom-[1px] -right-[1px] w-3 h-3 border-b-2 border-r-2 z-10", cornerColor[color])} />

      {/* Label Tag */}
      {label && (
        <div className={cn(
            "absolute -top-3 px-2 py-0.5 text-[9px] font-bold font-mono uppercase tracking-widest border z-20 shadow-sm",
            corner === 'tr' ? "right-4" : "left-4",
            labelStyles[color]
        )}>
            {label}
        </div>
      )}
      {children}
    </div>
  );
}
