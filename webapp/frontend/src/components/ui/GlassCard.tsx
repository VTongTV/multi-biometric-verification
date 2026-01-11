import React from "react";
import { cn } from "@/lib/utils";

interface GlassCardProps extends React.HTMLAttributes<HTMLDivElement> {
  children: React.ReactNode;
  className?: string;
  hoverEffect?: boolean;
}

export function GlassCard({ 
  children, 
  className, 
  hoverEffect = false,
  ...props 
}: GlassCardProps) {
  return (
    <div
      className={cn(
        "glass-panel rounded-xl p-6 transition-all duration-300",
        hoverEffect && "glass-panel-hover cursor-pointer",
        className
      )}
      {...props}
    >
      {children}
    </div>
  );
}
