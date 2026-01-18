import React, { useState, useRef, useEffect, useLayoutEffect } from "react";
import { createPortal } from "react-dom";
import { Info } from "lucide-react";
import { cn } from "@/lib/utils";

interface InfoTipProps {
  content: string;
  placement?: 'top' | 'bottom';
  children?: React.ReactNode;
  className?: string;
  delay?: number;
}

export function InfoTip({ content, placement = 'top', children, className, delay = 0 }: InfoTipProps) {
  const [isVisible, setIsVisible] = useState(false);
  const [isRendered, setIsRendered] = useState(false);
  const [coords, setCoords] = useState({ top: 0, left: 0, arrowLeft: 0 });
  const [mounted, setMounted] = useState(false);
  
  const triggerRef = useRef<HTMLDivElement>(null);
  const tooltipRef = useRef<HTMLDivElement>(null);
  const timeoutRef = useRef<NodeJS.Timeout | null>(null);
  
  // Handle SSR
  useEffect(() => {
    setMounted(true);
  }, []);

  const handleMouseEnter = () => {
    if (timeoutRef.current) clearTimeout(timeoutRef.current);
    timeoutRef.current = setTimeout(() => {
      setIsRendered(true);
      // Note: setIsVisible(true) is handled in useLayoutEffect after positioning
    }, delay);
  };

  const handleMouseLeave = () => {
    if (timeoutRef.current) clearTimeout(timeoutRef.current);
    setIsVisible(false);
  };

  const onTransitionEnd = () => {
    if (!isVisible) {
      setIsRendered(false);
    }
  };

  // Close tooltip on scroll to prevent detached floating tooltips
  useEffect(() => {
    if (isVisible) {
      const handleScroll = () => {
        setIsVisible(false);
      };
      // Capture scroll events on window to catch any scrolling container
      window.addEventListener('scroll', handleScroll, { capture: true });
      return () => window.removeEventListener('scroll', handleScroll, { capture: true });
    }
  }, [isVisible]);

  // Calculate position and clamp to viewport
  useLayoutEffect(() => {
    if (isRendered && triggerRef.current && tooltipRef.current) {
      const triggerRect = triggerRef.current.getBoundingClientRect();
      const tooltipRect = tooltipRef.current.getBoundingClientRect();
      
      const padding = 12; // Minimal distance from viewport edges
      const gap = 10; // Gap between trigger and tooltip
      
      // Calculate ideal horizontal position (centered)
      let left = triggerRect.left + (triggerRect.width / 2) - (tooltipRect.width / 2);
      
      // Horizontal Clamping
      const maxLeft = window.innerWidth - tooltipRect.width - padding;
      const minLeft = padding;
      
      if (left < minLeft) left = minLeft;
      if (left > maxLeft) left = maxLeft;
      
      // Vertical Position
      let top = 0;
      if (placement === 'top') {
        top = triggerRect.top - tooltipRect.height - gap;
      } else {
        top = triggerRect.bottom + gap;
      }

      // Calculate Arrow Position (relative to tooltip) to point at trigger center
      const triggerCenter = triggerRect.left + (triggerRect.width / 2);
      let arrowLeft = triggerCenter - left;
      
      // Clamp arrow to be inside tooltip (prevent detaching from corners)
      const arrowPadding = 12; 
      if (arrowLeft < arrowPadding) arrowLeft = arrowPadding;
      if (arrowLeft > tooltipRect.width - arrowPadding) arrowLeft = tooltipRect.width - arrowPadding;

      setCoords({ top, left, arrowLeft });
      
      // Use requestAnimationFrame to ensure the DOM is updated/painted 
      // before we start the transition (opacity 1)
      requestAnimationFrame(() => {
          setIsVisible(true);
      });
    }
  }, [isRendered, placement, content]);

  const tooltipContent = (
    <div 
      ref={tooltipRef}
      className={cn(
        "fixed z-[9999] pointer-events-none left-0 top-0",
      )}
      style={{
        transform: `translate3d(${coords.left}px, ${coords.top}px, 0)`,
      }}
    >
        {/* Animation Wrapper */}
        <div 
            className={cn(
                "relative bg-slate-900/90 backdrop-blur-md border border-cyan-500/50 text-cyan-50 text-[10px] px-3 py-2 rounded-sm shadow-[0_0_15px_rgba(6,182,212,0.3)] font-mono tracking-wide w-max max-w-[220px]",
                "transition-all duration-300 ease-out transform",
                isVisible ? "opacity-100 translate-y-0" : "opacity-0",
                !isVisible && (placement === 'top' ? "translate-y-2" : "-translate-y-2")
            )}
            onTransitionEnd={onTransitionEnd}
        >
            {/* Scanline */}
            <div className="absolute inset-0 bg-[linear-gradient(rgba(18,16,16,0)_50%,rgba(0,0,0,0.25)_50%),linear-gradient(90deg,rgba(255,0,0,0.06),rgba(0,255,0,0.02),rgba(0,0,255,0.06))] pointer-events-none bg-[length:100%_2px,3px_100%] opacity-20 rounded-sm"></div>
            
            {/* Corner Accents */}
            <div className="absolute top-0 left-0 w-1.5 h-1.5 border-t border-l border-cyan-400"></div>
            <div className="absolute bottom-0 right-0 w-1.5 h-1.5 border-b border-r border-cyan-400"></div>
            
            <span className="relative z-10">{content}</span>
            
            {/* Arrow */}
            <div className={cn(
                "absolute w-0 h-0 border-l-[6px] border-l-transparent border-r-[6px] border-r-transparent",
                placement === 'top'
                    ? "top-full border-t-[6px] border-t-cyan-500/50"
                    : "bottom-full border-b-[6px] border-b-cyan-500/50"
            )}
            style={{
                left: coords.arrowLeft,
                transform: 'translateX(-50%)'
            }}
            ></div>
        </div>
    </div>
  );

  return (
    <>
      <div 
        ref={triggerRef}
        className={cn(
            "group relative inline-flex items-center cursor-help",
            !children && "ml-1.5",
            className
        )}
        onMouseEnter={handleMouseEnter}
        onMouseLeave={handleMouseLeave}
      >
        {children ? children : (
            <Info className="w-3.5 h-3.5 text-slate-400 hover:text-cyan-400 transition-colors" />
        )}
      </div>
      {mounted && isRendered && createPortal(tooltipContent, document.body)}
    </>
  );
}
