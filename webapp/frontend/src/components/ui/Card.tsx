import React from "react";
import { cn } from "@/lib/utils";
import { motion, HTMLMotionProps } from "framer-motion";

interface CardProps extends Omit<HTMLMotionProps<"div">, "children"> {
  children?: React.ReactNode;
  noPadding?: boolean;
  variant?: 'default' | 'elevated' | 'outline' | 'flat' | 'tech';
}

export const Card = React.forwardRef<HTMLDivElement, CardProps>(
  ({ className, children, noPadding = false, variant = 'default', ...props }, ref) => {
    return (
      <motion.div
        ref={ref}
        initial={{ opacity: 0, y: 10 }}
        animate={{ opacity: 1, y: 0 }}
        className={cn(
          "relative transition-all duration-300 bg-white",
          variant === 'default' && "border border-slate-300 shadow-sm rounded-lg overflow-hidden",
          variant === 'elevated' && "border border-slate-300 shadow-[0_4px_20px_-4px_rgba(0,0,0,0.1)] rounded-lg overflow-hidden", 
          variant === 'outline' && "border-2 border-dashed border-slate-300 bg-slate-50/50 rounded-lg overflow-hidden",
          variant === 'flat' && "bg-slate-50 border border-slate-200 rounded-lg overflow-hidden",
          // Tech variant has square corners and specific border styles. Overflow visible to allow dropdowns.
          variant === 'tech' && "border border-slate-300 bg-white shadow-md relative",
          className
        )}
        {...props}
      >
        {variant === 'tech' && (
            <>
                <div className="absolute top-0 left-0 w-full h-1 bg-gradient-to-r from-cyan-500 via-blue-500 to-cyan-500 opacity-80" />
                <div className="absolute bottom-0 right-0 w-4 h-4 border-b-2 border-r-2 border-slate-400" />
            </>
        )}
        
        <div className={cn("relative z-10 h-full flex flex-col", !noPadding && "p-6")}>
          {children}
        </div>
      </motion.div>
    );
  }
);
Card.displayName = "Card";
