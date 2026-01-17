import React, { useRef, useState } from "react";
import { motion } from "framer-motion";
import { cn } from "@/lib/utils";

interface TiltWrapperProps {
  children: React.ReactNode;
  className?: string;
  intensity?: number;
}

export function TiltWrapper({ children, className, intensity = 5 }: TiltWrapperProps) {
  const ref = useRef<HTMLDivElement>(null);
  const [rotation, setRotation] = useState({ x: 0, y: 0 });

  const handleMouseMove = (e: React.MouseEvent<HTMLDivElement>) => {
    if (!ref.current) return;
    const rect = ref.current.getBoundingClientRect();
    
    // Calculate mouse position relative to center of element
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;
    const centerX = rect.width / 2;
    const centerY = rect.height / 2;
    
    // Calculate rotation (inverted X for natural tilt)
    // Reduce intensity for subtle effect
    const rotateX = ((y - centerY) / centerY) * -intensity;
    const rotateY = ((x - centerX) / centerX) * intensity;

    setRotation({ x: rotateX, y: rotateY });
  };

  const handleMouseLeave = () => {
    setRotation({ x: 0, y: 0 });
  };

  return (
    <motion.div
      ref={ref}
      className={cn("relative [perspective:1000px] flex flex-col", className)}
      onMouseMove={handleMouseMove}
      onMouseLeave={handleMouseLeave}
      animate={{ 
        rotateX: rotation.x, 
        rotateY: rotation.y 
      }}
      transition={{ 
        type: "spring", 
        stiffness: 300, 
        damping: 30,
        mass: 0.5 
      }}
      style={{ 
        transformStyle: "preserve-3d" 
      }}
    >
      {children}
    </motion.div>
  );
}
