import React, { useEffect, useRef } from "react";

export function NoiseBackground() {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    let width = canvas.width = window.innerWidth;
    let height = canvas.height = window.innerHeight;

    const columns = Math.floor(width / 20);
    const drops: number[] = new Array(columns).fill(1);
    
    // Mouse interaction
    let mouseX = 0;
    let mouseY = 0;
    
    const handleResize = () => {
        width = canvas.width = window.innerWidth;
        height = canvas.height = window.innerHeight;
    };
    
    const handleMouseMove = (e: MouseEvent) => {
        mouseX = e.clientX;
        mouseY = e.clientY;
    };

    window.addEventListener("resize", handleResize);
    window.addEventListener("mousemove", handleMouseMove);

    const draw = () => {
        // Semi-transparent black to create trail effect
        ctx.fillStyle = "rgba(10, 10, 15, 0.1)";
        ctx.fillRect(0, 0, width, height);

        ctx.fillStyle = "rgba(6, 182, 212, 0.15)"; // Cyan low opacity
        ctx.font = "12px monospace";

        for (let i = 0; i < drops.length; i++) {
            const text = String.fromCharCode(0x30A0 + Math.random() * 96); // Katakana or similar
            const x = i * 20;
            const y = drops[i] * 20;
            
            // Interaction: if near mouse, glow brighter
            const dist = Math.hypot(x - mouseX, y - mouseY);
            if (dist < 100) {
                ctx.fillStyle = "rgba(34, 211, 238, 0.8)"; // Bright cyan
            } else {
                ctx.fillStyle = "rgba(6, 182, 212, 0.15)";
            }

            ctx.fillText(text, x, y);

            if (y > height && Math.random() > 0.975) {
                drops[i] = 0;
            }
            drops[i]++;
        }
        requestAnimationFrame(draw);
    };

    const animationId = requestAnimationFrame(draw);

    return () => {
        window.removeEventListener("resize", handleResize);
        window.removeEventListener("mousemove", handleMouseMove);
        cancelAnimationFrame(animationId);
    };
  }, []);

  return (
    <canvas 
        ref={canvasRef} 
        className="fixed inset-0 pointer-events-none z-0 opacity-40 mix-blend-screen"
    />
  );
}
