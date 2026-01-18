import React, { useRef, useEffect } from "react";
import { Activity, Database, Server } from "lucide-react";
import { cn } from "@/lib/utils";

interface SystemMonitorProps {
  latency: number;
  dbCount: number;
  totalCapacity: number;
  status: 'online' | 'degraded' | 'offline';
}

export function SystemMonitor({ latency, dbCount, totalCapacity, status }: SystemMonitorProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  // Oscilloscope Animation
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    let animationId: number;
    let offset = 0;

    const draw = () => {
      if(!canvas || !ctx) return;
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      ctx.beginPath();
      // Tech Green/Cyan line
      ctx.strokeStyle = status === 'online' ? '#0891b2' : '#e11d48'; // Cyan-600 or Rose-600
      ctx.lineWidth = 1.5;

      for (let x = 0; x < canvas.width; x++) {
        const y = (canvas.height / 2) + 
                 Math.sin((x + offset) * 0.15) * 8 + 
                 (Math.random() - 0.5) * (status === 'degraded' ? 8 : 1.5);
        
        if (x === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
      }
      ctx.stroke();
      offset += 3;
      animationId = requestAnimationFrame(draw);
    };

    draw();
    return () => cancelAnimationFrame(animationId);
  }, [status]);

  return (
    <div className="flex items-center gap-4 text-xs font-mono">
       {/* Signal Visualizer */}
       <div className="w-24 h-8 bg-slate-50 border border-slate-300 overflow-hidden relative hidden md:block shadow-inner group">
          <div className="absolute inset-0 bg-[linear-gradient(90deg,transparent_90%,rgba(6,182,212,0.1)_100%)] bg-[length:10px_100%] pointer-events-none" />
          <div className="absolute top-0 left-0 w-[2px] h-[2px] bg-cyan-600"></div>
          <div className="absolute top-0 right-0 w-[2px] h-[2px] bg-cyan-600"></div>
          <div className="absolute bottom-0 left-0 w-[2px] h-[2px] bg-cyan-600"></div>
          <div className="absolute bottom-0 right-0 w-[2px] h-[2px] bg-cyan-600"></div>
          <canvas ref={canvasRef} width={96} height={32} />
       </div>

       {/* Latency Monitor */}
       <div className={cn(
           "flex items-center gap-1.5 px-3 py-1 border-l-2 transition-colors bg-white border border-y-slate-200 border-r-slate-200 relative",
           latency < 100 ? "border-l-emerald-500 text-emerald-700" : 
           latency < 500 ? "border-l-amber-500 text-amber-700" : 
           "border-l-rose-500 text-rose-700"
       )}>
          <Activity className="w-3.5 h-3.5" />
          <span className="font-bold">{latency > 0 ? `${latency.toFixed(0)}ms` : "--"}</span>
       </div>

       {/* Backend Status */}
       <div className="flex items-center gap-1.5 px-3 py-1 bg-white border border-slate-300 text-slate-600 relative overflow-hidden group">
          <div className="absolute top-0 right-0 w-2 h-2 border-t border-r border-slate-300 group-hover:border-cyan-400 transition-colors"></div>
          <Server className={cn("w-3.5 h-3.5", status === 'online' ? "text-cyan-600" : "text-rose-600")} />
          <span className="font-bold uppercase tracking-wider">{status}</span>
       </div>

       {/* Database Status */}
       <div className="flex items-center gap-1.5 px-3 py-1 bg-white border border-slate-300 text-slate-600 relative overflow-hidden group">
          <div className="absolute bottom-0 left-0 w-2 h-2 border-b border-l border-slate-300 group-hover:border-cyan-400 transition-colors"></div>
          <Database className="w-3.5 h-3.5 text-slate-500" />
          <span className="font-bold">{dbCount}<span className="text-slate-400 mx-0.5">/</span>{totalCapacity}</span>
       </div>
    </div>
  );
}
