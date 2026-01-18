import React, { useRef, useEffect, useState } from "react";
import { Activity, Database, Server, Cpu, Radio, ShieldCheck, Zap } from "lucide-react";
import { cn } from "@/lib/utils";

interface HeaderStatsProps {
  latency: number;
  dbCount: number;
  totalCapacity: number;
  status: 'online' | 'degraded' | 'offline';
}

export function HeaderStats({ latency, dbCount, totalCapacity, status }: HeaderStatsProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  
  // Simulated System Stats
  const [cpuTemp, setCpuTemp] = useState(45);
  const [gpuLoad, setGpuLoad] = useState(12);
  const [memUsage, setMemUsage] = useState("4.2GB");

  useEffect(() => {
    // Stat fluctuations
    const timer = setInterval(() => {
        setCpuTemp(prev => Math.max(30, Math.min(85, prev + (Math.random() - 0.5) * 5)));
        setGpuLoad(prev => Math.max(0, Math.min(100, prev + (Math.random() - 0.5) * 10)));
        // Mem fluctuation
        if (typeof performance !== 'undefined' && (performance as any).memory) {
             setMemUsage(`${((performance as any).memory.usedJSHeapSize / 1024 / 1024).toFixed(1)}MB`);
        }
    }, 2000);
    return () => clearInterval(timer);
  }, []);

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
      // Tech Cyan
      ctx.strokeStyle = status === 'online' ? '#0891b2' : '#e11d48'; 
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
    <div className="flex items-center gap-3 text-[10px] font-mono select-none overflow-hidden">
       {/* Signal Visualizer */}
       <div className="w-20 h-7 bg-slate-50 border border-slate-300 overflow-hidden relative hidden xl:block shadow-inner group shrink-0">
          <div className="absolute inset-0 bg-[linear-gradient(90deg,transparent_90%,rgba(6,182,212,0.1)_100%)] bg-[length:10px_100%] pointer-events-none" />
          <div className="absolute top-0 left-0 w-[2px] h-[2px] bg-cyan-600"></div>
          <div className="absolute top-0 right-0 w-[2px] h-[2px] bg-cyan-600"></div>
          <div className="absolute bottom-0 left-0 w-[2px] h-[2px] bg-cyan-600"></div>
          <div className="absolute bottom-0 right-0 w-[2px] h-[2px] bg-cyan-600"></div>
          <canvas ref={canvasRef} width={80} height={28} />
       </div>

       {/* Horizontal Stats Strip */}
       <div className="flex items-center bg-white border border-slate-200 h-7 px-1 shadow-sm shrink-0">
           {/* Latency */}
           <div className={cn(
               "flex items-center gap-1.5 px-3 border-r border-slate-100",
               latency < 100 ? "text-emerald-600" : latency < 500 ? "text-amber-600" : "text-rose-600"
           )}>
              <Activity className="w-3 h-3" />
              <span className="font-bold">{latency > 0 ? `${latency.toFixed(0)}ms` : "--"}</span>
           </div>

           {/* CPU */}
           <div className="flex items-center gap-1.5 px-3 border-r border-slate-100 text-slate-600 hidden md:flex">
              <Cpu className="w-3 h-3 text-slate-400" />
              <span className="font-bold">CPU: {cpuTemp.toFixed(0)}°C</span>
           </div>

           {/* MEM */}
           <div className="flex items-center gap-1.5 px-3 border-r border-slate-100 text-slate-600 hidden lg:flex">
              <Zap className="w-3 h-3 text-slate-400" />
              <span className="font-bold">MEM: {memUsage}</span>
           </div>

           {/* DB */}
           <div className="flex items-center gap-1.5 px-3 border-r border-slate-100 text-slate-600 hidden sm:flex">
              <Database className="w-3 h-3 text-slate-400" />
              <span className="font-bold">{dbCount}/{totalCapacity}</span>
           </div>

           {/* Encryption Status (Static Mock) */}
           <div className="flex items-center gap-1.5 px-3 text-cyan-700 bg-cyan-50 h-full">
              <ShieldCheck className="w-3 h-3" />
              <span className="font-bold">AES-256</span>
           </div>
       </div>
    </div>
  );
}
