import React, { useState, useEffect } from "react";
import { cn } from "@/lib/utils";

interface SystemMarqueeProps {
    latency?: number;
    dbCount?: number;
}

export function SystemMarquee({ latency = 0, dbCount = 0 }: SystemMarqueeProps) {
  const [browserInfo, setBrowserInfo] = useState("Initializing...");
  const [windowSize, setWindowSize] = useState("0x0");
  const [time, setTime] = useState("");
  // Simulated fluctuating values for "liveness"
  const [cpuTemp, setCpuTemp] = useState(45);
  const [gpuLoad, setGpuLoad] = useState(12);

  useEffect(() => {
    // 1. Browser Info
    const ua = window.navigator.userAgent;
    let browser = "Unknown";
    if (ua.indexOf("Chrome") > -1) browser = "Chrome";
    else if (ua.indexOf("Safari") > -1) browser = "Safari";
    else if (ua.indexOf("Firefox") > -1) browser = "Firefox";
    setBrowserInfo(`${window.navigator.platform.toUpperCase()} :: ${browser.toUpperCase()}`);

    // 2. Window Size
    const handleResize = () => {
        setWindowSize(`${window.innerWidth}x${window.innerHeight}`);
    };
    handleResize(); // Initial
    window.addEventListener('resize', handleResize);

    // 3. Timer & Fluctuations
    const timer = setInterval(() => {
        // Time
        const now = new Date();
        setTime(now.toISOString().split('T')[1].split('.')[0] + " UTC");

        // Fluctuations
        setCpuTemp(prev => Math.max(30, Math.min(85, prev + (Math.random() - 0.5) * 5)));
        setGpuLoad(prev => Math.max(0, Math.min(100, prev + (Math.random() - 0.5) * 10)));

    }, 1000);

    return () => {
        window.removeEventListener('resize', handleResize);
        clearInterval(timer);
    };
  }, []);

  const stats = [
      `SYS_TIME: ${time}`,
      `CLIENT: ${browserInfo}`,
      `VIEWPORT: ${windowSize}`,
      `CPU_TEMP: ${cpuTemp.toFixed(1)}°C`,
      `NET_LATENCY: ${latency.toFixed(0)}ms`,
      `DB_RECORDS: ${dbCount}`,
      `ENCRYPTION: AES-256-GCM`,
      `FIREWALL: ACTIVE`,
      `GPU_LOAD: ${gpuLoad.toFixed(0)}%`,
      `MEM_ALLOC: ${(typeof performance !== 'undefined' && (performance as any).memory ? ((performance as any).memory.usedJSHeapSize / 1024 / 1024).toFixed(0) : "---")} MB`,
      `THREAT_LEVEL: LOW`,
      `KERNEL: 5.15.0-91`
  ];

  return (
    <div className="w-full bg-slate-950 border-b border-slate-800 overflow-hidden relative h-6 flex items-center select-none z-50">
        <div className="absolute left-0 top-0 bottom-0 w-8 bg-gradient-to-r from-slate-950 to-transparent z-10"></div>
        <div className="absolute right-0 top-0 bottom-0 w-8 bg-gradient-to-l from-slate-950 to-transparent z-10"></div>
        
        <div className="animate-[marquee_40s_linear_infinite] whitespace-nowrap flex gap-8 items-center text-[9px] font-mono font-bold text-slate-500">
            {/* Duplicate list for seamless loop - Tripled for safety on wide screens */}
            {[...stats, ...stats, ...stats].map((stat, i) => (
                <span key={i} className="flex items-center gap-2">
                    <span className={cn("w-1 h-1 rounded-full", stat.includes("LATENCY") && latency > 100 ? "bg-amber-500" : "bg-cyan-900")}></span>
                    {stat}
                </span>
            ))}
        </div>
    </div>
  );
}
