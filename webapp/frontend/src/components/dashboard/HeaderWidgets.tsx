import React, { useState, useEffect } from 'react';

export function HeaderWidgets() {
  const [time, setTime] = useState<Date | null>(null);
  const [mounted, setMounted] = useState(false);
  const [sessionTime, setSessionTime] = useState(0);

  useEffect(() => {
    setMounted(true);
    // Initialize time on client only to avoid hydration mismatch
    setTime(new Date());

    const timer = setInterval(() => {
      setTime(new Date());
      setSessionTime(prev => prev + 1);
    }, 100); // Keep 100ms for session timer smoothness or move to 1s? 100ms is fine for "tech" feel if we wanted milliseconds, but we are only showing SS.

    return () => clearInterval(timer);
  }, []);

  if (!mounted || !time) return null;

  const formatTime = (date: Date) => {
    return date.toLocaleTimeString('en-US', { hour12: false });
  };

  const formatSession = (seconds: number) => {
    const mins = Math.floor(seconds / 600); // 100ms ticks -> seconds/10
    const secs = Math.floor((seconds % 600) / 10);
    return `${mins.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}`;
  };

  return (
    <div className="flex items-center h-full">
        {/* Clock */}
        <div className="flex flex-col items-center leading-tight">
             <div className="text-2xl font-bold font-mono text-slate-700 tracking-tight tabular-nums">
                {formatTime(time)}
             </div>
             <div className="text-[10px] font-bold text-cyan-600 uppercase tracking-[0.2em] font-mono">
                SESSION: {formatSession(sessionTime)}
             </div>
        </div>
    </div>
  );
}

