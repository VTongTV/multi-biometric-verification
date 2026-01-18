import React, { useEffect, useRef } from "react";
import { Card } from "@/components/ui/Card";
import { cn } from "@/lib/utils";
import { Terminal, ShieldCheck, ShieldAlert, AlertCircle, Info, CheckCircle2, Activity } from "lucide-react";
import { TypewriterText } from "@/components/ui/TypewriterText";

export interface LogEntry {
  id: string;
  timestamp: string;
  message: string;
  type: "info" | "success" | "warning" | "error";
}

interface AnalysisFeedProps {
  logs: LogEntry[];
}

export function AnalysisFeed({ logs }: AnalysisFeedProps) {
  const scrollRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    }
  }, [logs]);

  return (
    <Card variant="tech" className="flex flex-col gap-0 h-full w-full bg-slate-50 border-slate-200" noPadding>
      <div className="p-3 border-b border-slate-200 flex items-center justify-between bg-white">
          <h3 className="text-xs font-bold text-slate-700 flex items-center gap-2 uppercase tracking-widest font-mono">
            <Terminal className="w-3.5 h-3.5 text-cyan-600" />
            System_Audit_Log
          </h3>
          <div className="flex items-center gap-2 px-2 py-0.5 bg-slate-50 border border-slate-200">
             <span className="w-1.5 h-1.5 bg-cyan-500 animate-[pulse_1s_infinite] shadow-sm"></span>
             <span className="text-[9px] font-bold text-cyan-600 uppercase tracking-wide font-mono">Stream_Active</span>
          </div>
      </div>
      
      <div className="flex-1 bg-white p-4 overflow-hidden relative">
        {/* Subtle Scanline effect overlay for light theme - extremely low opacity */}
        <div className="absolute inset-0 bg-[linear-gradient(rgba(18,16,16,0)_50%,rgba(0,0,0,0.02)_50%),linear-gradient(90deg,rgba(0,0,0,0.01),rgba(0,0,0,0.01),rgba(0,0,0,0.01))] z-10 pointer-events-none bg-[length:100%_4px,3px_100%] opacity-30"></div>
        
        <div 
            ref={scrollRef}
            className="absolute inset-0 p-4 overflow-y-auto space-y-1 custom-scrollbar font-mono text-xs z-0"
        >
            {logs.length === 0 && (
            <div className="text-slate-400 italic text-center mt-10 font-mono">
                <span className="animate-pulse">_</span> AWAITING_SYSTEM_EVENTS...
            </div>
            )}
            
            {logs.map((log, index) => {
                const isLast = index === logs.length - 1;
                return (
                    <div key={log.id} className="flex gap-3 items-start animate-in fade-in slide-in-from-left-2 duration-300 group border-l-2 border-transparent hover:border-slate-300 pl-2 -ml-2 transition-all hover:bg-slate-50/50 py-1">
                        <span className="text-slate-400 shrink-0 text-[10px] pt-0.5 font-mono">[{log.timestamp}]</span>
                        <div className="flex-1 break-words">
                            <div className={cn("flex items-start gap-2 font-mono text-[11px]", {
                                "text-slate-600": log.type === "info",
                                "text-emerald-600 font-bold": log.type === "success",
                                "text-amber-600 font-bold": log.type === "warning",
                                "text-rose-600 font-bold": log.type === "error",
                            })}>
                                <span className="opacity-80 font-bold">
                                    {log.type === 'success' && ">> SUCCESS:"}
                                    {log.type === 'warning' && ">> WARNING:"}
                                    {log.type === 'error' && ">> FATAL:"}
                                    {log.type === 'info' && ">"}
                                </span>
                                <span className="leading-tight tracking-wide">
                                    {isLast ? <TypewriterText text={log.message} speed={5} /> : log.message}
                                </span>
                            </div>
                        </div>
                    </div>
                );
            })}
            
            <div className="w-2 h-4 bg-slate-300 animate-pulse ml-2 inline-block align-middle opacity-50 mt-1"></div>
        </div>
      </div>
    </Card>
  );
}
