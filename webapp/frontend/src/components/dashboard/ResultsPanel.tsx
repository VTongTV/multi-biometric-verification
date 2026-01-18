import React, { useState } from "react";
import { Card } from "@/components/ui/Card";
import { CheckCircle2, XCircle, AlertTriangle, ShieldCheck, ShieldAlert, Fingerprint, ScanFace, Eye, Activity, History, ChevronRight, BarChart3, Scan } from "lucide-react";
import { cn } from "@/lib/utils";
import { getDisplayName } from "@/lib/identityMap";
import { RadarChart } from "@/components/dashboard/RadarChart";
import { InfoTip } from "@/components/ui/InfoTip";

export interface AnalysisResult {
  mode: 'verify' | 'identify';
  status: 'granted' | 'denied' | 'revoked' | 'identified' | 'unknown' | 'unauthorized_match';
  subjectId: string;
  score: number;
  details?: Record<string, number>;
  timestamp: string;
  refId: string;
}

interface ResultsPanelProps {
  result: AnalysisResult | null;
  history: AnalysisResult[];
  onSelectHistory: (res: AnalysisResult) => void;
}

export function ResultsPanel({ result, history, onSelectHistory }: ResultsPanelProps) {
  const [view, setView] = useState<'current' | 'history'>('current');

  if (!result && history.length === 0) {
    return (
      <Card variant="tech" className="h-full flex flex-col items-center justify-center p-8 text-center bg-slate-50 border-slate-200">
        <div className="w-24 h-24 bg-white flex items-center justify-center mb-6 text-slate-300 border border-slate-200 shadow-sm relative group">
            <div className="absolute top-0 left-0 w-2 h-2 border-t border-l border-slate-400 group-hover:border-cyan-500 transition-colors"></div>
            <div className="absolute bottom-0 right-0 w-2 h-2 border-b border-r border-slate-400 group-hover:border-cyan-500 transition-colors"></div>
            <div className="absolute top-0 right-0 w-2 h-2 border-t border-r border-slate-400 group-hover:border-cyan-500 transition-colors"></div>
            <div className="absolute bottom-0 left-0 w-2 h-2 border-b border-l border-slate-400 group-hover:border-cyan-500 transition-colors"></div>
            <Scan className="w-10 h-10 animate-pulse text-cyan-500/50" />
        </div>
        <h3 className="text-slate-700 font-bold text-lg mb-2 font-mono uppercase tracking-widest flex items-center gap-2">
            <div className="w-2 h-2 bg-emerald-500 rounded-full animate-pulse"></div>
            System_Ready
        </h3>
        <p className="text-slate-400 text-[10px] font-mono max-w-xs leading-relaxed uppercase tracking-wider">
          // AWAITING_INPUT_STREAM...<br/>
          // SENSORS_ONLINE
        </p>
      </Card>
    );
  }
  
  const statusConfig = {
    granted: { color: "text-emerald-700", bg: "bg-emerald-50", border: "border-emerald-300", icon: ShieldCheck, label: "Access Granted" },
    denied: { color: "text-rose-700", bg: "bg-rose-50", border: "border-rose-300", icon: XCircle, label: "Access Denied" },
    revoked: { color: "text-amber-700", bg: "bg-amber-50", border: "border-amber-300", icon: ShieldAlert, label: "Access Revoked" },
    identified: { color: "text-cyan-700", bg: "bg-cyan-50", border: "border-cyan-300", icon: CheckCircle2, label: "Identity Confirmed" },
    unknown: { color: "text-slate-600", bg: "bg-slate-100", border: "border-slate-300", icon: AlertTriangle, label: "Unknown Subject" },
    unauthorized_match: { color: "text-amber-700", bg: "bg-amber-50", border: "border-amber-300", icon: AlertTriangle, label: "Match (Unauthorized)" }
  };
  
  const displayResult = result || history[0]; 
  const config = statusConfig[displayResult?.status] || statusConfig.unknown;
  const Icon = config.icon;

  return (
    <Card variant="tech" className="h-full flex flex-col overflow-hidden bg-white" noPadding>
      {/* Header Tabs */}
      <div className="flex border-b border-slate-200 p-1 bg-slate-50/50 shrink-0">
          <button 
            onClick={() => setView('current')}
            className={cn(
                "flex-1 py-2 text-[10px] font-bold font-mono uppercase tracking-widest flex items-center justify-center gap-2 transition-all border relative overflow-hidden",
                view === 'current' ? "bg-white text-cyan-700 shadow-sm border-slate-300" : "text-slate-400 border-transparent hover:text-slate-600 hover:bg-slate-100/50"
            )}
          >
            {view === 'current' && <div className="absolute top-0 left-0 w-full h-[2px] bg-cyan-500"></div>}
            <Activity className="w-3.5 h-3.5" /> <InfoTip content="Real-time biometric analysis stream">Live_Analysis</InfoTip>
          </button>
          <button 
            onClick={() => setView('history')}
            className={cn(
                "flex-1 py-2 text-[10px] font-bold font-mono uppercase tracking-widest flex items-center justify-center gap-2 transition-all border relative overflow-hidden",
                view === 'history' ? "bg-white text-cyan-700 shadow-sm border-slate-300" : "text-slate-400 border-transparent hover:text-slate-600 hover:bg-slate-100/50"
            )}
          >
             {view === 'history' && <div className="absolute top-0 left-0 w-full h-[2px] bg-cyan-500"></div>}
             <History className="w-3.5 h-3.5" /> <InfoTip content="Review past verification attempts">History_Log ({history.length})</InfoTip>
          </button>
      </div>

      {view === 'current' ? (
          <div className="flex-1 flex flex-col min-h-0">
            <div className={cn("p-5 border-b relative overflow-hidden shrink-0", config.bg, config.border)}>
                {/* Background Pattern */}
                <div className="absolute inset-0 opacity-5 bg-[linear-gradient(45deg,#000_25%,transparent_25%,transparent_75%,#000_75%,#000),linear-gradient(45deg,#000_25%,transparent_25%,transparent_75%,#000_75%,#000)] bg-[length:10px_10px] bg-[position:0_0,5px_5px]"></div>
                
                <div className="flex items-center justify-between mb-6 relative z-10">
                    <div className="flex items-center gap-4">
                        <div className={cn("w-16 h-16 flex items-center justify-center bg-white shadow-sm border-2 relative", config.border)}>
                             {/* Corners */}
                            <div className="absolute top-0 left-0 w-1.5 h-1.5 border-t-2 border-l-2 border-current opacity-50"></div>
                            <div className="absolute bottom-0 right-0 w-1.5 h-1.5 border-b-2 border-r-2 border-current opacity-50"></div>
                            <Icon className={cn("w-8 h-8", config.color)} />
                        </div>
                        <div>
                            <div className={cn("font-bold text-xl tracking-tighter font-mono flex items-center gap-2", config.color)}>
                                {config.label.toUpperCase()}
                            </div>
                            <div className="text-slate-500 text-[9px] font-mono font-bold flex gap-3 uppercase tracking-wider mt-1">
                                <span className="flex items-center gap-1"><span className="w-1 h-1 bg-slate-400 rounded-full"></span> TS: {displayResult.timestamp}</span>
                                <span className="text-slate-300">|</span>
                                <span className="flex items-center gap-1"><span className="w-1 h-1 bg-slate-400 rounded-full"></span> REF: {displayResult.refId}</span>
                            </div>
                        </div>
                    </div>
                    <div className="text-right">
                        <div className="text-[9px] font-bold text-slate-500 uppercase tracking-widest mb-1 font-mono">
                            <InfoTip content="Aggregate matching probability">Confidence_Score</InfoTip>
                        </div>
                        <div className={cn("text-4xl font-black tracking-tighter font-mono", config.color)}>
                            {(displayResult.score * 100).toFixed(1)}<span className="text-xl opacity-60">%</span>
                        </div>
                    </div>
                </div>
                
                <div className="bg-white/90 backdrop-blur-sm border border-slate-200 p-3 flex justify-between items-center shadow-sm relative overflow-hidden z-10">
                    <div className={cn("absolute left-0 top-0 bottom-0 w-1.5", config.color.replace('text-', 'bg-').replace('700', '500'))} />
                    <span className="text-[9px] font-bold text-slate-500 uppercase tracking-widest font-mono pl-2 flex items-center gap-2">
                        <ScanFace className="w-3 h-3" /> <InfoTip content="Matched identity from database">Subject_Identity</InfoTip>
                    </span>
                    <span className="font-bold text-lg text-slate-800 font-mono">
                        {displayResult.subjectId === "Unknown" ? "UNKNOWN_ENTITY" : (
                            <span className="flex items-center gap-2">
                                <span className="text-cyan-700">{getDisplayName(displayResult.subjectId)}</span>
                                <span className="text-[9px] text-slate-400 border border-slate-200 px-1 bg-slate-50">
                                    {displayResult.subjectId.replace('user_', 'ID-')}
                                </span>
                            </span>
                        )}
                    </span>
                </div>
            </div>

            <div className="p-5 bg-white flex-1 overflow-y-auto">
                
                {displayResult.details ? (
                    <div className="flex flex-col gap-6">
                        {/* Radar Chart */}
                        <div className="flex justify-center py-4 bg-slate-50 border border-slate-200 relative">
                            <div className="absolute top-2 left-2 text-[8px] font-mono text-slate-400 uppercase tracking-widest flex items-center gap-1">
                                <Activity className="w-2.5 h-2.5" /> <InfoTip content="Feature vector comparison visualization">Vector_Map</InfoTip>
                            </div>
                            <RadarChart data={displayResult.details} />
                        </div>

                        {/* Detailed Bars */}
                        <div className="grid grid-cols-2 gap-3">
                            {Object.entries(displayResult.details).map(([key, value]) => (
                                <div key={key} className="p-2 bg-white border border-slate-200 shadow-sm relative group hover:border-cyan-200 transition-colors">
                                    <div className="flex justify-between items-center mb-1.5">
                                        <span className="text-[9px] font-bold text-slate-500 uppercase flex items-center gap-1.5 font-mono tracking-wider">
                                            {key.includes('face') && <ScanFace className="w-2.5 h-2.5 text-cyan-500" />}
                                            {key.includes('finger') && <Fingerprint className="w-2.5 h-2.5 text-blue-500" />}
                                            {key.includes('iris') && <Eye className="w-2.5 h-2.5 text-emerald-500" />}
                                            {key.replace('_', ' ')}
                                        </span>
                                        <span className={cn(
                                            "text-[10px] font-bold font-mono",
                                            value > 0.8 ? "text-emerald-600" : value > 0.5 ? "text-amber-600" : "text-slate-400"
                                        )}>
                                            {(value * 100).toFixed(1)}%
                                        </span>
                                    </div>
                                    <div className="h-1.5 w-full bg-slate-100 overflow-hidden">
                                        <div 
                                            className={cn("h-full transition-all duration-500", 
                                                value > 0.8 ? "bg-emerald-500" : value > 0.5 ? "bg-amber-500" : "bg-slate-300"
                                            )} 
                                            style={{ width: `${value * 100}%` }}
                                        />
                                    </div>
                                </div>
                            ))}
                        </div>
                    </div>
                ) : (
                    <div className="text-center py-8 text-slate-400 text-xs font-mono">
                        // NO_VECTOR_DATA_AVAILABLE
                    </div>
                )}
            </div>
          </div>
      ) : (
          /* HISTORY VIEW */
          <div className="flex-1 flex flex-col min-h-0 overflow-y-auto p-4 bg-slate-50 custom-scrollbar relative">
             <div className="space-y-2">
                {history.map((item) => {
                    const hConfig = statusConfig[item.status] || statusConfig.unknown;
                    const HIcon = hConfig.icon;
                    return (
                        <button 
                            key={item.refId}
                            onClick={() => { onSelectHistory(item); setView('current'); }}
                            className="w-full flex items-center gap-3 p-2 bg-white border border-slate-200 hover:border-cyan-400 hover:shadow-md transition-all text-left group relative overflow-hidden"
                        >
                             <div className={cn("w-8 h-8 flex items-center justify-center shrink-0 border bg-slate-50 transition-colors group-hover:bg-white", hConfig.border)}>
                                <HIcon className={cn("w-4 h-4", hConfig.color)} />
                             </div>
                             <div className="flex-1 min-w-0">
                                 <div className="flex items-center justify-between">
                                     <span className={cn("text-[9px] font-bold uppercase tracking-wider font-mono", hConfig.color)}>{hConfig.label}</span>
                                     <span className="text-[9px] text-slate-400 font-mono flex items-center gap-1">
                                         {item.timestamp}
                                     </span>
                                 </div>
                                 <div className="flex items-center justify-between mt-1">
                                    <span className="text-[11px] font-bold text-slate-700 truncate font-mono">
                                        {item.subjectId === 'Unknown' ? 'UNK_SUBJECT' : getDisplayName(item.subjectId)}
                                    </span>
                                    <span className="text-[9px] font-bold bg-slate-100 px-1.5 py-0.5 border border-slate-200 text-slate-600 font-mono group-hover:border-cyan-200 group-hover:text-cyan-700 transition-colors">
                                        {(item.score * 100).toFixed(1)}%
                                    </span>
                                 </div>
                             </div>
                             <div className="w-1 h-full absolute left-0 top-0 bg-transparent group-hover:bg-cyan-400 transition-colors"></div>
                        </button>
                    )
                })}
                {history.length === 0 && (
                    <div className="text-center py-10 text-slate-400 text-xs font-mono">// LOG_EMPTY</div>
                )}
             </div>
          </div>
      )}
    </Card>
  );
}
