import React from "react";
import { Card } from "@/components/ui/Card";
import { Users, CheckCircle2, XCircle, Search, Shield, Database, Lock, Unlock } from "lucide-react";
import { InfoTip } from "@/components/ui/InfoTip";
import { cn } from "@/lib/utils";
import { getDisplayName } from "@/lib/identityMap";
import { DecryptionText } from "@/components/ui/DecryptionText";

interface IdentityManagerProps {
  identities: string[];
  activeCount: number;
  authorizedIds: Set<string>;
  onCountChange: (count: number) => void;
  onToggleAuth: (id: string) => void;
  threshold: number;
  onThresholdChange: (val: number) => void;
  onInspect: (id: string) => void;
}

export function IdentityManager({ 
  identities, 
  activeCount, 
  authorizedIds, 
  onCountChange, 
  onToggleAuth,
  threshold,
  onThresholdChange,
  onInspect
}: IdentityManagerProps) {

  return (
    <Card variant="tech" className="flex flex-col gap-0 h-full bg-slate-50" noPadding>
      <div className="flex items-center justify-between border-b border-slate-200 p-3 shrink-0 bg-white">
        <div className="text-xs font-bold text-slate-700 flex items-center gap-2 font-mono uppercase tracking-widest">
           <div className="p-1 border border-cyan-200 bg-cyan-50 text-cyan-700 shadow-[0_0_10px_rgba(6,182,212,0.1)]">
                <Database className="w-3.5 h-3.5" /> 
           </div>
           <InfoTip content="Manage active user pool and set authorization permissions." placement="bottom">Subject_Database</InfoTip>
        </div>
        <div className="flex items-center gap-0 text-[9px] font-bold border border-slate-300 font-mono divide-x divide-slate-300">
           <span className="text-emerald-700 bg-emerald-50 px-2 py-0.5">{authorizedIds.size} AUTH</span>
           <span className="text-cyan-700 bg-cyan-50 px-2 py-0.5">{activeCount} ACTIVE</span>
        </div>
      </div>

      <div className="flex flex-col gap-3 p-3 flex-1 min-h-0 overflow-hidden relative bg-slate-50/50">
          {/* Controls Container */}
          <div className="grid grid-cols-2 gap-2 shrink-0">
             {/* DB Size Control */}
             <div className="bg-white p-2 border border-slate-200 hover:border-cyan-400 transition-all group shadow-sm">
                 <div className="flex items-center justify-between mb-2">
                    <label className="text-[9px] font-bold text-slate-500 uppercase tracking-widest font-mono flex items-center gap-1">
                        <div className="w-1 h-1 bg-cyan-500 rounded-full"></div>
                        <InfoTip content="Adjust size of active subset for identification tests.">DB_SIZE</InfoTip>
                    </label>
                    <span className="text-[9px] font-mono font-bold text-cyan-600 bg-cyan-50 px-1.5 py-0.5 border border-cyan-100">{activeCount.toString().padStart(2, '0')}</span>
                 </div>
                 <input 
                    type="range" 
                    min="1" 
                    max={identities.length || 20} 
                    value={activeCount}
                    onChange={(e) => onCountChange(parseInt(e.target.value))}
                    className="w-full accent-cyan-600 h-1 bg-slate-100 appearance-none cursor-pointer rounded-none"
                 />
             </div>

             {/* Threshold Control */}
             <div className="bg-white p-2 border border-slate-200 hover:border-cyan-400 transition-all group shadow-sm">
                 <div className="flex items-center justify-between mb-2">
                    <label className="text-[9px] font-bold text-slate-500 uppercase tracking-widest font-mono flex items-center gap-1">
                        <div className="w-1 h-1 rounded-full bg-emerald-500"></div>
                        <InfoTip content="Minimum confidence score required for a positive match.">SEC_LEVEL</InfoTip>
                    </label>
                    <span className={cn("text-[9px] font-mono font-bold px-1.5 py-0.5 border", threshold > 0.85 ? "text-rose-700 bg-rose-50 border-rose-200" : "text-emerald-700 bg-emerald-50 border-emerald-200")}>
                        {(threshold * 100).toFixed(0)}%
                    </span>
                 </div>
                 <input 
                    type="range" 
                    min="50" 
                    max="99" 
                    step="1"
                    value={threshold * 100}
                    onChange={(e) => onThresholdChange(parseInt(e.target.value) / 100)}
                    className={cn(
                        "w-full h-1 bg-slate-100 appearance-none cursor-pointer rounded-none",
                        threshold > 0.85 ? "accent-rose-500" : threshold > 0.7 ? "accent-emerald-500" : "accent-amber-500"
                    )}
                 />
             </div>
          </div>

          {/* Grid */}
          <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 lg:grid-cols-2 xl:grid-cols-3 gap-1.5 p-0.5 flex-1 overflow-y-auto custom-scrollbar content-start">
             {identities.map((id, index) => {
                const isActive = index < activeCount;
                const isAuth = authorizedIds.has(id);
                
                return (
                   <div
                       key={id}
                       className={cn(
                        "h-8 border flex items-center justify-between px-2 transition-all relative overflow-hidden group bg-white",
                        !isActive 
                            ? "border-slate-100 opacity-40 grayscale bg-slate-50" 
                            : isAuth
                                ? "border-emerald-300 bg-emerald-50/20 shadow-sm"
                                : "border-slate-300 hover:border-cyan-400 hover:shadow-md hover:bg-cyan-50/10"
                      )}
                   >
                       {/* Tech markers corners */}
                       {isActive && (
                           <>
                               <div className="absolute top-0 left-0 w-[2px] h-[2px] bg-current opacity-20" />
                               <div className="absolute top-0 right-0 w-[2px] h-[2px] bg-current opacity-20" />
                               <div className="absolute bottom-0 left-0 w-[2px] h-[2px] bg-current opacity-20" />
                               <div className="absolute bottom-0 right-0 w-[2px] h-[2px] bg-current opacity-20" />
                           </>
                       )}

                       {/* Auth Toggle Area */}
                       <button 
                           onClick={() => isActive && onToggleAuth(id)}
                           disabled={!isActive}
                           className="flex-1 flex items-center gap-2.5 min-w-0 h-full text-left outline-none group/btn"
                           title={isActive ? "Toggle Permissions" : "Inactive"}
                       >
                            <div className={cn(
                                "w-3.5 h-3.5 border flex items-center justify-center transition-colors shrink-0",
                                !isActive ? "bg-slate-100 border-slate-300" : isAuth ? "bg-emerald-500 border-emerald-600 text-white" : "bg-white border-slate-300 text-slate-300 group-hover/btn:border-cyan-400 group-hover/btn:text-cyan-400"
                            )}>
                                {isAuth ? <Lock className="w-2 h-2" /> : <Unlock className="w-2 h-2" />}
                            </div>
                            
                            <div className="flex flex-col min-w-0 flex-1">
                                <span className={cn(
                                    "text-[10px] font-bold truncate transition-colors font-mono tracking-wide",
                                    !isActive ? "text-slate-400" : "text-slate-600 group-hover:text-cyan-700"
                                )}>
                                    <DecryptionText 
                                        text={id.replace('user_', 'ID_')} 
                                        speed={30} 
                                        revealSpeed={50} 
                                        useHover={isActive} 
                                    />
                                </span>
                            </div>
                       </button>
                      
                       {/* Inspector Button */}
                       {isActive && (
                           <button 
                                onClick={(e) => { e.stopPropagation(); onInspect(id); }}
                                className="w-5 h-5 flex items-center justify-center hover:bg-cyan-50 text-slate-400 hover:text-cyan-600 transition-colors border border-transparent hover:border-cyan-200 rounded-[1px] shrink-0"
                                title="Inspect Record"
                           >
                               <Search className="w-2.5 h-2.5" />
                           </button>
                       )}
                   </div>
                );
             })}
          </div>
          
          <div className="flex justify-between items-center text-[8px] text-slate-400 font-mono shrink-0 pt-2 border-t border-slate-100">
             <span className="uppercase tracking-widest">// ACCESS_CONTROL_LIST</span>
             <span>VER: 2.4.0</span>
          </div>
      </div>
    </Card>
  );
}
