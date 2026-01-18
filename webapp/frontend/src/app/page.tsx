"use client";

import { useState, useCallback, useEffect, useRef } from "react";
import { Server, Database, RefreshCw, Zap, Activity, Radio, Lock, Search } from "lucide-react";
import { HeaderWidgets } from "@/components/dashboard/HeaderWidgets";
import { BiometricForm } from "@/components/dashboard/BiometricForm";
import { AnalysisFeed, LogEntry } from "@/components/dashboard/AnalysisFeed";
import { IdentityManager } from "@/components/dashboard/IdentityManager";
import { ResultsPanel, AnalysisResult } from "@/components/dashboard/ResultsPanel";
import { HeaderStats } from "@/components/dashboard/HeaderStats";
import { SubjectDetailModal } from "@/components/dashboard/DetailModaSubjectl";
import { cn } from "@/lib/utils";
import { getDisplayName } from "@/lib/identityMap";
import { motion } from "framer-motion";
import { NoiseBackground } from "@/components/ui/NoiseBackground";

export default function Home() {
  const [mode, setMode] = useState<"verify" | "identify">("verify");
  const [identities, setIdentities] = useState<string[]>([]);
  const [activeCount, setActiveCount] = useState(20);
  const [authorizedIds, setAuthorizedIds] = useState<Set<string>>(new Set());
  const [isProcessing, setIsProcessing] = useState(false);
  const [isSystemReady, setIsSystemReady] = useState(false);
  const [logs, setLogs] = useState<LogEntry[]>([
    {
      id: "init",
      timestamp: new Date().toLocaleTimeString(),
      message: "SYSTEM_BOOT: Biometric Core Active",
      type: "info"
    }
  ]);
  const [analysisResult, setAnalysisResult] = useState<AnalysisResult | null>(null);
  
  // New Features State
  const [history, setHistory] = useState<AnalysisResult[]>([]);
  const [threshold, setThreshold] = useState(0.75); // Default 75%
  const [inspectId, setInspectId] = useState<string | null>(null);
  const [latency, setLatency] = useState(0);
  const [systemStatus, setSystemStatus] = useState<'online'|'degraded'|'offline'>('online');

  const addLog = useCallback((message: string, type: LogEntry["type"] = "info") => {
    setLogs(prev => [...prev, {
      id: Math.random().toString(36).substr(2, 9),
      timestamp: new Date().toLocaleTimeString(),
      message,
      type
    }]);
  }, []);

  useEffect(() => {
    let isMounted = true;
    const fetchIdentities = async () => {
        try {
            const API_BASE = `${window.location.protocol}//${window.location.hostname}:8000`;
            const res = await fetch(`${API_BASE}/api/v1/identities`);
            if (res.ok) {
                const data: string[] = await res.json();
                if (isMounted) {
                    if (data.length > 0) {
                        // Numeric Sort
                        const sortedData = data.sort((a, b) => {
                            const numA = parseInt(a.replace(/\D/g, '')) || 0;
                            const numB = parseInt(b.replace(/\D/g, '')) || 0;
                            return numA - numB;
                        });

                        setIdentities(sortedData);
                        setActiveCount(sortedData.length);
                        setAuthorizedIds(new Set(sortedData));
                        setIsSystemReady(true);
                        addLog(`DB_LOAD: ${sortedData.length} records indexed`, "success");
                    } else {
                        // Retry if empty (system might be ready but cache empty? or still loading?)
                        // For now assuming we expect data.
                         setTimeout(fetchIdentities, 2000);
                    }
                }
            } else {
                if (isMounted) setTimeout(fetchIdentities, 2000);
            }
        } catch (e) {
            // Backend likely starting up
            if (isMounted) setTimeout(fetchIdentities, 2000);
        }
    };
    
    fetchIdentities();
    
    return () => { isMounted = false; };
  }, [addLog]);

  if (!isSystemReady) {
      return (
        <div className="min-h-screen bg-slate-50 text-cyan-700 font-mono flex flex-col items-center justify-center p-4 relative overflow-hidden">
            <div className="absolute inset-0 bg-[linear-gradient(rgba(18,16,16,0)_50%,rgba(0,0,0,0.1)_50%),linear-gradient(90deg,rgba(255,0,0,0.06),rgba(0,255,0,0.02),rgba(0,0,255,0.06))] z-0 pointer-events-none bg-[length:100%_4px,3px_100%]"></div>
            <div className="animate-spin mb-8 text-cyan-600 relative z-10">
                <RefreshCw className="w-12 h-12" />
            </div>
            <h1 className="text-2xl font-bold mb-4 tracking-widest uppercase relative z-10">Biometric Secure Core</h1>
            <div className="flex flex-col gap-2 items-center text-sm relative z-10">
                <span className="font-bold uppercase tracking-wider text-slate-600">Establishing Secure Connection...</span>
                <span className="text-xs text-slate-400 uppercase tracking-widest">Initializing Caching Protocol & Vector Databases</span>
            </div>
            <div className="mt-8 w-64 h-1 bg-slate-200 overflow-hidden relative z-10">
                <div className="h-full bg-cyan-500 animate-[loading_2s_ease-in-out_infinite] w-1/2"></div>
            </div>
        </div>
      );
  }

  const toggleAuth = (id: string) => {
      setAuthorizedIds(prev => {
          const next = new Set(prev);
          if (next.has(id)) {
              next.delete(id);
              addLog(`AUTH_REVOKED: Subject ${id}`, "warning");
          } else {
              next.add(id);
              addLog(`AUTH_GRANTED: Subject ${id}`, "success");
          }
          return next;
      });
  };

  const handleCountChange = (count: number) => {
      setActiveCount(count);
      addLog(`DB_RESIZE: Active subset set to ${count}`, "info");
  };

  const handleSubmit = async (formData: FormData) => {
    setIsProcessing(true);
    addLog(`PROC_START: Initiating ${mode.toUpperCase()} sequence`, "info");
    
    const steps = ["Acquiring Sensor Data...", "Feature Extraction...", "Vector Normalization...", "Fusion Algorithm..."];
    for (const step of steps) {
        await new Promise(r => setTimeout(r, 300));
        addLog(`>> ${step}`, "info");
    }

    const endpoint = mode === 'verify' ? '/api/v1/verify' : '/api/v1/identify';
    const API_BASE = `${window.location.protocol}//${window.location.hostname}:8000`;
    const API_URL = `${API_BASE}${endpoint}`;

    if (mode === 'identify' && activeCount < identities.length) {
        const activeIdsList = identities.slice(0, activeCount).join(',');
        formData.append('active_ids', activeIdsList);
        addLog(`FILTER_ACTIVE: Searching subset of ${activeCount} records`, "info");
    }

    const startTime = performance.now(); // Start latency timer

    try {
      const response = await fetch(API_URL, {
        method: "POST",
        body: formData,
      });

      const endTime = performance.now(); // End latency timer
      setLatency(endTime - startTime);

      const data = await response.json();

      if (!response.ok) {
        throw new Error(data.detail || `API Error: ${response.statusText}`);
      }
      
      if (mode === 'verify') {
         const claimedId = formData.get('identity') as string;
         // Use custom threshold instead of backend decision for visual demo flexibility
         const passedThreshold = data.score >= threshold;
         const isPermitted = authorizedIds.has(claimedId);
         
         let resultStatus: AnalysisResult['status'] = 'denied';

         if (passedThreshold && isPermitted) {
             resultStatus = 'granted';
             addLog(`RESULT: ACCESS GRANTED [${getDisplayName(claimedId)}]`, "success");
             addLog(`CONFIDENCE: ${(data.score * 100).toFixed(4)}% (Threshold: ${(threshold * 100).toFixed(0)}%)`, "success");
         } else if (passedThreshold && !isPermitted) {
             resultStatus = 'revoked';
             addLog(`RESULT: ACCESS DENIED [REVOKED ID: ${getDisplayName(claimedId)}]`, "error");
             addLog(`WARN: Biometrics Match but Subject is REVOKED`, "warning");
         } else {
             resultStatus = 'denied';
             addLog(`RESULT: ACCESS DENIED [MISMATCH]`, "error");
             addLog(`CONFIDENCE: ${(data.score * 100).toFixed(4)}% (Below Threshold: ${(threshold * 100).toFixed(0)}%)`, "warning");
         }
         
         const newResult: AnalysisResult = {
             mode,
             status: resultStatus,
             subjectId: claimedId,
             score: data.score,
             details: data.details,
             timestamp: new Date().toLocaleTimeString(),
             refId: Math.random().toString(36).substr(2, 6).toUpperCase()
         };

         setAnalysisResult(newResult);
         setHistory(prev => [newResult, ...prev].slice(0, 50)); // Keep last 50

      } else {
         let resultStatus: AnalysisResult['status'] = 'unknown';
         let matchId = "Unknown";

         // Check if score meets our custom threshold
         if (data.score >= threshold && data.match_id !== "Unknown") {
             matchId = data.match_id;
             if (authorizedIds.has(matchId)) {
                resultStatus = 'identified';
                addLog(`IDENTIFIED: ${getDisplayName(matchId)}`, "success");
                addLog(`CONFIDENCE: ${(data.score * 100).toFixed(4)}%`, "success");
             } else {
                resultStatus = 'unauthorized_match';
                addLog(`IDENTIFIED: ${getDisplayName(matchId)} [UNAUTHORIZED]`, "warning");
                addLog(`CONFIDENCE: ${(data.score * 100).toFixed(4)}%`, "success");
             }
         } else {
             // Even if backend found a match, if it's below OUR visual threshold, we treat as unknown
             addLog("RESULT: UNKNOWN SUBJECT (No match above threshold)", "warning");
             addLog(`MAX_SIMILARITY: ${(data.score * 100).toFixed(4)}%`, "info");
         }
         
         const newResult: AnalysisResult = {
             mode,
             status: resultStatus,
             subjectId: matchId,
             score: data.score,
             details: data.details,
             timestamp: new Date().toLocaleTimeString(),
             refId: Math.random().toString(36).substr(2, 6).toUpperCase()
         };

         setAnalysisResult(newResult);
         setHistory(prev => [newResult, ...prev].slice(0, 50));
      }

      if (data.details) {
         const detailsStr = Object.entries(data.details)
            .map(([k, v]) => `${k}:${((v as number) * 100).toFixed(1)}%`)
            .join(" | ");
         addLog(`VECTOR_BREAKDOWN: [ ${detailsStr} ]`, "info");
      }

    } catch (error) {
      console.error(error);
      addLog(`CRITICAL_FAIL: ${error instanceof Error ? error.message : "Unknown Error"}`, "error");
    } finally {
      setIsProcessing(false);
    }
  };

  const handlePresetLoad = (type: "authorized" | "impostor" | "identify" | "stress" | "random") => {
      if (type === 'identify') {
          setMode('identify');
      } else if (type !== 'random') {
          setMode('verify');
      }
      addLog(`PRESET_LOAD: Loading scenario [${type.toUpperCase()}]`, "info");
  };

  return (
    <div className="min-h-screen bg-slate-50 text-slate-800 font-sans text-sm selection:bg-cyan-100 selection:text-cyan-900 flex flex-col relative pb-4">
      <NoiseBackground />
      {/* Global Background Grid */}
      <div className="absolute inset-0 pointer-events-none z-0" 
           style={{ 
               backgroundImage: 'linear-gradient(rgba(0,0,0,0.03) 1px, transparent 1px), linear-gradient(90deg, rgba(0,0,0,0.03) 1px, transparent 1px)', 
               backgroundSize: '40px 40px' 
           }} 
      />
      <div className="absolute inset-0 pointer-events-none z-0 bg-[radial-gradient(circle_at_center,_var(--tw-gradient-stops))] from-transparent via-slate-100/50 to-slate-200/50 opacity-60"></div>

      {/* Light Header */}
      <nav className="bg-white/80 backdrop-blur-md border-b border-slate-200 shrink-0 z-40 h-14 flex items-center shadow-sm relative">
        <div className="max-w-full w-full mx-auto px-4 flex justify-between items-center">
            <div className="flex items-center gap-4 group cursor-default">
              {/* Tech Logo Box */}
              <div className="w-9 h-9 bg-slate-900 text-cyan-400 flex items-center justify-center font-bold font-mono text-sm shadow-sm relative overflow-hidden transition-all group-hover:shadow-[0_0_15px_rgba(6,182,212,0.3)]">
                 <div className="absolute inset-0 bg-[linear-gradient(45deg,transparent_25%,rgba(255,255,255,0.1)_50%,transparent_75%,transparent_100%)] bg-[length:250%_250%,100%_100%] bg-[position:-100%_0,0_0] bg-no-repeat transition-[background-position_0s_ease] hover:bg-[position:200%_0,0_0] duration-1000"></div>
                 <div className="absolute top-0 left-0 w-2 h-2 border-t border-l border-cyan-500"></div>
                 <div className="absolute bottom-0 right-0 w-2 h-2 border-b border-r border-cyan-500"></div>
                 BL
              </div>
              
              <div className="leading-tight font-mono hidden md:block">
                  <div className="font-bold text-slate-900 tracking-widest text-base uppercase flex items-center gap-2">
                    BIOSECURE <span className="text-cyan-600">LABS</span>
                  </div>
                  <div className="text-[9px] text-slate-400 tracking-[0.2em] uppercase">
                      Advanced Biometric Systems
                  </div>
              </div>
            </div>

            <HeaderWidgets />
            
            <div className="flex items-center gap-4">
               <HeaderStats 
                  latency={latency} 
                  dbCount={activeCount} 
                  totalCapacity={identities.length}
                  status={systemStatus}
               />
            </div>
        </div>
      </nav>

      {/* Main Content */}
      <main className="flex-1 w-full mx-auto p-4 flex flex-col gap-4 relative z-10">
        <SubjectDetailModal 
            subjectId={inspectId} 
            onClose={() => setInspectId(null)} 
        />
        
        {/* TOP SECTION: Controls (Flex Grow - Dominating Space) */}
        <div className="flex-1 grid grid-cols-1 lg:grid-cols-12 gap-4">
            {/* Identity Manager (5 cols) */}
            <div className="lg:col-span-5 flex flex-col h-[0px] min-h-full">
                 <div className="h-full flex-1 min-h-0">
                     <IdentityManager 
                        identities={identities}
                        activeCount={activeCount}
                        authorizedIds={authorizedIds}
                        onCountChange={handleCountChange}
                        onToggleAuth={toggleAuth}
                        threshold={threshold}
                        onThresholdChange={setThreshold}
                        onInspect={setInspectId}
                    />
                 </div>
            </div>

            {/* Biometric Controls (7 cols) */}
            <div className="lg:col-span-7 flex flex-col gap-3 h-full">
                 {/* Mode Switcher */}
                <div className="bg-white border border-slate-300 p-1 flex shadow-sm shrink-0">
                   <button 
                      onClick={() => setMode('verify')}
                      className={cn(
                          "flex-1 py-1.5 text-xs font-bold uppercase tracking-widest transition-all duration-200 relative overflow-hidden font-mono flex items-center justify-center gap-2",
                          mode === 'verify' 
                            ? "bg-slate-900 text-cyan-400 shadow-sm border border-slate-700" 
                            : "text-slate-500 hover:text-slate-800 hover:bg-slate-50"
                      )}
                   >
                      <Lock className="w-3 h-3" /> Verify (1:1)
                   </button>
                   <button 
                      onClick={() => setMode('identify')}
                      className={cn(
                          "flex-1 py-1.5 text-xs font-bold uppercase tracking-widest transition-all duration-200 relative overflow-hidden font-mono flex items-center justify-center gap-2",
                          mode === 'identify' 
                            ? "bg-slate-900 text-cyan-400 shadow-sm border border-slate-700" 
                            : "text-slate-500 hover:text-slate-800 hover:bg-slate-50"
                      )}
                   >
                      <Search className="w-3 h-3" /> Identify (1:N)
                   </button>
                </div>

                <div className="flex-1">
                        <BiometricForm 
                            mode={mode}
                            identities={identities} 
                            onSubmit={handleSubmit}
                            isProcessing={isProcessing}
                            onPresetLoad={handlePresetLoad}
                            result={analysisResult}
                        />
                </div>
            </div>
        </div>

        {/* BOTTOM SECTION: Results & Logs (Fixed Height) */}
        <div className="h-96 shrink-0 grid grid-cols-1 lg:grid-cols-2 gap-4">
            <div className="h-full">
                <ResultsPanel result={analysisResult} history={history} onSelectHistory={setAnalysisResult} />
            </div>
            <div className="h-full">
                <AnalysisFeed logs={logs} />
            </div>
        </div>
      </main>
    </div>
  );
}
