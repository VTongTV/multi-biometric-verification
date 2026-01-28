import React, { useState, useEffect, useRef } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { ScanFace, Fingerprint, Eye, CheckCircle2, XCircle, Activity, ShieldCheck, ShieldAlert, Cpu, Scan, Binary, Terminal, Lock, Hash } from "lucide-react";
import { cn } from "@/lib/utils";
import { AnalysisResult } from "./ResultsPanel";
import { DecryptionText } from "@/components/ui/DecryptionText";

interface ModalityData {
    enabled: boolean;
    file: File | null;
    preview: string | null;
    sourceId: string;
    inputType: 'upload' | 'camera';
}

interface VerificationSequenceProps {
    mode: 'verify' | 'identify';
    modalities: {
        face: ModalityData;
        fingerprint: ModalityData;
        irisNir: ModalityData;
        irisVis: ModalityData;
    };
    result: AnalysisResult | null;
    onComplete: () => void;
    isProtected?: boolean;
}

export function VerificationSequence({ mode, modalities, result, onComplete, isProtected = false }: VerificationSequenceProps) {
    const [viewState, setViewState] = useState<'scanning' | 'protecting' | 'analyzed' | 'final'>('scanning');
    
    // Auto-advance logic
    useEffect(() => {
        if (!result) {
            setViewState('scanning');
            return;
        }

        const showResultsTimer = setTimeout(() => {
            // Always show protection step (Template Protection)
            setViewState('protecting');
            
            const protectionTimer = setTimeout(() => {
                setViewState('analyzed');
                scheduleFinal();
            }, 1500); // 1.5s protection animation
            
            return () => clearTimeout(protectionTimer);
        }, 1500); // 1.5s scanning minimum

        function scheduleFinal() {
            const finalTimer = setTimeout(() => {
                setViewState('final');
                
                const closeTimer = setTimeout(() => {
                    onComplete();
                }, 2500); // 2.5s to read the verdict
            }, 2000); // 2s to read the scores
        }

        return () => clearTimeout(showResultsTimer);
    }, [result, onComplete, isProtected]);


    const getScore = (type: string) => {
        if (!result) return 0;
        
        const keys = Object.keys(result.details || {});
        
        if (type === 'face') {
            const k = keys.find(key => key.includes('face'));
            return k && result.details ? (result.details[k] * 100).toFixed(1) : "---";
        }
        if (type === 'fingerprint') {
             const k = keys.find(key => key.includes('fingerprint'));
             return k && result.details ? (result.details[k] * 100).toFixed(1) : "---";
        }
        if (type === 'iris_nir') {
             const k = keys.find(key => key.includes('iris_nir'));
             return k && result.details ? (result.details[k] * 100).toFixed(1) : "---";
        }
        if (type === 'iris_vis') {
             const k = keys.find(key => key.includes('iris_vis'));
             return k && result.details ? (result.details[k] * 100).toFixed(1) : "---";
        }
        return "---";
    };
    
    const getVerdict = (type: string) => {
        const s = getScore(type);
        if (s === "---") return null;
        return parseFloat(s as string) > 75;
    }

    return (
        <div className="absolute inset-0 z-50 bg-slate-950/95 backdrop-blur-xl flex items-center justify-center text-white overflow-hidden">
            {/* Hexagonal Hive Background */}
            <div className="absolute inset-0 opacity-10 pointer-events-none" 
                 style={{ 
                     backgroundImage: `url("data:image/svg+xml,%3Csvg width='60' height='104' viewBox='0 0 60 104' xmlns='http://www.w3.org/2000/svg'%3E%3Cpath d='M30 0l25.98 15v30L30 60 4.02 45V15z' fill='none' stroke='%2306b6d4' stroke-width='1'/%3E%3Cpath d='M30 52l25.98 15v30L30 112 4.02 97V67z' fill='none' stroke='%2306b6d4' stroke-width='1'/%3E%3C/svg%3E")`, 
                     backgroundSize: '60px 104px'
                 }} 
            />
            
            {/* Ambient Glow */}
            <div className="absolute inset-0 bg-radial-gradient from-transparent via-transparent to-black/80 pointer-events-none" />

            <AnimatePresence mode="wait">
                {viewState !== 'final' ? (
                    <motion.div 
                        key="grid"
                        exit={{ opacity: 0, scale: 1.1, filter: "blur(20px)" }}
                        className="relative z-10 grid grid-cols-2 gap-6 w-full h-full max-w-6xl max-h-[85vh] p-6"
                    >
                        {modalities.face.enabled && (
                            <ModalityScanner 
                                title="FACIAL_RECOG"
                                icon={<ScanFace className="w-5 h-5 text-cyan-400" />}
                                image={modalities.face.preview}
                                score={getScore('face')}
                                verdict={getVerdict('face')}
                                color="cyan"
                                type="face"
                                isAnalyzed={viewState === 'analyzed'}
                                isProtecting={viewState === 'protecting'}
                                mode={mode}
                            />
                        )}
                        {modalities.fingerprint.enabled && (
                            <ModalityScanner 
                                title="FINGERPRINT"
                                icon={<Fingerprint className="w-5 h-5 text-emerald-400" />}
                                image={modalities.fingerprint.preview}
                                score={getScore('fingerprint')}
                                verdict={getVerdict('fingerprint')}
                                color="emerald"
                                type="fingerprint"
                                isAnalyzed={viewState === 'analyzed'}
                                isProtecting={viewState === 'protecting'}
                                mode={mode}
                            />
                        )}
                        {modalities.irisNir.enabled && (
                            <ModalityScanner 
                                title="IRIS_NIR"
                                icon={<Eye className="w-5 h-5 text-violet-400" />}
                                image={modalities.irisNir.preview}
                                score={getScore('iris_nir')}
                                verdict={getVerdict('iris_nir')}
                                color="violet"
                                type="iris"
                                isAnalyzed={viewState === 'analyzed'}
                                isProtecting={viewState === 'protecting'}
                                mode={mode}
                            />
                        )}
                        {modalities.irisVis.enabled && (
                            <ModalityScanner 
                                title="IRIS_VIS"
                                icon={<Eye className="w-5 h-5 text-indigo-400" />}
                                image={modalities.irisVis.preview}
                                score={getScore('iris_vis')}
                                verdict={getVerdict('iris_vis')}
                                color="indigo"
                                type="iris"
                                isAnalyzed={viewState === 'analyzed'}
                                isProtecting={viewState === 'protecting'}
                                mode={mode}
                            />
                        )}
                    </motion.div>
                ) : (
                     <FinalVerdict key="final" result={result} mode={mode} />
                )}
            </AnimatePresence>

            {/* Rapid Fire Log */}
            {viewState !== 'final' && <RapidLog />}
        </div>
    );
}

function ModalityScanner({ title, icon, image, score, verdict, color, type, isAnalyzed, isProtecting, mode }: { title: string, icon: React.ReactNode, image: string | null, score: string | number, verdict: boolean | null, color: string, type: string, isAnalyzed: boolean, isProtecting?: boolean, mode: 'verify' | 'identify' }) {
    // Generate some random binary particles
    const particles = useRef([...Array(10)].map((_, i) => ({
        id: i,
        x: Math.random() * 100,
        y: Math.random() * 100,
        delay: Math.random() * 2
    }))).current;

    return (
        <div className={cn(
            "flex flex-col items-center justify-center gap-4 relative w-full h-full p-1 rounded-sm overflow-hidden transition-all duration-500",
            // Holographic Glass Panel
            "bg-slate-900/40 backdrop-blur-md border border-white/10 shadow-[0_0_15px_rgba(0,0,0,0.5)]",
            `hover:border-${color}-500/50 hover:shadow-[0_0_20px_rgba(var(--color-${color}-500),0.2)]`
        )}>
            {/* Header */}
            <div className="flex flex-col items-center gap-1 z-20 w-full pt-4">
                <div className={`p-2 rounded-full bg-${color}-500/10 border border-${color}-500/30 animate-pulse`}>
                    {icon}
                </div>
                <h2 className={cn(`text-lg font-bold font-mono tracking-[0.2em] text-${color}-100`)}>
                    <DecryptionText text={title} speed={50} revealSpeed={100} />
                </h2>
                <div className="flex items-center gap-2">
                     <span className={cn("w-1.5 h-1.5 rounded-full", isAnalyzed ? `bg-${color}-500` : (isProtecting ? "bg-cyan-400 animate-pulse" : "bg-amber-500 animate-ping"))} />
                     <span className={cn("text-[10px] font-mono", isAnalyzed ? `text-${color}-400` : (isProtecting ? "text-cyan-400" : "text-amber-400"))}>
                        {isAnalyzed ? <DecryptionText text="ANALYSIS_COMPLETE" /> : (isProtecting ? "SECURING..." : (mode === 'identify' ? "SCANNING_DB..." : "VERIFYING..."))}
                     </span>
                </div>
            </div>

            {/* Main Visual Container */}
            <div className="relative w-56 h-56 border border-slate-700/50 bg-slate-950/80 rounded-sm overflow-hidden group shadow-inner z-10">
                 {/* Protection Overlay */}
                 {isProtecting && (
                    <motion.div 
                        initial={{ opacity: 0 }}
                        animate={{ opacity: 1 }}
                        exit={{ opacity: 0 }}
                        className="absolute inset-0 z-40 bg-slate-950/90 flex flex-col items-center justify-center backdrop-blur-[2px]"
                    >
                         {/* Animated Grid Background */}
                         <div className="absolute inset-0 opacity-20" style={{ 
                             backgroundImage: `linear-gradient(to right, ${color === 'cyan' ? '#06b6d4' : color === 'emerald' ? '#10b981' : '#8b5cf6'}33 1px, transparent 1px), linear-gradient(to bottom, ${color === 'cyan' ? '#06b6d4' : color === 'emerald' ? '#10b981' : '#8b5cf6'}33 1px, transparent 1px)`, 
                             backgroundSize: '20px 20px' 
                         }}>
                         </div>

                         {/* Lock Animation */}
                         <motion.div
                             initial={{ scale: 2, opacity: 0 }}
                             animate={{ scale: 1, opacity: 1 }}
                             transition={{ duration: 0.5, type: "spring" }}
                             className={`p-3 rounded-full border-2 border-${color}-500 bg-${color}-500/20 shadow-[0_0_30px_rgba(var(--color-${color}-500),0.3)] relative z-10`}
                         >
                             <Lock className={`w-8 h-8 text-${color}-400`} />
                             <motion.div 
                                 animate={{ rotate: 360 }}
                                 transition={{ duration: 2, repeat: Infinity, ease: "linear" }}
                                 className={`absolute inset-[-4px] border border-dashed border-${color}-500/50 rounded-full`}
                             />
                         </motion.div>
                         
                         {/* Hashing Text */}
                         <div className={`mt-4 font-mono text-xs font-bold text-${color}-400 flex flex-col items-center gap-1`}>
                             <DecryptionText text="BIOHASH_PROTECTION" speed={30} />
                             <span className="text-[8px] text-slate-500 tracking-wider">ORTHOGONAL_PROJECTION</span>
                         </div>
                         
                         {/* Random Hash Stream */}
                         <div className="mt-2 font-mono text-[8px] text-slate-600/80 w-40 text-center break-all leading-tight">
                             <DecryptionText text="8F3A2C1B9D4E5F6A7B8C9D0E1F2A3B4C5D6E7F8A9B0C1D2E3F4A5B6C7D8E9F0A" speed={5} revealSpeed={20} />
                         </div>
                    </motion.div>
                 )}

                 {/* Data Stream Particles */}
                 {!isAnalyzed && !isProtecting && particles.map(p => (
                     <motion.div 
                        key={p.id}
                        initial={{ top: "-10%", left: `${p.x}%`, opacity: 0 }}
                        animate={{ top: "110%", opacity: [0, 1, 0] }}
                        transition={{ duration: 1.5, repeat: Infinity, delay: p.delay, ease: "linear" }}
                        className={cn(`absolute w-0.5 h-4 bg-${color}-500/50 z-30`)}
                     />
                 ))}

                 {/* Image */}
                {image && (
                     <img src={image} className="w-full h-full object-cover opacity-60 grayscale mix-blend-luminosity" alt="scan target" />
                )}
                
                {/* Animations based on type */}
                {type === 'face' && !isAnalyzed && (
                    <>
                        <motion.div 
                            initial={{ top: "0%" }}
                            animate={{ top: "100%" }}
                            transition={{ repeat: Infinity, duration: 1.5, ease: "linear" }}
                            className="absolute left-0 right-0 h-[2px] bg-cyan-500 shadow-[0_0_15px_rgba(6,182,212,0.8)] z-20"
                        />
                        <div className="absolute inset-0 border-[20px] border-cyan-500/5 rounded-[40px] scale-75" />
                        <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-32 h-32 border border-dashed border-cyan-500/30 rounded-full animate-spin-slow" />
                    </>
                )}

                {/* Target Lock Reticles (Animated Corners) */}
                {!isAnalyzed && (
                    <>
                        <motion.div initial={{ scale: 1.5, opacity: 0 }} animate={{ scale: 1, opacity: 1 }} transition={{ duration: 0.5 }} className={`absolute top-2 left-2 w-4 h-4 border-t-2 border-l-2 border-${color}-500`} />
                        <motion.div initial={{ scale: 1.5, opacity: 0 }} animate={{ scale: 1, opacity: 1 }} transition={{ duration: 0.5 }} className={`absolute top-2 right-2 w-4 h-4 border-t-2 border-r-2 border-${color}-500`} />
                        <motion.div initial={{ scale: 1.5, opacity: 0 }} animate={{ scale: 1, opacity: 1 }} transition={{ duration: 0.5 }} className={`absolute bottom-2 left-2 w-4 h-4 border-b-2 border-l-2 border-${color}-500`} />
                        <motion.div initial={{ scale: 1.5, opacity: 0 }} animate={{ scale: 1, opacity: 1 }} transition={{ duration: 0.5 }} className={`absolute bottom-2 right-2 w-4 h-4 border-b-2 border-r-2 border-${color}-500`} />
                    </>
                )}

                {type === 'fingerprint' && !isAnalyzed && (
                     <motion.div 
                        initial={{ height: "0%" }}
                        animate={{ height: "100%" }}
                        transition={{ repeat: Infinity, duration: 2, ease: "easeInOut" }}
                        className="absolute bottom-0 left-0 w-full bg-emerald-500/20 border-t border-emerald-400 z-20"
                    />
                )}

                {type === 'iris' && !isAnalyzed && (
                    <>
                        <motion.div 
                            animate={{ rotate: 360 }}
                            transition={{ repeat: Infinity, duration: 4, ease: "linear" }}
                            className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-40 h-40 border border-violet-500/30 rounded-full"
                        />
                         <motion.div 
                            animate={{ rotate: -360 }}
                            transition={{ repeat: Infinity, duration: 3, ease: "linear" }}
                            className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-28 h-28 border border-dashed border-violet-500/50 rounded-full"
                        />
                    </>
                )}

                {/* Technical Overlay */}
                <div className="absolute top-2 left-2 text-[8px] font-mono text-slate-500/80">
                    ID: {Math.random().toString(36).substr(2, 6).toUpperCase()}
                </div>
                
                {/* Result Overlay */}
                {isAnalyzed && (
                    <motion.div 
                        initial={{ opacity: 0 }}
                        animate={{ opacity: 1 }}
                        className="absolute inset-0 flex items-center justify-center bg-black/80 backdrop-blur-[2px] z-30"
                    >
                        <div className="text-center">
                            <div className={cn("text-4xl font-black font-mono tracking-tighter", `text-${color}-400`)}>
                                <DecryptionText text={`${score}%`} />
                            </div>
                            <div className={cn("text-[10px] font-bold uppercase tracking-widest mt-1", verdict ? "text-emerald-400" : "text-rose-400")}>
                                {verdict ? "MATCH CONFIRMED" : "NO MATCH"}
                            </div>
                        </div>
                    </motion.div>
                )}
            </div>
        </div>
    );
}

function RapidLog() {
    const [logs, setLogs] = useState<string[]>([]);
    const messages = [
        "EXTRACTING_FEATURE_VECTORS...",
        "NORMALIZING_TENSOR_DATA...",
        "SEARCHING_INDEX_SHARD_04...",
        "COMPARING_HASH_SIGNATURES...",
        "VALIDATING_BIOMETRIC_KEY...",
        "DECRYPTING_PACKET_HEADER...",
        "SYNCING_NODES_12_14...",
        "OPTIMIZING_CUDA_CORES...",
        "CACHE_MISS_RETRYING...",
        "BUFFER_FLUSH_COMPLETE."
    ];

    useEffect(() => {
        const interval = setInterval(() => {
            setLogs(prev => {
                const newLogs = [...prev, `> ${messages[Math.floor(Math.random() * messages.length)]}`];
                return newLogs.slice(-5); // Keep last 5
            });
        }, 150); // Fast update
        return () => clearInterval(interval);
    }, []);

    return (
        <div className="absolute bottom-8 left-8 p-4 font-mono text-[10px] text-cyan-500/70 border-l-2 border-cyan-500/30 bg-slate-950/50 backdrop-blur-sm w-64 h-32 overflow-hidden pointer-events-none">
            <div className="flex flex-col justify-end h-full">
                {logs.map((log, i) => (
                    <div key={i} className="opacity-70">{log}</div>
                ))}
                <div className="animate-pulse">_</div>
            </div>
        </div>
    );
}

function FinalVerdict({ result, mode }: { result: AnalysisResult | null, mode: 'verify' | 'identify' }) {
    if (!result) return null; 

    const isVerified = result.status === 'granted' || result.status === 'identified';
    
    let verdictText = "";
    if (mode === 'verify') {
        verdictText = isVerified ? "ACCESS GRANTED" : "ACCESS DENIED";
    } else {
        verdictText = isVerified ? "SUBJECT IDENTIFIED" : "NO MATCH FOUND";
    }
    
    return (
        <>
            {/* Flash Effect */}
            <motion.div 
                initial={{ opacity: 1 }}
                animate={{ opacity: 0 }}
                transition={{ duration: 0.5, ease: "easeOut" }}
                className="absolute inset-0 bg-white pointer-events-none z-50 mix-blend-overlay"
            />

            <motion.div
                 initial={{ opacity: 0, scale: 0.8 }}
                 animate={{ opacity: 1, scale: 1 }}
                 className="flex flex-col items-center gap-8 text-center relative z-40"
            >
                 <div className={cn(
                     "w-48 h-48 rounded-full flex items-center justify-center border-[6px] shadow-[0_0_100px_currentColor] bg-black/50 backdrop-blur-lg relative",
                     isVerified ? "border-emerald-500 text-emerald-500" : "border-rose-500 text-rose-500"
                 )}>
                     {isVerified ? <ShieldCheck className="w-24 h-24" /> : <ShieldAlert className="w-24 h-24" />}
                     
                     {/* Orbital Rings */}
                     <motion.div 
                        animate={{ rotate: 360 }} 
                        transition={{ duration: 10, ease: "linear", repeat: Infinity }}
                        className="absolute inset-[-20px] border border-dashed border-current rounded-full opacity-30" 
                     />
                     <motion.div 
                        animate={{ rotate: -360 }} 
                        transition={{ duration: 15, ease: "linear", repeat: Infinity }}
                        className="absolute inset-[-40px] border border-current rounded-full opacity-10" 
                     />
                 </div>

                 <div className="flex flex-col gap-2">
                     <h1 className={cn("text-5xl font-black font-mono tracking-tighter uppercase drop-shadow-[0_0_15px_rgba(0,0,0,0.8)]", isVerified ? "text-emerald-400" : "text-rose-400")}>
                         <DecryptionText text={verdictText} speed={40} revealSpeed={80} />
                     </h1>
                     <p className="text-slate-400 font-mono tracking-[0.3em] text-sm uppercase">
                         CONFIDENCE: {(result.score * 100).toFixed(1)}%
                     </p>
                 </div>
                 
                 <div className="animate-bounce mt-12 text-slate-500 text-xs font-mono tracking-widest">
                     // SYSTEM_RESET_PENDING...
                 </div>
            </motion.div>
        </>
    );
}
