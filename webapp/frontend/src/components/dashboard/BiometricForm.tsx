import React, { useState, useEffect, useRef, useMemo } from "react";
import { Card } from "@/components/ui/Card";
import { Eye, Fingerprint, User, RefreshCw, CheckCircle2, XCircle, Search, FlaskConical, Camera, Upload, ScanFace, FileUp, SlidersHorizontal, Settings2, Activity, Zap, Lock } from "lucide-react";
import { InfoTip } from "@/components/ui/InfoTip";
import { Select } from "@/components/ui/Select";
import { getDisplayName } from "@/lib/identityMap";
import { cn } from "@/lib/utils";
import { getApiBase } from "@/lib/api";
import { TechFrame } from "@/components/ui/TechFrame";
import { VerificationSequence } from "./VerificationSequence";
import { AnalysisResult } from "./ResultsPanel";

interface BiometricFormProps {
  mode: "verify" | "identify";
  identities: string[];
  onSubmit: (formData: FormData) => void;
  isProcessing: boolean;
  onPresetLoad?: (type: "authorized" | "impostor" | "identify" | "stress" | "random") => void;
  result: AnalysisResult | null;
}

function getRandomElement<T>(arr: T[]): T {
    return arr[Math.floor(Math.random() * arr.length)];
}

export function BiometricForm({ mode, identities, onSubmit, isProcessing, onPresetLoad, result }: BiometricFormProps) {
  const [claim, setClaim] = useState<string>("");
  const [modalities, setModalities] = useState({
    face: { enabled: true, file: null as File | null, preview: null as string | null, sourceId: "", inputType: 'upload' as 'upload' | 'camera' },
    fingerprint: { enabled: true, file: null as File | null, preview: null as string | null, sourceId: "", inputType: 'upload' as 'upload' | 'camera' },
    irisNir: { enabled: true, file: null as File | null, preview: null as string | null, sourceId: "", inputType: 'upload' as 'upload' | 'camera' },
    irisVis: { enabled: true, file: null as File | null, preview: null as string | null, sourceId: "", inputType: 'upload' as 'upload' | 'camera' },
  });
  
  const [protectionKey, setProtectionKey] = useState("");
  const [showSequence, setShowSequence] = useState(false);

  const faceRef = useRef<HTMLInputElement>(null);
  const fingerprintRef = useRef<HTMLInputElement>(null);
  const irisNirRef = useRef<HTMLInputElement>(null);
  const irisVisRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    if (identities.length > 0 && !claim) {
      setClaim(identities[0]);
    }
  }, [identities, claim]);

  const handleFileChange = (type: keyof typeof modalities, file: File, sourceId: string = "Manual") => {
    const reader = new FileReader();
    reader.onload = (e) => {
      setModalities(prev => ({
        ...prev,
        [type]: { ...prev[type], file, preview: e.target?.result as string, sourceId, inputType: 'upload' }
      }));
    };
    reader.readAsDataURL(file);
  };

  // Helper to fetch file without complex state updates
  const fetchFile = async (url: string, filename: string): Promise<File | null> => {
      try {
          const resp = await fetch(url);
          if (resp.ok) {
              const blob = await resp.blob();
              return new File([blob], filename, { type: blob.type });
          }
      } catch(e) { console.error(e); }
      return null;
  };

  const handleCameraCapture = (type: keyof typeof modalities, file: File) => {
      handleFileChange(type, file, "Live_Camera");
  };

  const loadIdentityImages = async (id: string) => {
    const apiBase = getApiBase();
    const sources = {
        face: `${apiBase}/static/gallery/${id}/face.jpg`,
        fingerprint: `${apiBase}/static/gallery/${id}/fingerprint.bmp`,
        irisNir: `${apiBase}/static/gallery/${id}/iris_nir.jpg`,
        irisVis: `${apiBase}/static/gallery/${id}/iris_vis.jpg`
    };

    for (const [key, url] of Object.entries(sources)) {
        try {
            const resp = await fetch(url);
            if (!resp.ok) continue;
            const blob = await resp.blob();
            const file = new File([blob], url.split('/').pop()!, { type: blob.type });
            handleFileChange(key as keyof typeof modalities, file, id);
        } catch (e) {
            console.error(`Failed to load ${key}`, e);
        }
    }
  };

  const handleClaimChange = (newClaim: string) => {
      setClaim(newClaim);
      loadIdentityImages(newClaim);
  };

  const identityOptions = useMemo(() => {
    return identities.map(id => ({
        value: id,
        label: getDisplayName(id),
        subLabel: id.replace('user_', 'SUB_')
    }));
  }, [identities]);

  const loadPreset = async (type: string) => {
    if (onPresetLoad) onPresetLoad(type as any);

    // Reset all files first and ensure all modalities are enabled by default
    setModalities(prev => ({
        face: { ...prev.face, file: null, preview: null, sourceId: "", inputType: 'upload', enabled: true },
        fingerprint: { ...prev.fingerprint, file: null, preview: null, sourceId: "", inputType: 'upload', enabled: true },
        irisNir: { ...prev.irisNir, file: null, preview: null, sourceId: "", inputType: 'upload', enabled: true },
        irisVis: { ...prev.irisVis, file: null, preview: null, sourceId: "", inputType: 'upload', enabled: true },
    }));

    const apiBase = getApiBase();
    
    // --- BASIC PRESETS ---
    
    if (type === 'stress') {
        const types = ['face', 'fingerprint', 'irisNir', 'irisVis'] as const;
        for (const t of types) {
             const randomId = getRandomElement(identities);
             const urlMap: Record<string, string> = {
                 face: 'face.jpg', fingerprint: 'fingerprint.bmp', irisNir: 'iris_nir.jpg', irisVis: 'iris_vis.jpg'
             };
             const url = `${apiBase}/static/gallery/${randomId}/${urlMap[t]}`;
             try {
                const resp = await fetch(url);
                if (resp.ok) {
                    const blob = await resp.blob();
                    const file = new File([blob], urlMap[t], { type: blob.type });
                    handleFileChange(t, file, randomId);
                }
             } catch(e) { console.error(e) }
        }
        return;
    }

    if (type === 'random') {
        const types = ['face', 'fingerprint', 'irisNir', 'irisVis'] as const;
        if (mode === 'verify') {
             const randomId = getRandomElement(identities);
             setClaim(randomId);
        } else {
             setClaim(""); 
        }
        for (const t of types) {
             try {
                const resp = await fetch(`${apiBase}/api/v1/random-sample?modality=${t}`);
                if (resp.ok) {
                    const blob = await resp.blob();
                    const ext = blob.type.split('/')[1] || 'dat';
                    const file = new File([blob], `random_${t}.${ext}`, { type: blob.type });
                    handleFileChange(t, file, "Unknown_Source");
                }
             } catch(e) { console.error(e) }
        }
        return;
    }

    if (type === 'identify') {
        return loadPreset('random');
    }

    // --- NEW SCI-FI PRESETS ---

    if (type === 'cyclops_mode') { // Iris Only
        const id = getRandomElement(identities);
        setClaim(id);
        const sources = {
            irisNir: `${apiBase}/static/gallery/${id}/iris_nir.jpg`,
            irisVis: `${apiBase}/static/gallery/${id}/iris_vis.jpg`
        };
        // Disable face/finger
        setModalities(p => ({
            ...p,
            face: { ...p.face, enabled: false },
            fingerprint: { ...p.fingerprint, enabled: false },
            irisNir: { ...p.irisNir, enabled: true },
            irisVis: { ...p.irisVis, enabled: true }
        }));
        
        for (const [key, url] of Object.entries(sources)) {
            try {
                const resp = await fetch(url);
                if (resp.ok) {
                    const blob = await resp.blob();
                    handleFileChange(key as any, new File([blob], url.split('/').pop()!, {type:blob.type}), id);
                }
            } catch(e) {}
        }
        return;
    }

    if (type === 'stealth_entry') { // Fingerprint Only
        const id = getRandomElement(identities);
        setClaim(id);
        const url = `${apiBase}/static/gallery/${id}/fingerprint.bmp`;
        setModalities(p => ({
            ...p,
            face: { ...p.face, enabled: false },
            fingerprint: { ...p.fingerprint, enabled: true },
            irisNir: { ...p.irisNir, enabled: false },
            irisVis: { ...p.irisVis, enabled: false }
        }));
        try {
            const resp = await fetch(url);
            if (resp.ok) {
                const blob = await resp.blob();
                handleFileChange('fingerprint', new File([blob], 'fingerprint.bmp', {type:blob.type}), id);
            }
        } catch(e) {}
        return;
    }

    if (type === 'partial_face') { // Face Only
        const id = getRandomElement(identities);
        setClaim(id);
        const url = `${apiBase}/static/gallery/${id}/face.jpg`;
        setModalities(p => ({
            ...p,
            face: { ...p.face, enabled: true },
            fingerprint: { ...p.fingerprint, enabled: false },
            irisNir: { ...p.irisNir, enabled: false },
            irisVis: { ...p.irisVis, enabled: false }
        }));
        try {
            const resp = await fetch(url);
            if (resp.ok) {
                const blob = await resp.blob();
                handleFileChange('face', new File([blob], 'face.jpg', {type:blob.type}), id);
            }
        } catch(e) {}
        return;
    }

    // Default Fallback (Authorized / Impostor)
    const randomId = getRandomElement(identities);
    let sourceId = randomId;
    const claimId = randomId;

    if (type === 'impostor') {
        const others = identities.filter(id => id !== claimId);
        sourceId = getRandomElement(others);
    }
    
    setClaim(claimId);
    await loadIdentityImages(sourceId);
  };

  const handleSubmit = () => {
    const formData = new FormData();
    if (mode === 'verify') {
        formData.append('identity', claim);
        if (protectionKey) formData.append('protection_key', protectionKey);
    }
    
    if (modalities.face.enabled && modalities.face.file) formData.append('face_file', modalities.face.file);
    if (modalities.fingerprint.enabled && modalities.fingerprint.file) formData.append('fingerprint_file', modalities.fingerprint.file);
    if (modalities.irisNir.enabled && modalities.irisNir.file) formData.append('iris_nir_file', modalities.irisNir.file);
    if (modalities.irisVis.enabled && modalities.irisVis.file) formData.append('iris_vis_file', modalities.irisVis.file);

    setShowSequence(true);
    onSubmit(formData);
  };

  return (
    <Card variant="tech" className="flex flex-col bg-white min-h-full relative overflow-hidden shadow-[0_0_20px_rgba(6,182,212,0.1)] border-cyan-500/30" noPadding>
      {/* Border Glow Container - Pseudo-element simulation for pulsing border */}
      <div className="absolute inset-0 border-2 border-cyan-500/0 pointer-events-none rounded-sm animate-[border-pulse_4s_ease-in-out_infinite]" />
      
      {showSequence && (
          <VerificationSequence 
            mode={mode}
            modalities={modalities} 
            result={result} 
            isProtected={!!protectionKey}
            onComplete={() => setShowSequence(false)} 
          />
      )}
      
      {/* ... header ... */}
      <div className="flex flex-col gap-2 p-3 border-b border-slate-200 bg-slate-50/50 shrink-0">
          <div className="flex justify-between items-center">
             <div className="text-xs font-bold text-slate-700 font-mono uppercase tracking-widest flex items-center gap-2">
                 <div className="p-0.5 border border-slate-300 bg-white text-cyan-600">
                    <Activity className="w-3.5 h-3.5" />
                 </div>
                 <InfoTip content="Configure active biometric sensors and input sources.">SENSOR_INPUT_ARRAY</InfoTip>
             </div>
             <span className="text-[9px] text-cyan-600 font-bold font-mono bg-cyan-50 px-2 py-0.5 border border-cyan-200 flex items-center gap-2">
                <InfoTip content="System operational. All sensors ready." placement="bottom">STATUS: ONLINE</InfoTip>
             </span>
          </div>
          
          {/* Integrated Toolbar - SCROLLABLE NOW */}
          <div className="flex gap-1.5 items-center overflow-x-auto pb-1 no-scrollbar w-full">
             <span className="text-[9px] font-bold text-slate-400 uppercase tracking-wider mr-1 shrink-0 flex items-center gap-1">
                <InfoTip content="Quickly load pre-configured test scenarios.">PRESETS:</InfoTip>
             </span>
             
             {/* Standard Group */}
             <InfoTip content="Standard authorized test.">
                 <button onClick={() => loadPreset('authorized')} className="shrink-0 px-2 py-1 bg-white border border-slate-300 hover:border-emerald-400 hover:text-emerald-700 text-slate-600 rounded-sm transition-all text-[9px] font-bold uppercase tracking-wider">
                     [ AUTH ]
                 </button>
             </InfoTip>
             <InfoTip content="Standard unauthorized test.">
                 <button onClick={() => loadPreset('impostor')} className="shrink-0 px-2 py-1 bg-white border border-slate-300 hover:border-rose-400 hover:text-rose-700 text-slate-600 rounded-sm transition-all text-[9px] font-bold uppercase tracking-wider">
                     [ IMPOSTOR ]
                 </button>
             </InfoTip>
             
             <div className="w-[1px] h-4 bg-slate-300 shrink-0 mx-1" />

             {/* Scenarios Group */}
             <InfoTip content="Iris Only">
                 <button onClick={() => loadPreset('cyclops_mode')} className="shrink-0 px-2 py-1 bg-slate-50 border border-slate-300 hover:border-violet-400 hover:text-violet-700 text-slate-600 rounded-sm transition-all text-[9px] font-bold uppercase tracking-wider">
                     [ CYCLOPS ]
                 </button>
             </InfoTip>
             <InfoTip content="Fingerprint Only">
                 <button onClick={() => loadPreset('stealth_entry')} className="shrink-0 px-2 py-1 bg-slate-50 border border-slate-300 hover:border-emerald-400 hover:text-emerald-700 text-slate-600 rounded-sm transition-all text-[9px] font-bold uppercase tracking-wider">
                     [ STEALTH ]
                 </button>
             </InfoTip>
             <InfoTip content="Face Only">
                 <button onClick={() => loadPreset('partial_face')} className="shrink-0 px-2 py-1 bg-slate-50 border border-slate-300 hover:border-cyan-400 hover:text-cyan-700 text-slate-600 rounded-sm transition-all text-[9px] font-bold uppercase tracking-wider">
                     [ PARTIAL ]
                 </button>
             </InfoTip>

             <div className="w-[1px] h-4 bg-slate-300 shrink-0 mx-1" />

             {/* Stress/Random Group */}
             <InfoTip content="Random Identity/Modality">
                 <button onClick={() => loadPreset('random')} className="shrink-0 px-2 py-1 bg-white border border-slate-300 hover:border-cyan-400 hover:text-cyan-700 text-slate-600 rounded-sm transition-all text-[9px] font-bold uppercase tracking-wider">
                     [ RND ]
                 </button>
             </InfoTip>
             <InfoTip content="Stress Test System">
                 <button onClick={() => loadPreset('stress')} className="shrink-0 px-2 py-1 bg-amber-50 border border-amber-200 hover:border-amber-400 text-amber-800 rounded-sm transition-all text-[9px] font-bold uppercase tracking-wider flex items-center gap-1">
                    <FlaskConical className="w-3 h-3" /> STRESS
                 </button>
             </InfoTip>
          </div>
      </div>

      <div className="flex flex-col flex-1">
        {/* Claim Identity (Verify Only) */}
        <div className="p-3 bg-white border-b border-slate-100 flex items-center gap-3 shrink-0 z-20 relative">
            <label className="text-[10px] font-bold text-slate-500 uppercase font-mono tracking-wider w-20 shrink-0 flex items-center gap-1">
                <InfoTip content="Select the identity claim to verify against.">
                    {mode === 'verify' ? "SUBJECT_ID" : "TARGET_LOAD"}
                </InfoTip>
            </label>
            <div className="relative flex-1">
                <Select 
                    value={claim}
                    onChange={handleClaimChange}
                    options={identityOptions}
                    placeholder={mode === 'verify' ? "SELECT_ID..." : "LOAD_PATTERN..."}
                />
            </div>
            
            {/* Protection Key Input */}
            {mode === 'verify' && (
                <div className="flex items-center gap-2 border-l border-slate-200 pl-3">
                    <label className="text-[10px] font-bold text-slate-400 uppercase font-mono tracking-wider flex items-center gap-1">
                        <Lock className="w-3 h-3" />
                        <InfoTip content="Enter a secret key to enable Template Protection (BioHashing).">KEY</InfoTip>
                    </label>
                    <input 
                        type="password"
                        value={protectionKey}
                        onChange={(e) => setProtectionKey(e.target.value)}
                        placeholder="OPTIONAL..."
                        className="w-24 bg-slate-50 border border-slate-200 rounded-sm px-2 py-1 text-[10px] font-mono focus:outline-none focus:border-cyan-400 text-slate-700 placeholder:text-slate-300"
                    />
                </div>
            )}
        </div>

        {/* Biometric Modalities Grid - Symmetrical & Auto-Height */}
        <div className="grid grid-cols-2 gap-2 p-2">
                <ModalityInput 
                    label={<InfoTip content="Facial recognition sensor">FACE</InfoTip>}
                    icon={<ScanFace className="w-3.5 h-3.5" />}
                    data={modalities.face} 
                    inputRef={faceRef}
                    onToggle={(val) => setModalities(p => ({ ...p, face: { ...p.face, enabled: val } }))}
                    onFile={(f) => handleFileChange('face', f)}
                    onCapture={(f) => handleCameraCapture('face', f)}
                    onModeChange={(type) => setModalities(p => ({ ...p, face: { ...p.face, inputType: type } }))}
                    supportsCamera={true}
                    color="cyan"
                    variant="large"
                />
                <ModalityInput 
                    label={<InfoTip content="Fingerprint scanner">FINGERPRINT</InfoTip>}
                    icon={<Fingerprint className="w-3.5 h-3.5" />}
                    data={modalities.fingerprint} 
                    inputRef={fingerprintRef}
                    onToggle={(val) => setModalities(p => ({ ...p, fingerprint: { ...p.fingerprint, enabled: val } }))}
                    onFile={(f) => handleFileChange('fingerprint', f)}
                    onCapture={(f) => handleCameraCapture('fingerprint', f)}
                    onModeChange={(type) => setModalities(p => ({ ...p, fingerprint: { ...p.fingerprint, inputType: type } }))}
                    supportsCamera={false}
                    color="blue"
                    variant="large"
                />
                <ModalityInput 
                    label={<InfoTip content="Near-Infrared iris sensor">IRIS (NIR)</InfoTip>}
                    icon={<Eye className="w-3.5 h-3.5" />}
                    data={modalities.irisNir} 
                    inputRef={irisNirRef}
                    onToggle={(val) => setModalities(p => ({ ...p, irisNir: { ...p.irisNir, enabled: val } }))}
                    onFile={(f) => handleFileChange('irisNir', f)}
                    onCapture={(f) => handleCameraCapture('irisNir', f)}
                    onModeChange={(type) => setModalities(p => ({ ...p, irisNir: { ...p.irisNir, inputType: type } }))}
                    supportsCamera={false}
                    color="cyan"
                    variant="large"
                />
                <ModalityInput 
                    label={<InfoTip content="Visible spectrum iris sensor">IRIS (VIS)</InfoTip>}
                    icon={<Eye className="w-3.5 h-3.5" />}
                    data={modalities.irisVis} 
                    inputRef={irisVisRef}
                    onToggle={(val) => setModalities(p => ({ ...p, irisVis: { ...p.irisVis, enabled: val } }))}
                    onFile={(f) => handleFileChange('irisVis', f)}
                    onCapture={(f) => handleCameraCapture('irisVis', f)}
                    onModeChange={(type) => setModalities(p => ({ ...p, irisVis: { ...p.irisVis, inputType: type } }))}
                    supportsCamera={false}
                    color="cyan"
                    variant="large"
                />
        </div>
      </div>

      <div className="p-3 border-t border-slate-200 shrink-0 bg-slate-50/30 relative z-30">
          <button
              onClick={handleSubmit}
              disabled={isProcessing}
              className={cn(
                  "w-full py-3 font-bold text-white shadow-md shadow-cyan-500/10 transition-all flex items-center justify-center gap-2 text-[10px] font-mono uppercase tracking-widest relative overflow-hidden group hover:-translate-y-0.5 border border-cyan-400",
                  isProcessing 
                      ? "bg-slate-400 cursor-not-allowed shadow-none border-slate-400" 
                      : "bg-cyan-600 hover:bg-cyan-500"
              )}
          >
              <div className="absolute top-0 left-0 w-full h-[1px] bg-white/30" />
              
              {isProcessing ? (
                  <>
                      <RefreshCw className="w-3.5 h-3.5 animate-spin text-white" /> <span>PROCESSING...</span>
                  </>
              ) : (
                  mode === 'verify' ? ">> VERIFY_IDENTITY" : ">> IDENTIFY_SUBJECT"
              )}
          </button>
      </div>
    </Card>
  );
}

interface ModalityData {
    enabled: boolean;
    file: File | null;
    preview: string | null;
    sourceId: string;
    inputType: 'upload' | 'camera';
}

interface ModalityInputProps {
    label: React.ReactNode;
    icon: React.ReactNode;
    data: ModalityData;
    onToggle: (val: boolean) => void;
    onFile: (file: File) => void;
    onCapture: (file: File) => void;
    onModeChange: (type: 'upload' | 'camera') => void;
    inputRef: React.RefObject<HTMLInputElement | null>;
    supportsCamera: boolean;
    color: 'blue' | 'indigo' | 'violet' | 'emerald' | 'cyan';
    variant?: 'standard' | 'large';
}

function ModalityInput({ label, icon, data, onToggle, onFile, onCapture, onModeChange, inputRef, supportsCamera, color, variant = 'standard' }: ModalityInputProps) {
    const [isStreaming, setIsStreaming] = useState(false);
    const videoRef = useRef<HTMLVideoElement>(null);

    const startCamera = async () => {
        if (!data.enabled) return;
        try {
            const stream = await navigator.mediaDevices.getUserMedia({ video: true });
            if (videoRef.current) {
                videoRef.current.srcObject = stream;
                setIsStreaming(true);
            }
        } catch (e) {
            console.error("Camera access denied", e);
            alert("Could not access camera. Please allow permissions.");
        }
    };

    const stopCamera = () => {
        if (videoRef.current && videoRef.current.srcObject) {
            const stream = videoRef.current.srcObject as MediaStream;
            stream.getTracks().forEach(track => track.stop());
            videoRef.current.srcObject = null;
        }
        setIsStreaming(false);
    };

    const captureImage = () => {
        if (!videoRef.current) return;
        
        const canvas = document.createElement('canvas');
        canvas.width = videoRef.current.videoWidth;
        canvas.height = videoRef.current.videoHeight;
        const ctx = canvas.getContext('2d');
        if (ctx) {
            ctx.drawImage(videoRef.current, 0, 0);
            canvas.toBlob((blob) => {
                if (blob) {
                    const file = new File([blob], "camera_capture.jpg", { type: "image/jpeg" });
                    onCapture(file);
                    stopCamera();
                }
            }, 'image/jpeg');
        }
    };

    useEffect(() => {
        if (!data.enabled) stopCamera();
        return () => stopCamera();
    }, [data.enabled]);

    return (
        <TechFrame 
            color={color}
            className={cn(
                "flex flex-col gap-1 p-2 transition-all duration-300 bg-white h-full", 
                variant === 'large' ? "min-h-[240px]" : "min-h-[140px]",
                data.enabled ? "opacity-100 shadow-sm border-slate-300" : "opacity-60 grayscale bg-slate-50 border-slate-200"
            )}
        >
             {/* Header */}
             <div className="flex justify-between items-center shrink-0 h-6 border-b border-slate-100 pb-1 mb-1">
                  <div className="flex items-center gap-2 text-slate-700">
                      <span className="p-0.5 bg-slate-100 border border-slate-200 text-slate-500 rounded-[1px]">{icon}</span>
                      <span className="text-[10px] font-bold font-mono uppercase tracking-widest">{label}</span>
                  </div>
                  
                  <div className="relative">
                     <input 
                         type="checkbox" 
                         checked={data.enabled} 
                         onChange={(e) => onToggle(e.target.checked)}
                         className="peer appearance-none w-9 h-5 bg-slate-200 rounded-full checked:bg-cyan-500 cursor-pointer transition-all shadow-inner hover:bg-slate-300 checked:shadow-[0_0_10px_rgba(6,182,212,0.4)]"
                     />
                     <div className="absolute top-0.5 left-0.5 w-4 h-4 bg-white rounded-full peer-checked:translate-x-4 transition-transform pointer-events-none shadow-sm" />
                  </div>
             </div>

             {variant === 'large' ? (
                 // SPLIT LAYOUT (Horizontal)
                 <div className="flex-1 flex min-h-0 gap-2">
                     {/* Preview Area (Left - Flexible) */}
                     <div className="flex-[3] relative border border-slate-200 bg-slate-100 overflow-hidden group rounded-[2px]">
                         {/* Technical overlay corners */}
                         {data.enabled && (
                             <>
                                <div className="absolute top-0 left-0 w-2 h-2 border-t-2 border-l-2 border-cyan-500/30 z-20" />
                                <div className="absolute top-0 right-0 w-2 h-2 border-t-2 border-r-2 border-cyan-500/30 z-20" />
                                <div className="absolute bottom-0 left-0 w-2 h-2 border-b-2 border-l-2 border-cyan-500/30 z-20" />
                                <div className="absolute bottom-0 right-0 w-2 h-2 border-b-2 border-r-2 border-cyan-500/30 z-20" />
                             </>
                         )}

                         <div 
                            onClick={() => data.enabled && data.inputType === 'upload' && inputRef.current?.click()}
                            className={cn(
                                "w-full h-full flex items-center justify-center transition-all absolute inset-0 overflow-hidden",
                                data.inputType === 'upload' && data.enabled ? "cursor-pointer hover:bg-slate-900/50" : ""
                            )}
                         >
                             {isStreaming ? (
                                 <video ref={videoRef} autoPlay playsInline muted className="w-full h-full object-contain transform scale-x-[-1] relative z-10" />
                             ) : data.preview ? (
                                <>
                                    <div 
                                        className="absolute inset-0 bg-cover bg-center opacity-30 blur-xl scale-125 grayscale-[20%]"
                                        style={{ backgroundImage: `url(${data.preview})` }}
                                    />
                                    <img src={data.preview} alt="Preview" className="w-full h-full object-contain relative z-10 drop-shadow-sm" />
                                </>
                             ) : (
                                 <div className="flex flex-col items-center gap-1 text-slate-600">
                                     {data.enabled ? <div className="text-[9px] font-mono font-bold tracking-widest text-slate-700">NO_SIGNAL</div> : null}
                                 </div>
                             )}
                         </div>
                     </div>

                     {/* Controls (Right - Stacked) */}
                     <div className="flex-1 flex flex-col gap-1 min-w-[70px]">
                        <button 
                            onClick={() => inputRef.current?.click()}
                            disabled={!data.enabled || data.inputType === 'camera'}
                            className="flex-1 border border-slate-200 bg-slate-50 hover:bg-white hover:border-slate-300 text-slate-600 rounded-[2px] text-[9px] font-bold uppercase tracking-wider transition-colors flex flex-col items-center justify-center gap-1"
                        >
                            <FileUp className="w-3.5 h-3.5 text-slate-400" />
                            <InfoTip content="Upload file from local system">FILE</InfoTip>
                        </button>
                        {supportsCamera && (
                            <button 
                                onClick={() => {
                                    if (isStreaming) captureImage();
                                    else startCamera();
                                    onModeChange('camera');
                                }}
                                disabled={!data.enabled}
                                className={cn(
                                    "flex-1 border text-[9px] font-bold uppercase tracking-wider transition-colors flex flex-col items-center justify-center gap-1 rounded-[2px]",
                                    isStreaming 
                                        ? "bg-rose-50 border-rose-200 text-rose-600 animate-pulse" 
                                        : "bg-slate-50 border-slate-200 text-slate-600 hover:bg-white hover:border-slate-300"
                                )}
                            >
                                <Camera className={cn("w-3.5 h-3.5", isStreaming ? "text-rose-500" : "text-slate-400")} />
                                <InfoTip content={isStreaming ? "Capture image" : "Activate camera"}>
                                    {isStreaming ? "SNAP" : "CAM"}
                                </InfoTip>
                            </button>
                        )}
                     </div>
                 </div>
             ) : (
                 // STANDARD LAYOUT (Vertical) - Iris
                 <>
                     <div className="flex-1 min-h-0 relative border border-slate-200 bg-slate-50/50 overflow-hidden group my-1 rounded-[2px]">
                         {data.enabled && (
                             <>
                                <div className="absolute top-0 left-0 w-1.5 h-1.5 border-t border-l border-cyan-500/30 z-20" />
                                <div className="absolute top-0 right-0 w-1.5 h-1.5 border-t border-r border-cyan-500/30 z-20" />
                                <div className="absolute bottom-0 left-0 w-1.5 h-1.5 border-b border-l border-cyan-500/30 z-20" />
                                <div className="absolute bottom-0 right-0 w-1.5 h-1.5 border-b border-r border-cyan-500/30 z-20" />
                             </>
                         )}

                         <div 
                            onClick={() => data.enabled && data.inputType === 'upload' && inputRef.current?.click()}
                            className={cn(
                                "w-full h-full flex items-center justify-center transition-all absolute inset-0 overflow-hidden",
                                data.inputType === 'upload' && data.enabled ? "cursor-pointer hover:bg-slate-200/50" : ""
                            )}
                         >
                            {isStreaming ? (
                                 <video ref={videoRef} autoPlay playsInline muted className="w-full h-full object-contain transform scale-x-[-1] relative z-10" />
                            ) : data.preview ? (
                                <>
                                    {/* Blurred Backdrop */}
                                    <div 
                                        className="absolute inset-0 bg-cover bg-center opacity-20 blur-xl scale-125"
                                        style={{ backgroundImage: `url(${data.preview})` }}
                                    />
                                    {/* Main Image */}
                                    <img src={data.preview} alt="Preview" className="w-full h-full object-contain relative z-10" />
                                </>
                            ) : (
                                 <div className="flex flex-col items-center gap-1 text-slate-400">
                                     {data.enabled ? <div className="text-[8px] font-mono font-bold tracking-widest text-slate-500">NO_SIGNAL</div> : null}
                                 </div>
                            )}
                         </div>
                     </div>

                     <div className="flex gap-2 shrink-0">
                        <button 
                            onClick={() => inputRef.current?.click()}
                            disabled={!data.enabled || data.inputType === 'camera'}
                            className="flex-1 py-1.5 border border-slate-200 bg-slate-50 hover:bg-white hover:border-slate-300 text-slate-600 rounded-[2px] text-[9px] font-bold uppercase tracking-wider transition-colors"
                        >
                            <InfoTip content="Upload file from local system">FILE</InfoTip>
                        </button>
                        <button 
                            onClick={() => {
                                if (isStreaming) captureImage();
                                else startCamera();
                                onModeChange('camera');
                            }}
                            disabled={!data.enabled || !supportsCamera}
                            className={cn(
                                "flex-1 py-1.5 border text-[9px] font-bold uppercase tracking-wider transition-colors rounded-[2px]",
                                isStreaming 
                                    ? "bg-rose-50 border-rose-200 text-rose-600 animate-pulse" 
                                    : "bg-slate-50 border-slate-200 text-slate-600 hover:bg-white hover:border-slate-300"
                            )}
                        >
                            <InfoTip content={isStreaming ? "Capture image" : "Activate camera"}>
                                {isStreaming ? "SNAP" : "CAM"}
                            </InfoTip>
                        </button>
                        <button className="flex-1 py-1.5 border border-slate-200 bg-slate-50 text-slate-300 cursor-not-allowed rounded-[2px] text-[9px] font-bold uppercase tracking-wider">
                            BROWSE
                        </button>
                     </div>
                 </>
             )}

             <input type="file" ref={inputRef} className="hidden" onChange={(e) => e.target.files?.[0] && onFile(e.target.files[0])} accept="image/*" />
        </TechFrame>
    );
}
