import React from "react";
import { X, User, Fingerprint, Eye, FileImage } from "lucide-react";
import { getDisplayName } from "@/lib/identityMap";
import { cn } from "@/lib/utils";
import { getApiBase } from "@/lib/api";

interface SubjectDetailModalProps {
  subjectId: string | null;
  onClose: () => void;
}

export function SubjectDetailModal({ subjectId, onClose }: SubjectDetailModalProps) {
  if (!subjectId) return null;

  const galleryPath = `${getApiBase()}/static/gallery/${subjectId}`;

  const images = [
    { type: 'Face', file: 'face.jpg', icon: User, color: 'blue' },
    { type: 'Fingerprint', file: 'fingerprint.bmp', icon: Fingerprint, color: 'violet' },
    { type: 'Iris (NIR)', file: 'iris_nir.jpg', icon: Eye, color: 'emerald' },
    { type: 'Iris (VIS)', file: 'iris_vis.jpg', icon: Eye, color: 'emerald' },
  ];

  return (
    <div className="fixed inset-0 z-[100] flex items-center justify-center bg-slate-900/50 backdrop-blur-sm p-4 animate-in fade-in duration-200">
       <div className="bg-white shadow-2xl max-w-2xl w-full flex flex-col max-h-[90vh] overflow-hidden border border-slate-400 relative">
          {/* Tech borders */}
          <div className="absolute top-0 left-0 w-4 h-4 border-t-2 border-l-2 border-cyan-500 z-50 pointer-events-none" />
          <div className="absolute top-0 right-0 w-4 h-4 border-t-2 border-r-2 border-cyan-500 z-50 pointer-events-none" />
          <div className="absolute bottom-0 left-0 w-4 h-4 border-b-2 border-l-2 border-cyan-500 z-50 pointer-events-none" />
          <div className="absolute bottom-0 right-0 w-4 h-4 border-b-2 border-r-2 border-cyan-500 z-50 pointer-events-none" />
          
          {/* Header */}
          <div className="p-6 border-b border-slate-200 flex justify-between items-center bg-slate-50 relative overflow-hidden">
             {/* Tech Decoration */}
             <div className="absolute top-0 left-0 w-full h-1 bg-gradient-to-r from-cyan-500 via-blue-500 to-cyan-500 opacity-30" />
             
             <div>
                <div className="flex items-center gap-3">
                    <div className="w-10 h-10 border border-cyan-500/30 bg-cyan-50 text-cyan-600 flex items-center justify-center relative">
                        <div className="absolute top-0 left-0 w-1 h-1 bg-cyan-500" />
                        <div className="absolute bottom-0 right-0 w-1 h-1 bg-cyan-500" />
                        <User className="w-5 h-5" />
                    </div>
                    <div>
                        <h3 className="text-xl font-bold text-slate-800 leading-tight font-mono uppercase tracking-wide">
                            Identity_Inspector
                        </h3>
                        <p className="text-xs text-slate-500 font-bold font-mono tracking-wider uppercase">
                            Target_ID: <span className="text-cyan-700">{subjectId?.replace('user_', 'SUB_')}</span>
                        </p>
                    </div>
                </div>
             </div>
             <button onClick={onClose} className="p-2 hover:bg-slate-200 transition-colors text-slate-400 hover:text-rose-500 border border-transparent hover:border-slate-300">
                <X className="w-5 h-5" />
             </button>
          </div>

          {/* Content */}
          <div className="p-8 overflow-y-auto bg-white">
             <div className="grid grid-cols-2 gap-6">
                {images.map((img) => (
                    <div key={img.file} className="bg-slate-50 p-1 border border-slate-200 hover:border-cyan-400 transition-colors group/card">
                        <div className="flex items-center gap-2 mb-2 px-2 pt-1 text-[10px] font-bold text-slate-500 uppercase tracking-widest font-mono">
                            <img.icon className={`w-3 h-3 text-${img.color}-500`} />
                            {img.type}
                        </div>
                        <div className="aspect-square bg-white relative border border-slate-200 overflow-hidden">
                             {/* Corner accents */}
                             <div className="absolute top-0 left-0 w-2 h-2 border-t border-l border-cyan-500/50 z-10" />
                             <div className="absolute bottom-0 right-0 w-2 h-2 border-b border-r border-cyan-500/50 z-10" />
                             
                             <img 
                                src={`${galleryPath}/${img.file}`} 
                                alt={img.type}
                                className="w-full h-full object-cover grayscale group-hover/card:grayscale-0 transition-all duration-500"
                                onError={(e) => {
                                    (e.target as HTMLImageElement).style.display = 'none';
                                    (e.target as HTMLImageElement).nextElementSibling?.classList.remove('hidden');
                                }}
                             />
                             <div className="hidden absolute inset-0 flex flex-col items-center justify-center text-slate-300 gap-2 bg-slate-50">
                                <FileImage className="w-8 h-8 opacity-50" />
                                <span className="text-[10px] font-bold font-mono">NO_DATA</span>
                             </div>
                        </div>
                    </div>
                ))}
             </div>
             
             <div className="mt-8 p-4 bg-slate-50 border border-slate-200 text-xs text-slate-600 font-mono relative">
                <div className="absolute left-0 top-0 bottom-0 w-1 bg-cyan-500" />
                <div className="space-y-1.5 pl-2">
                    <p><span className="font-bold text-slate-400 uppercase tracking-wider">Registration:</span> 2024-01-15</p>
                    <p><span className="font-bold text-slate-400 uppercase tracking-wider">Quality Score:</span> 98.5 [TIER_1]</p>
                    <p><span className="font-bold text-slate-400 uppercase tracking-wider">Encryption:</span> AES-256 [SECURE]</p>
                </div>
             </div>
          </div>
       </div>
    </div>
  );
}
