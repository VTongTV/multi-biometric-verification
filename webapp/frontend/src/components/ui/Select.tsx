import React from "react";
import { cn } from "@/lib/utils";
import { ChevronDown, Search, Check, User } from "lucide-react";
import { motion, AnimatePresence } from "framer-motion";

interface Option {
  value: string;
  label: string;
  subLabel?: string;
}

interface SelectProps {
  value: string;
  onChange: (value: string) => void;
  options: Option[];
  placeholder?: string;
  className?: string;
  disabled?: boolean;
}

export function Select({ value, onChange, options, placeholder = "Select...", className, disabled }: SelectProps) {
  const [isOpen, setIsOpen] = useState(false);
  const [search, setSearch] = useState("");
  const containerRef = useRef<HTMLDivElement>(null);

  // Close when clicking outside
  useEffect(() => {
    const handleClickOutside = (event: MouseEvent) => {
      if (containerRef.current && !containerRef.current.contains(event.target as Node)) {
        setIsOpen(false);
      }
    };
    document.addEventListener("mousedown", handleClickOutside);
    return () => document.removeEventListener("mousedown", handleClickOutside);
  }, []);

  const selectedOption = options.find(opt => opt.value === value);

  const filteredOptions = options.filter(opt => 
    opt.label.toLowerCase().includes(search.toLowerCase()) || 
    opt.subLabel?.toLowerCase().includes(search.toLowerCase()) ||
    opt.value.toLowerCase().includes(search.toLowerCase())
  );

// Select.tsx - Reverting to Tech style

  return (
    <div className={cn("relative text-sm font-mono", className)} ref={containerRef}>
      {/* Trigger Button */}
      <button
        type="button"
        onClick={() => !disabled && setIsOpen(!isOpen)}
        className={cn(
          "w-full flex items-center justify-between px-3 py-2 bg-white border shadow-sm transition-all outline-none",
          isOpen ? "border-cyan-500 text-cyan-700" : "border-slate-300 text-slate-600 hover:border-slate-400 hover:bg-slate-50",
          disabled && "opacity-50 cursor-not-allowed bg-slate-100"
        )}
        disabled={disabled}
      >
        <span className={cn("truncate flex items-center gap-2", !selectedOption && "text-slate-400")}>
          {selectedOption ? (
            <>
               <span className="font-bold tracking-tight">{selectedOption.label}</span>
               {selectedOption.subLabel && (
                   <span className="text-[10px] text-slate-400 font-normal hidden sm:inline">[{selectedOption.subLabel}]</span>
               )}
            </>
          ) : (
            placeholder
          )}
        </span>
        <ChevronDown className={cn("w-4 h-4 text-slate-400 transition-transform", isOpen && "rotate-180 text-cyan-500")} />
      </button>

      {/* Dropdown Menu */}
      <AnimatePresence>
        {isOpen && (
          <motion.div
            initial={{ opacity: 0, y: 5 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: 5 }}
            transition={{ duration: 0.1 }}
            className="absolute z-50 w-full mt-1 bg-white border border-slate-300 shadow-xl overflow-hidden flex flex-col max-h-[300px]"
          >
            {/* Search Input */}
            <div className="p-2 border-b border-slate-200 sticky top-0 bg-slate-50">
                <div className="relative">
                    <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-3 h-3 text-slate-400" />
                    <input 
                        type="text"
                        value={search}
                        onChange={(e) => setSearch(e.target.value)}
                        placeholder="FILTER_ID..."
                        className="w-full pl-8 pr-3 py-1.5 text-[10px] bg-white border border-slate-300 rounded-sm focus:outline-none focus:border-cyan-500 transition-colors uppercase tracking-wider font-mono"
                        autoFocus
                    />
                </div>
            </div>

            {/* Options List */}
            <div className="flex-1 overflow-y-auto custom-scrollbar">
                {filteredOptions.length > 0 ? (
                    filteredOptions.map((opt) => (
                        <button
                            key={opt.value}
                            onClick={() => {
                                onChange(opt.value);
                                setIsOpen(false);
                                setSearch("");
                            }}
                            className={cn(
                                "w-full text-left px-3 py-2 text-xs flex items-center justify-between group transition-colors border-b border-slate-100 last:border-0",
                                value === opt.value 
                                    ? "bg-cyan-50 text-cyan-800" 
                                    : "text-slate-600 hover:bg-slate-50 hover:text-slate-900"
                            )}
                        >
                            <div className="flex items-center gap-3">
                                <div className={cn(
                                    "w-4 h-4 flex items-center justify-center shrink-0 border transition-colors",
                                    value === opt.value 
                                        ? "border-cyan-300 bg-cyan-100 text-cyan-600" 
                                        : "border-slate-300 text-slate-400 group-hover:border-slate-400"
                                )}>
                                    <User className="w-2.5 h-2.5" />
                                </div>
                                <div>
                                    <div className="font-bold tracking-tight">{opt.label}</div>
                                    {opt.subLabel && <div className="text-[9px] text-slate-400 font-normal">{opt.subLabel}</div>}
                                </div>
                            </div>
                            {value === opt.value && <Check className="w-3 h-3 text-cyan-600" />}
                        </button>
                    ))
                ) : (
                    <div className="p-4 text-center text-slate-400 italic text-[10px] uppercase font-mono">
                        // NO_MATCH //
                    </div>
                )}
            </div>
            
            <div className="p-1 bg-slate-100 border-t border-slate-200 text-[9px] text-center text-slate-500 font-mono">
                {filteredOptions.length} RECS
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}

import { useState, useRef, useEffect } from "react";
