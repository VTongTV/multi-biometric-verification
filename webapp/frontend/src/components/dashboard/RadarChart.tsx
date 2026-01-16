import React from "react";
import { cn } from "@/lib/utils";

interface RadarChartProps {
  data: Record<string, number>;
  className?: string;
}

export function RadarChart({ data, className }: RadarChartProps) {
  // Normalize keys to 4 cardinal directions
  // Top: Face, Right: Finger, Bottom: Iris NIR, Left: Iris VIS
  const values = [
    data['face'] || 0,
    data['fingerprint'] || 0,
    data['iris_nir'] || 0,
    data['iris_vis'] || 0,
  ];

  const size = 200;
  const center = size / 2;
  const radius = (size / 2) - 30; // padding

  // Calculate polygon points
  const points = values.map((val, i) => {
    const angle = (Math.PI / 2) * i - (Math.PI / 2); // Start at top
    const r = val * radius;
    const x = center + r * Math.cos(angle);
    const y = center + r * Math.sin(angle);
    return `${x},${y}`;
  }).join(" ");

  // Background grid levels
  const gridLevels = [0.25, 0.5, 0.75, 1];

  return (
    <div className={cn("relative flex items-center justify-center", className)}>
        <svg width={size} height={size} viewBox={`0 0 ${size} ${size}`} className="overflow-visible">
            {/* Grid Lines */}
            {gridLevels.map((level) => (
                <polygon
                    key={level}
                    points={[0, 1, 2, 3].map(i => {
                        const angle = (Math.PI / 2) * i - (Math.PI / 2);
                        const r = level * radius;
                        return `${center + r * Math.cos(angle)},${center + r * Math.sin(angle)}`;
                    }).join(" ")}
                    fill="none"
                    stroke="#cbd5e1" // slate-300
                    strokeWidth="1"
                    strokeDasharray="2 2"
                />
            ))}

            {/* Axes */}
            {[0, 1, 2, 3].map(i => {
                const angle = (Math.PI / 2) * i - (Math.PI / 2);
                const x2 = center + radius * Math.cos(angle);
                const y2 = center + radius * Math.sin(angle);
                return (
                    <line 
                        key={i} 
                        x1={center} 
                        y1={center} 
                        x2={x2} 
                        y2={y2} 
                        stroke="#e2e8f0" // slate-200
                        strokeWidth="1" 
                    />
                );
            })}

            {/* Data Polygon */}
            <polygon
                points={points}
                fill="rgba(6, 182, 212, 0.2)" // cyan-500/20
                stroke="#06b6d4" // cyan-500
                strokeWidth="2"
                strokeLinejoin="round"
                className="drop-shadow-sm transition-all duration-500 ease-out"
            />

            {/* Data Points */}
            {values.map((val, i) => {
                const angle = (Math.PI / 2) * i - (Math.PI / 2);
                const r = val * radius;
                const x = center + r * Math.cos(angle);
                const y = center + r * Math.sin(angle);
                return (
                    <circle
                        key={i}
                        cx={x}
                        cy={y}
                        r="3"
                        fill="white"
                        stroke={val > 0.8 ? "#10b981" : val > 0.5 ? "#f59e0b" : "#64748b"} // Emerald/Amber/Slate
                        strokeWidth="2"
                        className="transition-all duration-500"
                    />
                );
            })}

            {/* Labels */}
            <text x={center} y={center - radius - 15} textAnchor="middle" className="fill-slate-500 text-[9px] font-bold font-mono tracking-widest">FACE</text>
            <text x={center + radius + 25} y={center} textAnchor="middle" className="fill-slate-500 text-[9px] font-bold font-mono tracking-widest">FINGER</text>
            <text x={center} y={center + radius + 20} textAnchor="middle" className="fill-slate-500 text-[9px] font-bold font-mono tracking-widest">IRIS_N</text>
            <text x={center - radius - 25} y={center} textAnchor="middle" className="fill-slate-500 text-[9px] font-bold font-mono tracking-widest">IRIS_V</text>
        </svg>
    </div>
  );
}
