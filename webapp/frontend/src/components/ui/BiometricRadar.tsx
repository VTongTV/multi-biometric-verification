import React from "react";

interface BiometricRadarProps {
  data: {
    face: number;
    fingerprint: number;
    irisNir: number;
    irisVis: number;
  };
  className?: string;
}

export function BiometricRadar({ data, className }: BiometricRadarProps) {
  // Config
  const size = 200;
  const center = size / 2;
  const radius = 70; // Max radius
  
  // Axes Points (North, East, South, West)
  const axes = [
    { label: "Face", angle: -90, val: data.face },
    { label: "Finger", angle: 0, val: data.fingerprint },
    { label: "Iris (N)", angle: 90, val: data.irisNir },
    { label: "Iris (V)", angle: 180, val: data.irisVis },
  ];

  // Calculate coordinates for value points
  const points = axes.map(axis => {
    const r = axis.val * radius; // Scale 0-1 to radius
    const x = center + r * Math.cos(axis.angle * Math.PI / 180);
    const y = center + r * Math.sin(axis.angle * Math.PI / 180);
    return `${x},${y}`;
  }).join(" ");

  // Calculate coordinates for background grid (100%, 75%, 50%, 25%)
  const gridLevels = [1, 0.75, 0.5, 0.25];

  return (
    <div className={className}>
        <svg viewBox={`0 0 ${size} ${size}`} className="w-full h-full overflow-visible">
            {/* Background Grid (Diamond Shape) */}
            {gridLevels.map((level, i) => (
                <polygon 
                    key={i}
                    points={[
                        `${center},${center - radius * level}`, // Top
                        `${center + radius * level},${center}`, // Right
                        `${center},${center + radius * level}`, // Bottom
                        `${center - radius * level},${center}`  // Left
                    ].join(" ")}
                    fill="none"
                    stroke="#1e293b" // Slate 800
                    strokeWidth="1"
                    className="opacity-50"
                />
            ))}

            {/* Axes Lines */}
            <line x1={center} y1={center - radius} x2={center} y2={center + radius} stroke="#1e293b" strokeWidth="1" />
            <line x1={center - radius} y1={center} x2={center + radius} y2={center} stroke="#1e293b" strokeWidth="1" />

            {/* Data Area */}
            <polygon 
                points={points}
                fill="rgba(6, 182, 212, 0.2)" // Cyan fill
                stroke="#06b6d4" // Cyan stroke
                strokeWidth="2"
                className="drop-shadow-[0_0_8px_rgba(6,182,212,0.5)] transition-all duration-500 ease-out"
            />

            {/* Data Points */}
            {axes.map((axis, i) => {
                const r = axis.val * radius;
                const x = center + r * Math.cos(axis.angle * Math.PI / 180);
                const y = center + r * Math.sin(axis.angle * Math.PI / 180);
                return (
                    <circle 
                        key={i}
                        cx={x} 
                        cy={y} 
                        r="3" 
                        fill="#06b6d4" 
                        className="transition-all duration-500 ease-out"
                    />
                );
            })}

            {/* Labels */}
            <text x={center} y={center - radius - 10} textAnchor="middle" fill="#94a3b8" fontSize="10" className="font-mono">FACE</text>
            <text x={center + radius + 25} y={center + 3} textAnchor="middle" fill="#94a3b8" fontSize="10" className="font-mono">FINGER</text>
            <text x={center} y={center + radius + 15} textAnchor="middle" fill="#94a3b8" fontSize="10" className="font-mono">IRIS (NIR)</text>
            <text x={center - radius - 25} y={center + 3} textAnchor="middle" fill="#94a3b8" fontSize="10" className="font-mono">IRIS (VIS)</text>
        </svg>
    </div>
  );
}
