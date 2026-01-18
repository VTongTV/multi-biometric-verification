import React, { useState, useEffect } from "react";

interface DecryptionTextProps {
  text: string;
  speed?: number; // ms per frame
  revealSpeed?: number; // ms per character reveal
  className?: string;
  useHover?: boolean;
}

const CHARS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789@#$%&";

export function DecryptionText({ text, speed = 30, revealSpeed = 50, className, useHover = false }: DecryptionTextProps) {
  const [displayText, setDisplayText] = useState(text);
  const [isHovered, setIsHovered] = useState(false);
  const [isRevealed, setIsRevealed] = useState(!useHover);

  useEffect(() => {
    if (useHover && !isHovered) {
        setDisplayText(text.split('').map(c => c === ' ' ? ' ' : CHARS[Math.floor(Math.random() * CHARS.length)]).join(''));
        return;
    }

    let revealIndex = 0;
    let interval: NodeJS.Timeout;

    const animate = () => {
      setDisplayText(prev => {
        return text.split('').map((char, i) => {
          if (i < revealIndex) return char;
          if (char === ' ') return ' ';
          return CHARS[Math.floor(Math.random() * CHARS.length)];
        }).join('');
      });

      if (revealIndex >= text.length) {
        clearInterval(interval);
      }
      
      // Increment reveal index slower than the scramble update
      // We can't do this easily in one interval. 
      // Let's just increment revealIndex every X calls or use a separate logic.
    };

    // Simplified approach: Separate timer for scrambling vs revealing
    interval = setInterval(animate, speed);
    
    const revealTimer = setInterval(() => {
        revealIndex++;
        if (revealIndex > text.length) clearInterval(revealTimer);
    }, revealSpeed);

    return () => {
        clearInterval(interval);
        clearInterval(revealTimer);
    };
  }, [text, speed, revealSpeed, useHover, isHovered]);

  return (
    <span 
        className={className} 
        onMouseEnter={() => useHover && setIsHovered(true)}
        onMouseLeave={() => useHover && setIsHovered(false)}
    >
      {displayText}
    </span>
  );
}
