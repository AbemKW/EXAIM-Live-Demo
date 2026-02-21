'use client';

import React, { useState, useEffect, useMemo } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { diffChars, DiffResult } from '@/lib/diff';
import { cn } from '@/lib/utils';

interface DiffGlowTextProps {
  oldText: string;
  newText: string;
  className?: string;
  glowDuration?: number; // Duration of the glow in seconds
}

export default function DiffGlowText({ 
  oldText, 
  newText, 
  className,
  glowDuration = 4
}: DiffGlowTextProps) {
  const [showDiff, setShowDiff] = useState(true);
  
  // Calculate diff
  const diffs = useMemo(() => {
    // If no old text, treat the entire new text as 'added' so it glows
    if (!oldText) {
      return [{ type: 'added', value: newText }] as DiffResult[];
    }
    return diffChars(oldText, newText);
  }, [oldText, newText]);

  // Turn off diff highlighting after glowDuration
  useEffect(() => {
    if (oldText && oldText !== newText) {
      setShowDiff(true);
      const timer = setTimeout(() => {
        setShowDiff(false);
      }, glowDuration * 1000);
      return () => clearTimeout(timer);
    } else {
      setShowDiff(false);
    }
  }, [oldText, newText, glowDuration]);

  return (
    <div className={cn("relative font-medium", className)}>
      <AnimatePresence mode="wait">
        {showDiff ? (
          <motion.div
            key="diff"
            initial={{ opacity: 0.8 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0, filter: 'blur(8px)' }}
            transition={{ duration: 0.5 }}
            className="flex flex-wrap"
          >
            {diffs.map((part, index) => {
              if (part.type === 'added') {
                return (
                  <motion.span
                    key={`added-${index}`}
                    initial={{ backgroundColor: 'rgba(34, 197, 94, 0)', color: 'inherit' }}
                    animate={{ 
                      backgroundColor: [
                        'rgba(34, 197, 94, 0.2)', 
                        'rgba(34, 197, 94, 0.7)', 
                        'rgba(34, 197, 94, 0.2)'
                      ],
                      color: '#ffffff',
                      textShadow: [
                        '0 0 4px rgba(34, 197, 94, 0.4)',
                        '0 0 12px rgba(34, 197, 94, 0.9)',
                        '0 0 4px rgba(34, 197, 94, 0.4)'
                      ]
                    }}
                    transition={{
                      duration: 2,
                      repeat: Infinity,
                      ease: "easeInOut"
                    }}
                    className="rounded px-0.5 whitespace-pre-wrap font-bold bg-green-500/30"
                  >
                    {part.value}
                  </motion.span>
                );
              }
              if (part.type === 'removed') {
                return (
                  <motion.span
                    key={`removed-${index}`}
                    initial={{ backgroundColor: 'rgba(239, 68, 68, 0)', color: 'inherit' }}
                    animate={{ 
                      backgroundColor: [
                        'rgba(239, 68, 68, 0.2)', 
                        'rgba(239, 68, 68, 0.7)', 
                        'rgba(239, 68, 68, 0.2)'
                      ],
                      color: '#ffffff',
                      textShadow: [
                        '0 0 4px rgba(239, 68, 68, 0.4)',
                        '0 0 12px rgba(239, 68, 68, 0.9)',
                        '0 0 4px rgba(239, 68, 68, 0.4)'
                      ],
                    }}
                    transition={{
                      duration: 2,
                      repeat: Infinity,
                      ease: "easeInOut"
                    }}
                    className="rounded px-0.5 whitespace-pre-wrap line-through font-bold bg-red-500/30"
                  >
                    {part.value}
                  </motion.span>
                );
              }
              return (
                <span key={`equal-${index}`} className="whitespace-pre-wrap">
                  {part.value}
                </span>
              );
            })}
          </motion.div>
        ) : (
          <motion.div
            key="normal"
            initial={{ opacity: 0, filter: 'blur(4px)' }}
            animate={{ opacity: 1, filter: 'blur(0px)' }}
            transition={{ duration: 0.8 }}
            className="whitespace-pre-wrap"
          >
            {newText}
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}
