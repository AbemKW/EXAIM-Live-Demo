'use client';

import React from 'react';
import { motion } from 'framer-motion';

interface MascotProps {
  className?: string;
}

export default function Mascot({ className = '' }: MascotProps) {
  return (
    <motion.div
      className={`relative ${className}`}
      initial={{ opacity: 0, scale: 0.8 }}
      animate={{ opacity: 1, scale: 1 }}
      transition={{ duration: 0.5 }}
    >
      <motion.div
        className="relative w-16 h-16 rounded-2xl bg-primary/20 backdrop-blur-md border border-white/10 glass-card flex items-center justify-center"
        animate={{
          y: [0, -8, 0],
        }}
        transition={{
          duration: 4,
          repeat: Infinity,
          ease: "easeInOut",
        }}
      >
        {/* Simple medical cross icon */}
        <svg
          width="32"
          height="32"
          viewBox="0 0 32 32"
          fill="none"
          xmlns="http://www.w3.org/2000/svg"
          className="text-primary"
        >
          {/* Vertical line */}
          <motion.rect
            x="14"
            y="6"
            width="4"
            height="20"
            rx="2"
            fill="currentColor"
            initial={{ scaleY: 0 }}
            animate={{ scaleY: 1 }}
            transition={{ duration: 0.5, delay: 0.2 }}
          />
          {/* Horizontal line */}
          <motion.rect
            x="6"
            y="14"
            width="20"
            height="4"
            rx="2"
            fill="currentColor"
            initial={{ scaleX: 0 }}
            animate={{ scaleX: 1 }}
            transition={{ duration: 0.5, delay: 0.4 }}
          />
        </svg>
        
        {/* Subtle glow effect */}
        <motion.div
          className="absolute inset-0 rounded-2xl bg-primary/10"
          animate={{
            opacity: [0.3, 0.6, 0.3],
          }}
          transition={{
            duration: 2,
            repeat: Infinity,
            ease: "easeInOut",
          }}
        />
      </motion.div>
    </motion.div>
  );
}

