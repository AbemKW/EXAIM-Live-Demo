'use client';

import React from 'react';

interface CompressionStatsProps {
  totalWords: number;
  summaryWords: number;
}

export default function CompressionStats({ totalWords, summaryWords }: CompressionStatsProps) {
  if (totalWords === 0 || summaryWords === 0) {
    return null;
  }

  // Calculate compression or expansion percentage
  // Compression = (original - compressed) / original * 100
  // Formula: (totalWords - summaryWords) / totalWords * 100
  // 
  // IMPORTANT: Summaries have a minimum size (~135-150 words) due to 6 structured fields,
  // even when created from small buffers. When summaries are created frequently from
  // small buffers (20-30 words), cumulative summary words can exceed cumulative trace words.
  // This is expected behavior - summaries add structure and context beyond raw compression.
  const compressionValue = totalWords > 0 
    ? ((totalWords - summaryWords) / totalWords) * 100
    : 0;
  
  const percentage = Math.abs(compressionValue).toFixed(1);
  const isCompression = compressionValue > 0;

  return (
    <div className="compression-stats">
      <div className="compression-stats-content">
        {isCompression ? (
          <>
            <span className="compression-label">EXAIM reduced</span>
            <span className="compression-number">{totalWords.toLocaleString()}</span>
            <span className="compression-arrow">→</span>
            <span className="compression-number">{summaryWords.toLocaleString()}</span>
            <span className="compression-label">words</span>
            <span className="compression-percentage">
              ({percentage}% compression)
            </span>
          </>
        ) : (
          <>
            <span className="compression-label">EXAIM expanded</span>
            <span className="compression-number">{totalWords.toLocaleString()}</span>
            <span className="compression-arrow">→</span>
            <span className="compression-number">{summaryWords.toLocaleString()}</span>
            <span className="compression-label">words</span>
            <span className="compression-percentage text-amber-400">
              ({percentage}% expansion)
            </span>
          </>
        )}
      </div>
    </div>
  );
}

