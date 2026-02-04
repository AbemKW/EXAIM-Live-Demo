'use client';

import React from 'react';

interface WordCountComparisonProps {
  totalWords: number;
  summaryWords: number;
}

export default function WordCountComparison({ totalWords, summaryWords }: WordCountComparisonProps) {
  return (
    <div className="word-count-comparison mb-3 flex-shrink-0">
      <div className="word-count-comparison-content bg-gradient-to-r from-zinc-950/60 to-zinc-900/50 backdrop-blur-md border border-white/10 rounded-lg p-4">
        <div className="flex flex-col gap-3">
          <div className="flex items-center justify-between">
            <span className="text-sm font-medium text-muted-foreground">MAS CDSS Generated</span>
            <span className="text-lg font-bold text-foreground">{totalWords.toLocaleString()} words</span>
          </div>
          <div className="flex items-center justify-between">
            <span className="text-sm font-medium text-muted-foreground">EXAIM Summaries</span>
            <span className="text-lg font-bold text-teal-400">{summaryWords.toLocaleString()} words</span>
          </div>
        </div>
      </div>
    </div>
  );
}





