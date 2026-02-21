'use client';

import React, { useState, useCallback } from 'react';
import { useSummaries, useTotalWords, useTotalSummaryWords } from '@/store/cdssStore';
import SummaryCard from './SummaryCard';
import CompressionStats from './CompressionStats';
import WordCountComparison from './WordCountComparison';
import { Card, CardHeader, CardTitle, CardContent } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';

export default function SummariesPanel() {
  const summaries = useSummaries();
  const totalWords = useTotalWords();
  const totalSummaryWords = useTotalSummaryWords();
  const [comparisonMode, setComparisonMode] = useState(false);
  
  // Calculate compression rate for button display
  // Show compression if summaries are smaller, expansion if larger
  const compressionInfo = React.useMemo(() => {
    if (totalWords === 0) {
      return null; // Can't calculate without original words
    }
    if (totalSummaryWords === 0) {
      return null; // No summaries yet
    }
    const compressionValue = ((totalWords - totalSummaryWords) / totalWords) * 100;
    const isCompression = compressionValue > 0;
    const percentage = Math.abs(compressionValue).toFixed(1);
    return {
      percentage,
      isCompression,
      label: isCompression ? 'compression' : 'expansion'
    };
  }, [totalWords, totalSummaryWords]);
  
  // Get only the spotlight summary (latest/expanded)
  // Prioritize expanded summary, otherwise use the first (newest) summary
  const spotlightSummary = React.useMemo(() => {
    const expanded = summaries.find(s => s.isExpanded);
    if (expanded) return expanded;
    // If no expanded summary, use the first one (newest, since they're added at the beginning)
    return summaries.length > 0 ? summaries[0] : null;
  }, [summaries]);

  // Find the previous summary to show diffs against
  const previousSummary = React.useMemo(() => {
    if (!spotlightSummary) return null;
    const index = summaries.findIndex(s => s.id === spotlightSummary.id);
    // The previous summary is the one added just before the current one (at index+1)
    if (index !== -1 && index + 1 < summaries.length) {
      return summaries[index + 1];
    }
    return null;
  }, [summaries, spotlightSummary]);
  
  // Stable click handler to prevent issues during re-renders
  const handleComparisonToggle = useCallback(() => {
    setComparisonMode(prev => !prev);
  }, []);

  // Debug: Log summaries state
  React.useEffect(() => {
    console.log('SummariesPanel - Total summaries:', summaries.length);
    console.log('SummariesPanel - Summaries:', summaries.map(s => ({ id: s.id, isExpanded: s.isExpanded })));
    console.log('SummariesPanel - Spotlight summary:', spotlightSummary?.id);
  }, [summaries, spotlightSummary]);

  return (
    <Card className="flex flex-col overflow-hidden h-full bg-card/10 backdrop-blur-xl border-border/40 dark:border-white/5 glass-card">
      {/* Panel Header */}
      <CardHeader className="bg-gradient-to-r from-muted/40 to-muted/20 dark:from-zinc-950/20 dark:to-zinc-900/10 backdrop-blur-md border-b border-border/40 dark:border-white/5 py-2 px-4 flex-shrink-0 glass-header">
        <div className="flex justify-between items-center">
          <CardTitle className="text-lg font-bold text-foreground">EXAIM Summaries</CardTitle>
          <div className="flex items-center gap-2">
            <Button
              variant={comparisonMode ? 'default' : 'outline'}
              size="sm"
              onClick={handleComparisonToggle}
              className="text-[10px] h-7 px-2"
            >
              {comparisonMode 
                ? compressionInfo !== null
                  ? `✓ ${compressionInfo.percentage}% ${compressionInfo.label}`
                  : '✓ Comp'
                : 'Comparison'}
            </Button>
            <Badge variant="secondary" className="text-[10px] px-1.5 h-5">
              {summaries.length}
            </Badge>
          </div>
        </div>
      </CardHeader>

      {/* Panel Content - Only Spotlight Summary */}
      <div className="flex-1 flex flex-col overflow-hidden relative">
        {/* Subtle background pattern to fill space */}
        <div className="absolute inset-0 opacity-[0.03] pointer-events-none dark:invert" style={{ backgroundImage: 'radial-gradient(#000 0.5px, transparent 0.5px)', backgroundSize: '10px 10px' }}></div>
        
        {!spotlightSummary ? (
          <CardContent className="flex-1 flex items-center justify-center px-4 py-8 relative z-10">
            <p className="text-muted-foreground text-center text-sm italic">
              Awaiting agent traces to generate summaries...
            </p>
          </CardContent>
        ) : (
          <div className="flex-1 flex flex-col overflow-hidden px-2 pt-2 pb-2 relative z-10">
            {comparisonMode && (
              <div className="mb-2 space-y-1">
                <WordCountComparison
                  totalWords={totalWords}
                  summaryWords={totalSummaryWords}
                />
                <CompressionStats
                  totalWords={totalWords}
                  summaryWords={totalSummaryWords}
                />
              </div>
            )}
            <div className="mb-1.5 flex-shrink-0 flex items-center justify-between">
              <div className="text-[10px] font-bold text-teal-500/80 uppercase tracking-widest flex items-center gap-1.5 ml-1">
                <span className="inline-block w-1.5 h-1.5 bg-teal-500 rounded-full animate-pulse"></span>
                Spotlight
              </div>
            </div>
            <div className="flex-1 overflow-hidden min-h-0 flex flex-col items-center justify-start">
              <SummaryCard
                key="spotlight-summary-card"
                summary={spotlightSummary}
                previousSummary={previousSummary}
                showComparison={comparisonMode}
                mode="spotlight"
              />
            </div>
          </div>
        )}
      </div>
    </Card>
  );
}

