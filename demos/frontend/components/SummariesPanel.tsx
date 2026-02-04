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
    <Card className="flex flex-col overflow-hidden h-full bg-card/30 backdrop-blur-xl border-border/50 dark:border-white/10 glass-card">
      {/* Panel Header */}
      <CardHeader className="bg-gradient-to-r from-muted/60 to-muted/40 dark:from-zinc-950/40 dark:to-zinc-900/30 backdrop-blur-md border-b border-border/50 dark:border-white/10 py-3 px-5 glass-header">
        <div className="flex justify-between items-center">
          <CardTitle className="text-xl font-bold text-foreground">EXAIM Summaries</CardTitle>
          <div className="flex items-center gap-2">
            <Button
              variant={comparisonMode ? 'default' : 'outline'}
              size="sm"
              onClick={handleComparisonToggle}
              className="text-xs"
            >
              {comparisonMode 
                ? compressionInfo !== null
                  ? `✓ Comparison (${compressionInfo.percentage}% ${compressionInfo.label})`
                  : '✓ Comparison'
                : 'Comparison'}
            </Button>
            <Badge variant="secondary" className="text-sm">
              {summaries.length} summar{summaries.length !== 1 ? 'ies' : 'y'}
            </Badge>
          </div>
        </div>
      </CardHeader>

      {/* Panel Content - Only Spotlight Summary */}
      <div className="flex-1 flex flex-col overflow-hidden min-h-0">
        {!spotlightSummary ? (
          <CardContent className="flex-1 flex items-center justify-center px-4 py-8">
            <p className="text-muted-foreground text-center text-base">
              No summaries yet. Summaries will appear here as EXAIM processes agent traces.
            </p>
          </CardContent>
        ) : (
          <div className="flex-1 flex flex-col overflow-hidden px-3 pt-1 pb-1">
            {comparisonMode && (
              <>
                <WordCountComparison
                  totalWords={totalWords}
                  summaryWords={totalSummaryWords}
                />
                <div className="mb-1 flex-shrink-0">
                  <CompressionStats
                    totalWords={totalWords}
                    summaryWords={totalSummaryWords}
                  />
                </div>
              </>
            )}
            <div className="mb-0.5 flex-shrink-0">
              <div className="text-xs font-semibold text-teal-400 uppercase tracking-wider flex items-center gap-2">
                <span className="inline-block w-2 h-2 bg-teal-400 rounded-full animate-pulse"></span>
                Latest Summary
              </div>
            </div>
            <div className="flex-1 overflow-hidden min-h-0">
              <SummaryCard
                key={spotlightSummary.id}
                summary={spotlightSummary}
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

