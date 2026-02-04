'use client';

import React, { useState, useMemo, createRef } from 'react';
import { useSummaries, useCDSSStore } from '@/store/cdssStore';
import SummaryCard from './SummaryCard';
import SummaryTimeline from './SummaryTimeline';
import { Card, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';

interface SummaryHistoryProps {
  onSummarySelect?: (summaryId: string) => void;
}

export default function SummaryHistory({ onSummarySelect }: SummaryHistoryProps) {
  const summaries = useSummaries();
  const toggleSummary = useCDSSStore((state) => state.toggleSummary);
  const [expandedSummaryId, setExpandedSummaryId] = useState<string | undefined>(
    summaries.find(s => s.isExpanded)?.id
  );
  
  // Create refs for each summary for timeline navigation
  const summaryRefs = useMemo(() => {
    const refs = new Map<string, React.RefObject<HTMLDivElement>>();
    summaries.forEach(summary => {
      refs.set(summary.id, createRef<HTMLDivElement>());
    });
    return refs;
  }, [summaries]);

  // Sort summaries by timestamp (newest first)
  const sortedSummaries = useMemo(() => {
    return [...summaries].sort((a, b) => b.timestamp.getTime() - a.timestamp.getTime());
  }, [summaries]);

  const handleSummaryClick = (summaryId: string) => {
    toggleSummary(summaryId);
    setExpandedSummaryId(summaryId);
    if (onSummarySelect) {
      onSummarySelect(summaryId);
    }
  };

  const handleTimelineDotClick = (summaryId: string) => {
    const ref = summaryRefs.get(summaryId);
    if (ref?.current) {
      ref.current.scrollIntoView({ behavior: 'smooth', block: 'center' });
    }
    handleSummaryClick(summaryId);
  };

  if (summaries.length === 0) {
    return (
      <Card className="flex flex-col h-full bg-card/30 backdrop-blur-xl border-white/10 glass-card">
        <CardHeader className="bg-gradient-to-r from-zinc-950/40 to-zinc-900/30 backdrop-blur-md border-b border-white/10 py-3 px-5 glass-header">
          <CardTitle className="text-xl font-bold">Summary Timeline</CardTitle>
        </CardHeader>
        <div className="flex-1 flex items-center justify-center p-8">
          <p className="text-muted-foreground text-center text-base">
            No summaries yet. Summaries will appear here as EXAIM processes agent traces.
          </p>
        </div>
      </Card>
    );
  }

  return (
    <Card className="flex flex-col h-full bg-card/30 backdrop-blur-xl border-border/50 dark:border-white/10 glass-card">
      {/* Panel Header */}
      <CardHeader className="bg-gradient-to-r from-muted/60 to-muted/40 dark:from-zinc-950/40 dark:to-zinc-900/30 backdrop-blur-md border-b border-border/50 dark:border-white/10 py-3 px-5 glass-header flex-shrink-0">
        <div className="flex justify-between items-center">
          <CardTitle className="text-xl font-bold text-foreground">Summary Timeline</CardTitle>
          <Badge variant="secondary" className="text-sm">
            {summaries.length} summar{summaries.length !== 1 ? 'ies' : 'y'}
          </Badge>
        </div>
      </CardHeader>

      {/* Timeline Navigation */}
      {summaries.length > 0 && (
        <div className="flex-shrink-0 px-4 pt-4">
          <SummaryTimeline
            summaries={sortedSummaries}
            expandedSummaryId={expandedSummaryId}
            onDotClick={handleTimelineDotClick}
            summaryRefs={summaryRefs}
          />
        </div>
      )}

      {/* Scrollable Summary List */}
      <div className="flex-1 overflow-y-auto px-4 pt-4 pb-4 custom-scrollbar min-h-0">
        <div className="space-y-3">
          {sortedSummaries.map((summary) => (
            <div
              key={summary.id}
              ref={summaryRefs.get(summary.id)}
              className="scroll-mt-4"
            >
              <SummaryCard
                summary={summary}
                showComparison={false}
                mode="list"
                onClick={() => handleSummaryClick(summary.id)}
              />
            </div>
          ))}
        </div>
      </div>
    </Card>
  );
}

