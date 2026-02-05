'use client';

import React from 'react';
import type { Summary } from '@/lib/types';

interface SummaryTimelineProps {
  summaries: Summary[];
  expandedSummaryId: string | undefined;
  onDotClick: (summaryId: string) => void;
  summaryRefs: Map<string, React.RefObject<HTMLDivElement | null>>;
}

export default function SummaryTimeline({
  summaries,
  expandedSummaryId,
  onDotClick,
  summaryRefs,
}: SummaryTimelineProps) {
  const formatTime = (date: Date) => {
    return date.toLocaleTimeString('en-US', {
      hour: '2-digit',
      minute: '2-digit',
      hour12: false,
    });
  };

  if (summaries.length === 0) {
    return null;
  }

  return (
    <div className="summary-timeline-container">
      <div className="summary-timeline-line" />
      <div className="summary-timeline-dots">
        {summaries.map((summary, index) => {
          const isExpanded = summary.id === expandedSummaryId;
          
          return (
            <div
              key={summary.id}
              className="summary-timeline-item"
              style={{ 
                left: summaries.length === 1 
                  ? '50%'
                  : `${(index / (summaries.length - 1)) * 100}%`,
                transform: 'translateX(-50%)',
              }}
            >
              <button
                onClick={() => onDotClick(summary.id)}
                className={`summary-timeline-dot ${isExpanded ? 'active' : ''}`}
                aria-label={`Go to summary at ${formatTime(summary.timestamp)}`}
              >
                <span className="summary-timeline-dot-inner" />
              </button>
              <div className={`summary-timeline-label ${isExpanded ? 'active' : ''}`}>
                {formatTime(summary.timestamp)}
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}

