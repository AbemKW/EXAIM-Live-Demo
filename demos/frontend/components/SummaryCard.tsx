'use client';

import React, { forwardRef, useState } from 'react';
import { motion } from 'framer-motion';
import { useCDSSStore } from '@/store/cdssStore';
import type { Summary, SummaryData } from '@/lib/types';
import { Card, CardHeader, CardTitle, CardContent } from '@/components/ui/card';
import { useTheme } from '@/hooks/useTheme';
import { ChevronDown, ChevronUp } from 'lucide-react';
import DiffGlowText from './DiffGlowText';

interface SummaryCardProps {
  summary: Summary;
  previousSummary?: Summary | null;
  showComparison?: boolean;
  mode?: 'spotlight' | 'list';
  onClick?: () => void;
}

const SummaryCard = forwardRef<HTMLDivElement, SummaryCardProps>(({ 
  summary, 
  previousSummary = null,
  showComparison = false, 
  mode = 'list',
  onClick 
}, ref) => {
  // Track which fields are expanded (showing full text)
  const [expandedFields, setExpandedFields] = useState<Set<string>>(new Set());

  const toggleFieldExpansion = (fieldKey: string) => {
    setExpandedFields(prev => {
      const next = new Set(prev);
      if (next.has(fieldKey)) {
        next.delete(fieldKey);
      } else {
        next.add(fieldKey);
      }
      return next;
    });
  };

  const formatTime = (date: Date) => {
    return date.toLocaleTimeString('en-US', {
      hour: '2-digit',
      minute: '2-digit',
      second: '2-digit',
    });
  };

  const { theme } = useTheme();
  const isDark = theme === 'dark';
  // Use theme-aware background colors - more visible in light theme
  const bgOpacity = isDark ? 0.04 : 0.12;
  
  const fields = [
    { 
      label: 'Status / Action', 
      key: 'status_action' as keyof SummaryData,
      value: summary.data.status_action,
      fullValue: summary.data.full_status_action,
      prevValue: previousSummary?.data.status_action,
      prevFullValue: previousSummary?.data.full_status_action,
      color: 'var(--summary-status-action)',
      bgColor: `oklch(0.50 0.08 260 / ${bgOpacity})`,
    },
    { 
      label: 'Key Findings', 
      key: 'key_findings' as keyof SummaryData,
      value: summary.data.key_findings,
      fullValue: summary.data.full_key_findings,
      prevValue: previousSummary?.data.key_findings,
      prevFullValue: previousSummary?.data.full_key_findings,
      color: 'var(--summary-key-findings)',
      bgColor: `oklch(0.55 0.08 150 / ${bgOpacity})`,
    },
    { 
      label: 'Differential & Rationale', 
      key: 'differential_rationale' as keyof SummaryData,
      value: summary.data.differential_rationale,
      fullValue: summary.data.full_differential_rationale,
      prevValue: previousSummary?.data.differential_rationale,
      prevFullValue: previousSummary?.data.full_differential_rationale,
      color: 'var(--summary-differential)',
      bgColor: `oklch(0.50 0.08 300 / ${bgOpacity})`,
    },
    { 
      label: 'Uncertainty / Confidence', 
      key: 'uncertainty_confidence' as keyof SummaryData,
      value: summary.data.uncertainty_confidence,
      fullValue: summary.data.full_uncertainty_confidence,
      prevValue: previousSummary?.data.uncertainty_confidence,
      prevFullValue: previousSummary?.data.full_uncertainty_confidence,
      color: 'var(--summary-uncertainty)',
      bgColor: `oklch(0.58 0.08 60 / ${bgOpacity})`,
    },
    { 
      label: 'Recommendation / Next Step', 
      key: 'recommendation_next_step' as keyof SummaryData,
      value: summary.data.recommendation_next_step,
      fullValue: summary.data.full_recommendation_next_step,
      prevValue: previousSummary?.data.recommendation_next_step,
      prevFullValue: previousSummary?.data.full_recommendation_next_step,
      color: 'var(--summary-recommendation)',
      bgColor: `oklch(0.50 0.08 180 / ${bgOpacity})`,
    },
    { 
      label: 'Agent Contributions', 
      key: 'agent_contributions' as keyof SummaryData,
      value: summary.data.agent_contributions,
      fullValue: summary.data.full_agent_contributions,
      prevValue: previousSummary?.data.agent_contributions,
      prevFullValue: previousSummary?.data.full_agent_contributions,
      color: 'var(--summary-contributions)',
      bgColor: `oklch(0.50 0.03 0 / ${bgOpacity})`,
    },
  ];

  // Spotlight mode - always show full expanded content
  if (mode === 'spotlight') {
    return (
      <motion.div
        ref={ref}
        layout
        data-summary-id={summary.id}
        className="max-h-full"
      >
        <Card className="overflow-hidden transition-all duration-300 border-border/50 bg-card/80 dark:bg-teal-950/30 backdrop-blur-xl shadow-2xl glass-card spotlight-glow flex flex-col max-h-full">
          {/* Header */}
          <CardHeader className="py-1.5 px-2 border-b border-border/30 dark:border-white/10 flex-shrink-0">
            <div className="flex justify-between items-center">
              <CardTitle className="text-sm font-bold text-foreground dark:text-teal-100 leading-tight">
                {summary.data.status_action}
              </CardTitle>
              <span className="text-xs text-muted-foreground font-medium flex-shrink-0 ml-2">
                {formatTime(summary.timestamp)}
              </span>
            </div>
          </CardHeader>

          {/* Full Content - Always Visible */}
          <CardContent className="pt-1 pb-2 px-2 flex-1 overflow-y-auto custom-scrollbar">
            <div className="grid grid-cols-1 md:grid-cols-2 gap-2 items-start">
              {fields.map((field, index) => {
                const isExpanded = expandedFields.has(field.key);
                const hasFull = field.fullValue != null && field.fullValue.trim() !== '';
                const displayValue = (isExpanded && hasFull) ? (field.fullValue || '') : field.value;
                const prevDisplayValue = (isExpanded && field.prevFullValue) ? (field.prevFullValue || '') : (field.prevValue || '');
                
                return (
                  <div 
                    key={index} 
                    className="summary-field-group rounded-lg p-1.5 transition-all backdrop-blur-md border border-border/50 dark:border-white/10 hover:border-border dark:hover:border-white/20 min-h-0"
                    style={{
                      borderLeft: `3px solid ${field.color}`,
                      backgroundColor: field.bgColor,
                      boxShadow: 'inset 0 1px 1px 0 rgba(0, 0, 0, 0.05)',
                    }}
                  >
                    <div className="flex items-center justify-between mb-1">
                      <div 
                        className="text-[10px] font-extrabold uppercase tracking-widest leading-tight"
                        style={{ 
                          color: field.color,
                          fontWeight: 900,
                        }}
                      >
                        {field.label}
                      </div>
                      {hasFull && (
                        <button
                          onClick={(e) => {
                            e.stopPropagation();
                            toggleFieldExpansion(field.key);
                          }}
                          className="text-[10px] px-1 py-0.5 rounded hover:bg-black/10 dark:hover:bg-white/10 transition-colors flex items-center gap-0.5"
                          style={{ color: field.color }}
                        >
                          {isExpanded ? (
                            <><ChevronUp className="w-3 h-3" /> Less</>
                          ) : (
                            <><ChevronDown className="w-3 h-3" /> Full</>
                          )}
                        </button>
                      )}
                    </div>
                    <div className="text-xs text-foreground leading-relaxed font-semibold break-words">
                      <DiffGlowText 
                        oldText={prevDisplayValue} 
                        newText={displayValue} 
                        glowDuration={3}
                      />
                    </div>
                  </div>
                );
              })}
            </div>
          </CardContent>
        </Card>
      </motion.div>
    );
  }

  // List mode - show collapsed header only
  return (
    <motion.div
      ref={ref}
      layout
      initial={{ opacity: 0, y: -10 }}
      animate={{ opacity: 1, y: 0 }}
      exit={{ opacity: 0, y: -10 }}
      transition={{ duration: 0.2 }}
      data-summary-id={summary.id}
    >
      <Card 
        className="overflow-hidden transition-all duration-200 border-border/50 dark:border-white/10 bg-card/60 dark:bg-card/40 backdrop-blur-md hover:bg-card/80 dark:hover:bg-card/60 hover:border-border dark:hover:border-white/20 cursor-pointer glass-card"
        onClick={onClick}
      >
        {/* Header - Clickable */}
        <CardHeader className="py-2 px-4">
          <div className="flex justify-between items-center">
            <CardTitle className="text-sm line-clamp-1 font-semibold flex-1 pr-4">
              {summary.data.status_action}
            </CardTitle>
            <div className="flex items-center gap-2 flex-shrink-0">
              <span className="text-xs text-muted-foreground font-medium">
                {formatTime(summary.timestamp)}
              </span>
              <svg 
                xmlns="http://www.w3.org/2000/svg" 
                width="16" 
                height="16" 
                viewBox="0 0 24 24" 
                fill="none" 
                stroke="currentColor" 
                strokeWidth="2" 
                strokeLinecap="round" 
                strokeLinejoin="round"
                className="text-muted-foreground"
              >
                <polyline points="9 18 15 12 9 6"></polyline>
              </svg>
            </div>
          </div>
        </CardHeader>
      </Card>
    </motion.div>
  );
});

SummaryCard.displayName = 'SummaryCard';

// Memoize to prevent unnecessary re-renders
export default React.memo(SummaryCard);
