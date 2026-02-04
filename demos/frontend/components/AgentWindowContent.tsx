'use client';

import { useRef, useEffect } from 'react';

interface AgentWindowContentProps {
  fullText: string;
  isExpanded: boolean;
}

export default function AgentWindowContent({
  fullText,
  isExpanded,
}: AgentWindowContentProps) {
  const contentRef = useRef<HTMLDivElement>(null);
  const scrollPositionRef = useRef({
    wasAtBottom: true,
    hadScrollableContent: false,
  });

  useEffect(() => {
    if (!contentRef.current) return;

    const container = contentRef.current;

    // Capture scroll metrics BEFORE React updates the DOM
    const scrollHeight = container.scrollHeight;
    const scrollTop = container.scrollTop;
    const clientHeight = container.clientHeight;

    const wasAtBottom = scrollHeight - scrollTop - clientHeight < 50; // 50px threshold
    const hadScrollableContent = scrollHeight > clientHeight;

    scrollPositionRef.current = { wasAtBottom, hadScrollableContent };
  }, [fullText]); // Run before each text update

  useEffect(() => {
    if (!contentRef.current) return;

    const { wasAtBottom, hadScrollableContent } = scrollPositionRef.current;

    // Auto-scroll if user was at bottom OR content wasn't scrollable before
    const shouldAutoScroll = !hadScrollableContent || wasAtBottom;

    if (shouldAutoScroll) {
      // Use requestAnimationFrame to ensure DOM update is complete
      requestAnimationFrame(() => {
        if (contentRef.current) {
          contentRef.current.scrollTop = contentRef.current.scrollHeight;
        }
      });
    }
  }, [fullText]); // Run after DOM updates

  // Force scroll to bottom when expanding
  useEffect(() => {
    if (isExpanded && contentRef.current) {
      setTimeout(() => {
        if (contentRef.current) {
          contentRef.current.scrollTop = contentRef.current.scrollHeight;
        }
      }, 300); // Match animation duration
    }
  }, [isExpanded]);

  return (
    <div
      ref={contentRef}
      className="flex-1 overflow-y-auto scroll-smooth p-4"
      style={{ scrollBehavior: 'smooth' }}
    >
      <pre className="text-sm leading-loose font-medium text-foreground whitespace-pre-wrap break-words">
        {fullText || 'Waiting for traces...'}
      </pre>
    </div>
  );
}

