'use client';

import { useRef, useEffect, useMemo, useCallback, useState } from 'react';
import { useAllAgents, useTotalWords, useActiveAgents } from '@/store/cdssStore';
import { Card, CardHeader, CardTitle, CardContent } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import ConsoleLogEntry from './ConsoleLogEntry';
import { useTheme } from '@/hooks/useTheme';

// Generate consistent color for an agent name using hash
function getAgentColor(agentName: string, isDark: boolean): string {
  let hash = 0;
  for (let i = 0; i < agentName.length; i++) {
    hash = agentName.charCodeAt(i) + ((hash << 5) - hash);
  }
  
  // Generate a color in the blue-green-purple range (console-friendly)
  const hue = (hash % 180) + 180; // 180-360 range (blue to purple)
  const saturation = 60 + (hash % 20); // 60-80%
  // Adjust lightness based on theme: darker for light theme, lighter for dark theme
  const lightness = isDark 
    ? 65 + (hash % 15) // 65-80% for dark theme
    : 35 + (hash % 10); // 35-45% for light theme (darker for better contrast)
  
  return `hsl(${hue}, ${saturation}%, ${lightness}%)`;
}

export default function AgentTracesPanel() {
  const agents = useAllAgents();
  const totalWords = useTotalWords();
  const activeAgents = useActiveAgents();
  const consoleRef = useRef<HTMLDivElement>(null);
  const shouldAutoScrollRef = useRef(true);
  const isProgrammaticScrollRef = useRef(false);
  const [showGoToLatest, setShowGoToLatest] = useState(false);
  const { theme } = useTheme();
  const isDark = theme === 'dark';

  // Create a map of agent colors for consistency
  const agentColors = useMemo(() => {
    const colorMap = new Map<string, string>();
    agents.forEach(agent => {
      if (!colorMap.has(agent.agentName)) {
        colorMap.set(agent.agentName, getAgentColor(agent.agentName, isDark));
      }
    });
    return colorMap;
  }, [agents, isDark]);

  // Check if user is at the bottom of the scroll container
  const isAtBottom = useCallback((element: HTMLDivElement, threshold = 50): boolean => {
    const { scrollTop, scrollHeight, clientHeight } = element;
    const distanceFromBottom = scrollHeight - scrollTop - clientHeight;
    return distanceFromBottom <= threshold;
  }, []);

  // Scroll to bottom function - more aggressive approach
  const scrollToBottom = useCallback(() => {
    if (!consoleRef.current) return;
    const container = consoleRef.current;
    
    // Mark as programmatic scroll to prevent handleScroll from disabling auto-scroll
    isProgrammaticScrollRef.current = true;
    
    // Direct scroll assignment
    container.scrollTop = container.scrollHeight;
    
    // Also try smooth scroll as fallback
    container.scrollTo({
      top: container.scrollHeight,
      behavior: 'auto'
    });
    
    // Reset flag after a short delay to allow scroll event to fire
    setTimeout(() => {
      isProgrammaticScrollRef.current = false;
    }, 100);
  }, []);

  // Handle "Go to Latest" button click
  const handleGoToLatest = useCallback(() => {
    shouldAutoScrollRef.current = true;
    setShowGoToLatest(false);
    scrollToBottom();
  }, [scrollToBottom]);

  // Detect user scroll and disable auto-scroll when user scrolls away
  useEffect(() => {
    const container = consoleRef.current;
    if (!container) return;

    let scrollCheckTimeout: NodeJS.Timeout | null = null;

    const handleScroll = () => {
      if (!container) return;
      
      // Ignore scroll events caused by programmatic scrolling
      if (isProgrammaticScrollRef.current) {
        return;
      }
      
      // Clear any pending checks
      if (scrollCheckTimeout) {
        clearTimeout(scrollCheckTimeout);
        scrollCheckTimeout = null;
      }
      
      // Check if user is at bottom
      const atBottom = isAtBottom(container);
      
      // If user scrolls away from bottom, disable auto-scroll and show button
      if (!atBottom) {
        shouldAutoScrollRef.current = false;
        setShowGoToLatest(true);
      } else {
        // User scrolled back to bottom - re-enable auto-scroll after a delay
        scrollCheckTimeout = setTimeout(() => {
          if (container && isAtBottom(container)) {
            shouldAutoScrollRef.current = true;
            setShowGoToLatest(false);
          }
        }, 200);
      }
    };

    container.addEventListener('scroll', handleScroll, { passive: true });
    
    return () => {
      container.removeEventListener('scroll', handleScroll);
      if (scrollCheckTimeout) {
        clearTimeout(scrollCheckTimeout);
      }
    };
  }, [isAtBottom]);

  // Track content changes to trigger auto-scroll
  const agentsContentKey = useMemo(() => {
    // Create a key that changes when any agent's content changes
    return agents.map(a => `${a.id}:${a.fullText.length}`).join('|');
  }, [agents]);

  // Use MutationObserver to detect DOM changes and auto-scroll
  useEffect(() => {
    const container = consoleRef.current;
    if (!container) return;

    let rafId: number | null = null;
    let timeoutId: NodeJS.Timeout | null = null;
    
    const observer = new MutationObserver(() => {
      // Only scroll if auto-scroll is enabled
      if (shouldAutoScrollRef.current) {
        // Clear any pending scrolls
        if (rafId) cancelAnimationFrame(rafId);
        if (timeoutId) clearTimeout(timeoutId);
        
        // Use double RAF to ensure DOM has updated
        rafId = requestAnimationFrame(() => {
          requestAnimationFrame(() => {
            scrollToBottom();
            rafId = null;
          });
        });
      }
    });

    observer.observe(container, {
      childList: true,
      subtree: true,
      characterData: true,
    });

    return () => {
      observer.disconnect();
      if (rafId) cancelAnimationFrame(rafId);
      if (timeoutId) clearTimeout(timeoutId);
    };
  }, [scrollToBottom, isAtBottom]);

  // Trigger scroll when content actually changes - only if auto-scroll is enabled
  useEffect(() => {
    if (agents.length > 0 && shouldAutoScrollRef.current) {
      // Use RAF to ensure DOM has updated before scrolling
      requestAnimationFrame(() => {
        requestAnimationFrame(() => {
          scrollToBottom();
        });
      });
    }
  }, [agentsContentKey, scrollToBottom]);

  // Initial scroll to bottom when first agents appear
  useEffect(() => {
    if (agents.length > 0 && shouldAutoScrollRef.current) {
      // Small delay to ensure DOM is ready
      const timeoutId = setTimeout(() => {
        scrollToBottom();
      }, 100);
      
      return () => clearTimeout(timeoutId);
    }
  }, [agents.length, scrollToBottom]);

  // Count unique agents
  const uniqueAgentCount = useMemo(() => {
    return new Set(agents.map(a => a.agentName)).size;
  }, [agents]);

  // Show all agents, sorted chronologically (oldest first, newest at bottom)
  const sortedAgents = useMemo(() => {
    if (agents.length === 0) return [];
    
    // Sort by creation time (oldest first) so new ones appear at bottom
    return [...agents].sort((a, b) => {
      // Use the timestamp from the ID or lastUpdate
      return a.lastUpdate.getTime() - b.lastUpdate.getTime();
    });
  }, [agents]);

  return (
    <Card className="flex flex-col overflow-hidden h-full console-panel bg-card/30 backdrop-blur-xl border-border/50 dark:border-white/10 glass-card">
      {/* Panel Header */}
      <CardHeader className="bg-gradient-to-r from-muted/60 to-muted/40 dark:from-zinc-950/40 dark:to-zinc-900/30 backdrop-blur-md border-b border-border/50 dark:border-white/10 py-3 px-5 glass-header">
        <div className="flex justify-between items-center">
          <CardTitle className="text-xl font-bold text-foreground">Raw Agent Traces</CardTitle>
          <div className="flex gap-2 items-center">
            <Badge variant="secondary" className="text-sm">
              {totalWords.toLocaleString()} word{totalWords !== 1 ? 's' : ''}
            </Badge>
            <Badge variant="secondary" className="text-sm">
              {uniqueAgentCount} agent{uniqueAgentCount !== 1 ? 's' : ''}
            </Badge>
          </div>
        </div>
      </CardHeader>

      {/* Console Content - Scrollable */}
      <CardContent className="flex-1 overflow-hidden p-0 relative">
        <div
          ref={consoleRef}
          className="console-container h-full overflow-y-auto custom-scrollbar"
        >
          {agents.length === 0 ? (
            <div className="console-empty p-8 text-center h-full flex items-center justify-center">
              <p className="text-base" style={{ color: 'var(--console-text)', opacity: 0.6 }}>
                No traces yet. Process a case to see verbose reasoning traces.
              </p>
            </div>
          ) : (
            <div className="console-logs flex flex-col">
              {sortedAgents.map((agent) => {
                const agentColor = agentColors.get(agent.agentName) || '#60a5fa';
                const isActive = activeAgents.has(agent.agentName);
                return (
                  <div key={agent.id} className="flex-shrink-0">
                    <ConsoleLogEntry
                      agentName={agent.agentName}
                      content={agent.fullText}
                      agentColor={agentColor}
                      isActive={isActive}
                    />
                  </div>
                );
              })}
            </div>
          )}
        </div>
        
        {/* "Go to Latest" button - positioned at bottom right */}
        {showGoToLatest && (
          <div className="absolute bottom-4 right-4 z-20 animate-in fade-in slide-in-from-bottom-2 duration-200">
            <Button
              onClick={handleGoToLatest}
              variant="default"
              size="sm"
              className="shadow-2xl backdrop-blur-md bg-primary hover:bg-primary/90 text-primary-foreground border-2 border-primary/30 font-semibold px-4 py-2.5 min-w-[140px] ring-2 ring-primary/20 hover:ring-primary/40 transition-all"
            >
              <span className="mr-2">↓</span>
              Go to Latest
            </Button>
          </div>
        )}
      </CardContent>
    </Card>
  );
}

