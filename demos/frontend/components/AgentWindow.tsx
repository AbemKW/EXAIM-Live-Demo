'use client';

import React from 'react';
import { motion } from 'framer-motion';
import { useCDSSStore, useAgentTrace } from '@/store/cdssStore';
import AgentWindowContent from './AgentWindowContent';
import { Card, CardHeader, CardTitle, CardContent } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';

interface AgentWindowProps {
  cardId: string;
}

function AgentWindow({ cardId }: AgentWindowProps) {
  const agent = useAgentTrace(cardId);
  const toggleAgent = useCDSSStore((state) => state.toggleAgent);
  const openModal = useCDSSStore((state) => state.openModal);

  if (!agent) return null;

  const handleToggle = () => {
    toggleAgent(cardId);
  };

  const handleViewFull = (e: React.MouseEvent) => {
    e.stopPropagation();
    openModal(cardId, agent.fullText);
  };

  return (
    <motion.div
      layout
      initial={{ opacity: 0, y: -20 }}
      animate={{ opacity: 1, y: 0 }}
    >
      <Card className="overflow-hidden border-border/50 dark:border-blue-900/10">
        {/* Header */}
        <CardHeader
          onClick={handleToggle}
          className="cursor-pointer hover:bg-accent/50 dark:hover:bg-blue-950/20 transition-colors border-b border-border/30 dark:border-blue-900/15 py-3 px-4"
        >
          <div className="flex justify-between items-center">
            <div className="flex items-center gap-3">
              <CardTitle className="text-base font-semibold">{agent.agentName}</CardTitle>
              <Badge variant="default" className="bg-green-600 dark:bg-green-700/60 hover:bg-green-600/90 dark:hover:bg-green-700/60 text-sm px-2 py-1">
                Active
              </Badge>
            </div>
            <div className="flex items-center gap-3">
              <Button
                onClick={handleViewFull}
                variant="secondary"
                size="sm"
                className="h-8 px-3 text-sm"
              >
                View Full
              </Button>
              <motion.span
                animate={{ rotate: agent.isExpanded ? 180 : 0 }}
                transition={{ duration: 0.3 }}
                className="text-muted-foreground text-base"
              >
                ▼
              </motion.span>
            </div>
          </div>
        </CardHeader>

        {/* Content */}
        <motion.div
          initial={false}
          animate={{
            height: agent.isExpanded ? '400px' : '180px',
          }}
          transition={{ duration: 0.3, ease: 'easeInOut' }}
          className="overflow-hidden"
        >
          <CardContent className="p-0 h-full">
            <AgentWindowContent
              fullText={agent.fullText}
              isExpanded={agent.isExpanded}
            />
          </CardContent>
        </motion.div>
      </Card>
    </motion.div>
  );
}

// Memoize to prevent unnecessary re-renders
export default React.memo(AgentWindow);

