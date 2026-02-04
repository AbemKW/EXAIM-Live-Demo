'use client';

import React from 'react';

interface ConsoleLogEntryProps {
  agentName: string;
  content: string;
  agentColor: string;
  isActive?: boolean;
}

function ConsoleLogEntry({ agentName, content, agentColor, isActive = false }: ConsoleLogEntryProps) {
  return (
    <div className={`console-log-entry ${isActive ? 'agent-active-glow' : ''}`}>
      <div 
        className="console-agent-label-sticky"
        style={{ 
          color: agentColor,
          ...(isActive && {
            boxShadow: `0 0 12px ${agentColor}40, 0 0 24px ${agentColor}20`,
          }),
        }}
      >
        {agentName}
      </div>
      <div className="console-content-wrapper">
        <span className="console-content">{content}</span>
      </div>
    </div>
  );
}

export default React.memo(ConsoleLogEntry);

