'use client';

import { useWsStatus, useCDSSStore, useActiveAgents } from '@/store/cdssStore';
import { Badge } from '@/components/ui/badge';
import TypingIndicator from './TypingIndicator';
import ThemeToggle from './ThemeToggle';
import { useTheme } from '@/hooks/useTheme';

export default function Header() {
  const wsStatus = useWsStatus();
  const reconnectAttempts = useCDSSStore((state) => state.reconnectAttempts);
  const activeAgents = useActiveAgents();
  const activeAgentCount = activeAgents.size;
  const { theme } = useTheme();

  const getStatusVariant = () => {
    switch (wsStatus) {
      case 'connected':
        return 'default';
      case 'connecting':
        return 'secondary';
      case 'disconnected':
        return 'destructive';
      default:
        return 'outline';
    }
  };

  const getStatusText = () => {
    if (wsStatus === 'connecting' && reconnectAttempts > 0) {
      return `Reconnecting (${reconnectAttempts}/5)`;
    }
    return wsStatus.charAt(0).toUpperCase() + wsStatus.slice(1);
  };

  const getStatusDotClass = () => {
    switch (wsStatus) {
      case 'connected':
        return 'bg-green-600/70 animate-pulse';
      case 'connecting':
        return 'bg-amber-600/70';
      case 'disconnected':
        return 'bg-red-700/70';
      default:
        return 'bg-zinc-500';
    }
  };

  return (
    <header className="fixed-header flex items-center px-6">
      <div className="max-w-[1800px] w-full mx-auto flex justify-between items-center h-full">
        <div className="flex items-center gap-3">
          <img
            src={theme === 'dark' ? '/EXAIMLogo.png' : '/EXAIMLogoLightTheme.png'}
            alt="EXAIM Logo"
            width={200}
            height={200}
            className="object-contain"
            style={{ display: 'block', flexShrink: 0 }}
            onError={(e) => {
              console.error('Failed to load logo:', e);
              (e.target as HTMLImageElement).style.display = 'none';
            }}
          />
          <p className="text-sm text-muted-foreground font-medium" style={{ fontFamily: 'var(--font-inter), sans-serif', marginTop: '32px' }}>
            A Real-Time Explainability Middleware
          </p>
        </div>
        <div className="flex items-center gap-3">
          {activeAgentCount > 0 && (
            <Badge
              variant="secondary"
              className="flex items-center gap-2 text-sm px-3 py-1 active-agents-badge"
            >
              <TypingIndicator />
              <span className="font-medium">
                {activeAgentCount} Agent{activeAgentCount !== 1 ? 's' : ''} Active
              </span>
            </Badge>
          )}
          <Badge
            variant={getStatusVariant()}
            className="flex items-center gap-2 uppercase tracking-wide text-sm px-3 py-1"
          >
            <span className={`w-2 h-2 rounded-full ${getStatusDotClass()}`} />
            {getStatusText()}
          </Badge>
          <ThemeToggle />
        </div>
      </div>
    </header>
  );
}

