import { create } from 'zustand';
import { subscribeWithSelector } from 'zustand/middleware';
import type { AgentTrace, Summary, ConnectionStatus, ModalState, SummaryData } from '@/lib/types';

interface CDSSState {
  // Agent traces (array for multiple invocations of same agent)
  agents: AgentTrace[];
  
  // Summaries
  summaries: Summary[];
  summaryIdCounter: number;
  
  // WebSocket state
  wsStatus: ConnectionStatus;
  reconnectAttempts: number;
  
  // Processing state
  isProcessing: boolean;
  
  // Word counting
  totalWords: number;
  totalSummaryWords: number;
  
  // Active agents (currently streaming)
  activeAgents: Set<string>;
  
  // Modal state
  modal: ModalState;
  
  // Actions
  startNewAgent: (agentId: string, runId: string) => void;
  addToken: (agentId: string, runId: string | null, token: string) => void;
  addSummary: (data: SummaryData, timestamp: Date) => void;
  toggleAgent: (cardId: string) => void;
  toggleSummary: (id: string) => void;
  setWsStatus: (status: ConnectionStatus) => void;
  setReconnectAttempts: (attempts: number) => void;
  setProcessing: (isProcessing: boolean) => void;
  openModal: (cardId: string, content: string) => void;
  closeModal: () => void;
  resetState: () => void;
}

// Helper function to count words in a string
// Only counts actual content words, not labels or formatting
function countWords(text: string): number {
  if (!text || typeof text !== 'string') return 0;
  
  // Trim whitespace and split by whitespace
  const trimmed = text.trim();
  if (trimmed.length === 0) return 0;
  
  // Split by whitespace (spaces, tabs, newlines) and filter out empty strings
  // This counts only the actual content words in the field value
  const words = trimmed.split(/\s+/).filter(word => word.length > 0);
  return words.length;
}

// Timeout map for active agent cleanup (outside Zustand store)
const activeAgentTimeouts = new Map<string, NodeJS.Timeout>();
const ACTIVE_AGENT_TIMEOUT_MS = 2000; // 2 seconds

// Helper function to update card text and track active agents
// Extracted to reduce code duplication between auto-creation path and normal update path
function updateCardWithToken(
  state: CDSSState,
  cardIndex: number,
  agentId: string,
  token: string,
  set: (partial: Partial<CDSSState> | ((state: CDSSState) => Partial<CDSSState>)) => void
): Partial<CDSSState> {
  const newAgents = [...state.agents];
  const oldFullText = newAgents[cardIndex].fullText;
  const newFullText = oldFullText + token;
  
  newAgents[cardIndex] = {
    ...newAgents[cardIndex],
    fullText: newFullText,
    lastUpdate: new Date(),
  };
  
  // Count words in the accumulated text (not just the token)
  // This correctly handles cases where tokens don't align with word boundaries
  // (e.g., partial words, punctuation-only tokens, multi-word tokens)
  const oldWordCount = countWords(oldFullText);
  const newWordCount = countWords(newFullText);
  const wordsAdded = newWordCount - oldWordCount;
  
  // Track active agent
  const newActiveAgents = new Set(state.activeAgents);
  newActiveAgents.add(agentId);
  
  // Clear existing timeout for this agent
  const existingTimeout = activeAgentTimeouts.get(agentId);
  if (existingTimeout) {
    clearTimeout(existingTimeout);
  }
  
  // Set new timeout to remove agent from active set
  const timeoutId = setTimeout(() => {
    set((state) => {
      const updatedActiveAgents = new Set(state.activeAgents);
      updatedActiveAgents.delete(agentId);
      activeAgentTimeouts.delete(agentId);
      return { activeAgents: updatedActiveAgents };
    });
  }, ACTIVE_AGENT_TIMEOUT_MS);
  
  activeAgentTimeouts.set(agentId, timeoutId);
  
  return {
    agents: newAgents,
    totalWords: state.totalWords + wordsAdded,
    activeAgents: newActiveAgents,
  };
}

export const useCDSSStore = create<CDSSState>()(
  subscribeWithSelector((set, get) => ({
    // Initial state
    agents: [],
    summaries: [],
    summaryIdCounter: 0,
    wsStatus: 'disconnected',
    reconnectAttempts: 0,
    isProcessing: false,
    totalWords: 0,
    totalSummaryWords: 0,
    activeAgents: new Set<string>(),
    modal: {
      isOpen: false,
      agentId: null,
      content: '',
    },
    
    // Start a new agent invocation (creates new card)
    startNewAgent: (agentId: string, runId: string) => {
      set((state) => {
        const newCard: AgentTrace = {
          id: `${agentId}_${state.agents.length}_${Date.now()}`,  // Unique ID for React key
          agentName: agentId,              // Base name for display
          runId: runId,                    // Unique run ID for this invocation
          fullText: '',
          isExpanded: false,
          lastUpdate: new Date(),
        };
        
        // Add new card at the end (append to bottom)
        return {
          agents: [...state.agents, newCard],
        };
      });
    },
    
    // Add token immediately - no batching, streams token-by-token
    addToken: (agentId: string, runId: string | null, token: string) => {
      const state = get();
      
      // Find the card matching both agent name and run_id
      // If runId is provided, match by it; otherwise fall back to most recent card for this agent
      let cardIndex = -1;
      if (runId) {
        cardIndex = state.agents.findIndex(a => a.agentName === agentId && a.runId === runId);
      }
      
      // Fallback: if no runId match or runId is null, use most recent card for this agent
      if (cardIndex === -1) {
        cardIndex = state.agents.findLastIndex(a => a.agentName === agentId);
      }
      
      if (cardIndex === -1) {
        // No card exists for this agent. This may indicate a missing agent_started message or a race condition.
        console.warn(`No active card found for agent '${agentId}'${runId ? ` with run_id '${runId}'` : ''}. This may indicate a missing agent_started message or a race condition. Auto-creating card to prevent token loss.`);
        // Generate a run_id if not provided using crypto.randomUUID() for better uniqueness guarantees
        const newRunId = runId || `${agentId}_${Date.now()}_${crypto.randomUUID()}`;
        get().startNewAgent(agentId, newRunId);
        // After creating, retrieve the card we just created
        const updatedState = get();
        const updatedCardIndex = updatedState.agents.findIndex(a => a.agentName === agentId && a.runId === newRunId);
        if (updatedCardIndex === -1) {
          // Still not found, abort
          console.error(`Failed to auto-create card for agent '${agentId}'. Token will be lost.`);
          return;
        }
        // Update the newly created card immediately with the token
        set((currentState) => updateCardWithToken(currentState, updatedCardIndex, agentId, token, set));
        return;
      }
      
      // Update the card immediately with the token
      set((currentState) => updateCardWithToken(currentState, cardIndex, agentId, token, set));
    },
    
    // Add new summary
    addSummary: (data: SummaryData, timestamp: Date) => {
      set((state) => {
        const newSummary: Summary = {
          id: `summary-${state.summaryIdCounter}`,
          data,
          timestamp,
          isExpanded: true,
        };
        
        // Calculate words in the new summary
        // Only count words in the field values themselves, not labels or formatting
        // Field values are plain strings containing only the content
        const summaryFields = [
          data.status_action,
          data.key_findings,
          data.differential_rationale,
          data.uncertainty_confidence,
          data.recommendation_next_step,
          data.agent_contributions,
        ];
        const wordsInSummary = summaryFields.reduce((sum, field) => {
          // countWords only counts actual content words in the field value string
          return sum + countWords(field);
        }, 0);
        
        // Collapse all existing summaries
        const updatedSummaries = state.summaries.map(s => ({
          ...s,
          isExpanded: false,
        }));
        
        // Add new summary at the beginning (newest first)
        return {
          summaries: [newSummary, ...updatedSummaries],
          summaryIdCounter: state.summaryIdCounter + 1,
          totalSummaryWords: state.totalSummaryWords + wordsInSummary,
        };
      });
    },
    
    // Toggle agent expand/collapse
    toggleAgent: (cardId: string) => {
      set((state) => {
        const newAgents = [...state.agents];
        const cardIndex = newAgents.findIndex(a => a.id === cardId);
        if (cardIndex !== -1) {
          newAgents[cardIndex] = {
            ...newAgents[cardIndex],
            isExpanded: !newAgents[cardIndex].isExpanded,
          };
        }
        return { agents: newAgents };
      });
    },
    
    // Toggle summary expand/collapse (spotlight swap behavior)
    toggleSummary: (id: string) => {
      set((state) => {
        // Find the currently expanded summary (in spotlight)
        const currentSpotlight = state.summaries.find(s => s.isExpanded);
        
        // If clicking the current spotlight, do nothing (or could collapse it)
        if (currentSpotlight?.id === id) {
          return { summaries: state.summaries };
        }
        
        // Swap: expand clicked summary, collapse all others
        const updatedSummaries = state.summaries.map(summary => {
          if (summary.id === id) {
            return { ...summary, isExpanded: true };
          }
          return { ...summary, isExpanded: false };
        });
        
        return { summaries: updatedSummaries };
      });
    },
    
    // Set WebSocket connection status
    setWsStatus: (status: ConnectionStatus) => {
      set({ wsStatus: status });
    },
    
    // Set reconnect attempts
    setReconnectAttempts: (attempts: number) => {
      set({ reconnectAttempts: attempts });
    },
    
    // Set processing state
    setProcessing: (isProcessing: boolean) => {
      set({ isProcessing });
    },
    
    // Open modal with agent trace
    openModal: (cardId: string, content: string) => {
      set({
        modal: {
          isOpen: true,
          agentId: cardId,  // Store cardId for modal
          content,
        },
      });
    },
    
    // Close modal
    closeModal: () => {
      set({
        modal: {
          isOpen: false,
          agentId: null,
          content: '',
        },
      });
    },
    
    // Reset state (called on processing_started)
    resetState: () => {
      // Clear all active agent timeouts
      activeAgentTimeouts.forEach((timeout) => clearTimeout(timeout));
      activeAgentTimeouts.clear();
      
      set({
        agents: [],
        summaries: [],
        summaryIdCounter: 0,
        totalWords: 0,
        totalSummaryWords: 0,
        activeAgents: new Set<string>(),
        modal: {
          isOpen: false,
          agentId: null,
          content: '',
        },
      });
    },
  }))
);

// Selector hooks for optimized subscriptions
export const useAgentTrace = (cardId: string) => {
  return useCDSSStore((state) => state.agents.find(a => a.id === cardId));
};

export const useAllAgents = () => {
  return useCDSSStore((state) => state.agents);
};

export const useSummaries = () => {
  return useCDSSStore((state) => state.summaries);
};

export const useWsStatus = () => {
  return useCDSSStore((state) => state.wsStatus);
};

export const useIsProcessing = () => {
  return useCDSSStore((state) => state.isProcessing);
};

export const useModal = () => {
  return useCDSSStore((state) => state.modal);
};

export const useTotalWords = () => {
  return useCDSSStore((state) => state.totalWords);
};

export const useTotalSummaryWords = () => {
  return useCDSSStore((state) => state.totalSummaryWords);
};

export const useActiveAgents = () => {
  return useCDSSStore((state) => state.activeAgents);
};

