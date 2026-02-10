// Type definitions for EXAID CDSS Frontend

export interface AgentTrace {
  id: string;  // Unique ID for React key
  agentName: string;  // Base agent name for display
  runId: string | null;  // Unique run ID for this agent invocation
  fullText: string;
  isExpanded: boolean;
  lastUpdate: Date;
}

export interface SummaryData {
  status_action: string;
  key_findings: string;
  differential_rationale: string;
  uncertainty_confidence: string;
  recommendation_next_step: string;
  agent_contributions: string;
  
  // Optional full versions (populated when truncation occurs)
  full_status_action?: string | null;
  full_key_findings?: string | null;
  full_differential_rationale?: string | null;
  full_uncertainty_confidence?: string | null;
  full_recommendation_next_step?: string | null;
  full_agent_contributions?: string | null;
}

export interface Summary {
  id: string;
  data: SummaryData;
  timestamp: Date;
  isExpanded: boolean;
}

export type ConnectionStatus = 'connecting' | 'connected' | 'disconnected' | 'error';

export interface ModalState {
  isOpen: boolean;
  agentId: string | null;
  content: string;
}

export type DemoMode = "live_demo" | "trace_replay";

export interface TraceFile {
  case_id: string;
  file_path: string;
}

export interface CaseRequest {
  mode: DemoMode;
  case?: string;
  trace_file?: string;
}

// WebSocket message types
export interface AgentStartedMessage {
  type: 'agent_started';
  agent_id: string;
  run_id: string;  // Unique run ID for this agent invocation
  timestamp: string;
}

export interface TokenMessage {
  type: 'token';
  agent_id: string;
  run_id: string | null;  // Run ID to match token to correct card
  token: string;
  timestamp: string;
}

export interface SummaryMessage {
  type: 'summary';
  summary_data: SummaryData;
  timestamp: string;
}

export interface ProcessingStartedMessage {
  type: 'processing_started';
  timestamp: string;
}

export interface ProcessingCompleteMessage {
  type: 'processing_complete';
  timestamp: string;
}

export interface ProcessingStoppedMessage {
  type: 'processing_stopped';
  timestamp: string;
}

export interface ErrorMessage {
  type: 'error';
  message: string;
  timestamp: string;
}

export type WebSocketMessage = 
  | AgentStartedMessage
  | TokenMessage 
  | SummaryMessage 
  | ProcessingStartedMessage 
  | ProcessingCompleteMessage 
  | ProcessingStoppedMessage
  | ErrorMessage;

