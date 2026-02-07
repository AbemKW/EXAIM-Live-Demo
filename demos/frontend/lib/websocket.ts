import { useCDSSStore } from '@/store/cdssStore';
import type { WebSocketMessage, SummaryData } from '@/lib/types';

// WebSocket URL validation - must be ws:// or wss://
function validateWebSocketUrl(url: string | undefined): string {
  // If URL is provided, use it
  if (url) {
    if (!url.startsWith('ws://') && !url.startsWith('wss://')) {
      throw new Error(
        `Invalid WebSocket URL: "${url}". URL must start with ws:// or wss://`
      );
    }
    
    // Validate URL format
    try {
      new URL(url);
    } catch {
      throw new Error(
        `Invalid WebSocket URL format: "${url}". Please provide a valid URL.`
      );
    }
    
    return url;
  }
  
  // Auto-detect WebSocket URL based on current window location
  if (typeof window !== 'undefined') {
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const host = window.location.host; // includes port if present
    const wsUrl = `${protocol}//${host}/ws`;
    return wsUrl;
  }
  
  // Fallback for SSR/build time
  return 'ws://localhost:7860/ws';
}

// Constants
const MAX_RECONNECT_ATTEMPTS = 5;
const RECONNECT_DELAY_MS = 3000;

/**
 * WebSocket service for real-time communication with the EXAID backend.
 * Implements singleton pattern to prevent duplicate connections.
 */
class WebSocketService {
  private ws: WebSocket | null = null;
  private reconnectAttempts = 0;
  private reconnectTimeout: NodeJS.Timeout | null = null;
  private intentionalClose = false;
  private wsUrl: string | null = null;
  private validationError: string | null = null;

  constructor() {
    // Delay URL resolution until connect() is called (client-side only)
    // This ensures window.location is available for auto-detection
  }

  /**
   * Connect to the WebSocket server.
   * Validates the URL and handles automatic reconnection on disconnect.
   */
  connect(): void {
    const store = useCDSSStore.getState();
    
    // Resolve WebSocket URL on first connect (client-side)
    if (!this.wsUrl) {
      try {
        this.wsUrl = validateWebSocketUrl(process.env.NEXT_PUBLIC_WS_URL);
      } catch (error) {
        this.validationError = error instanceof Error ? error.message : String(error);
        console.error('WebSocket URL validation failed:', this.validationError);
      }
    }
    
    // Check for validation errors
    if (this.validationError || !this.wsUrl) {
      console.error('Cannot connect: WebSocket URL is invalid or not configured');
      console.error(this.validationError || 'WebSocket URL could not be determined');
      store.setWsStatus('error');
      return;
    }
    
    // Prevent duplicate connections
    if (this.ws?.readyState === WebSocket.CONNECTING || 
        this.ws?.readyState === WebSocket.OPEN) {
      return;
    }

    this.intentionalClose = false;
    store.setWsStatus('connecting');

    try {
      this.ws = new WebSocket(this.wsUrl);
      this.setupEventHandlers();
    } catch (error) {
      console.error('Failed to create WebSocket:', error);
      store.setWsStatus('error');
      this.scheduleReconnect();
    }
  }

  /**
   * Disconnect from the WebSocket server.
   * Clears reconnection attempts.
   */
  disconnect(): void {
    this.intentionalClose = true;
    
    if (this.reconnectTimeout) {
      clearTimeout(this.reconnectTimeout);
      this.reconnectTimeout = null;
    }
    
    if (this.ws) {
      this.ws.close();
      this.ws = null;
    }
    
    const store = useCDSSStore.getState();
    store.setWsStatus('disconnected');
    store.setReconnectAttempts(0);
    this.reconnectAttempts = 0;
  }

  /**
   * Set up WebSocket event handlers.
   */
  private setupEventHandlers(): void {
    if (!this.ws) return;

    this.ws.onopen = () => {
      console.log('WebSocket connected');
      const store = useCDSSStore.getState();
      store.setWsStatus('connected');
      store.setReconnectAttempts(0);
      this.reconnectAttempts = 0;
    };

    this.ws.onclose = (event) => {
      // Normal closure codes: 1000 (normal), 1001 (going away), 1006 (abnormal - no close frame)
      const isNormalClose = event.code === 1000 || event.code === 1001;
      const isAbnormalClose = event.code === 1006; // Common during server restarts
      
      if (isNormalClose) {
        console.log('WebSocket closed normally');
      } else if (isAbnormalClose) {
        console.log('WebSocket closed abnormally (server may have restarted)');
      } else {
        console.log('WebSocket closed:', event.code, event.reason || 'No reason provided');
      }
      
      const store = useCDSSStore.getState();
      store.setWsStatus('disconnected');
      
      // Always attempt reconnection unless intentionally closed
      // This handles server restarts gracefully
      if (!this.intentionalClose) {
        this.scheduleReconnect();
      }
      
      // Clean up the WebSocket reference after checking intentionalClose
      // This prevents a race condition where connect() might be called during
      // the reconnection delay and create duplicate connections
      this.ws = null;
    };

    this.ws.onerror = (error) => {
      // Always log WebSocket errors at warning level for visibility
      // This helps with debugging connection issues during development and production
      const errorInfo = error instanceof Error ? error.message : 
                       (error && typeof error === 'object' && Object.keys(error).length > 0) ? error : null;
      
      if (errorInfo) {
        console.warn('WebSocket error:', errorInfo);
      } else {
        // Even without specific error details, log at warning level for visibility
        console.warn('WebSocket connection error (no specific error information provided - likely connection interruption)');
      }
      
      const store = useCDSSStore.getState();
      // Don't set error status immediately - let onclose handle reconnection
      // This prevents showing error state during normal server restarts
      if (this.ws?.readyState === WebSocket.CLOSED) {
        store.setWsStatus('disconnected');
        // Schedule reconnection if not intentional close
        if (!this.intentionalClose) {
          this.scheduleReconnect();
        }
      }
    };

    this.ws.onmessage = (event) => {
      try {
        const message: WebSocketMessage = JSON.parse(event.data);
        this.handleMessage(message);
      } catch (error) {
        console.error('Failed to parse WebSocket message:', error);
      }
    };
  }

  /**
   * Handle incoming WebSocket messages.
   */
  private handleMessage(message: WebSocketMessage): void {
    const store = useCDSSStore.getState();

    switch (message.type) {
      case 'agent_started':
        // Create a new card for this agent invocation with the provided run_id
        store.startNewAgent(message.agent_id, message.run_id);
        break;

      case 'token':
        // Add token to the card matching the run_id
        store.addToken(message.agent_id, message.run_id, message.token);
        break;

      case 'summary':
        console.log('WebSocket: Received summary message', message);
        store.addSummary(
          message.summary_data as SummaryData,
          new Date(message.timestamp)
        );
        console.log('WebSocket: Summary added to store');
        break;

      case 'processing_started':
        store.resetState();
        store.setProcessing(true);
        break;

      case 'processing_complete':
        store.setProcessing(false);
        break;

      case 'processing_stopped':
        store.resetState();
        store.setProcessing(false);
        break;

      case 'error':
        console.error('Server error:', message.message);
        store.setProcessing(false);
        break;

      default:
        console.warn('Unknown message type:', message);
    }
  }

  /**
   * Schedule a reconnection attempt with fixed delay.
   */
  private scheduleReconnect(): void {
    if (this.reconnectAttempts >= MAX_RECONNECT_ATTEMPTS) {
      console.error('Max reconnection attempts reached');
      return;
    }

    this.reconnectAttempts++;
    const store = useCDSSStore.getState();
    store.setReconnectAttempts(this.reconnectAttempts);

    console.log(
      `Scheduling reconnect attempt ${this.reconnectAttempts}/${MAX_RECONNECT_ATTEMPTS} in ${RECONNECT_DELAY_MS}ms`
    );

    this.reconnectTimeout = setTimeout(() => {
      this.connect();
    }, RECONNECT_DELAY_MS);
  }

  /**
   * Get current connection state.
   */
  isConnected(): boolean {
    return this.ws?.readyState === WebSocket.OPEN;
  }
}

// Singleton instance
let wsService: WebSocketService | null = null;

/**
 * Get the WebSocket service singleton instance.
 * Creates the instance on first call.
 */
export function getWebSocketService(): WebSocketService {
  if (!wsService) {
    wsService = new WebSocketService();
  }
  return wsService;
}

