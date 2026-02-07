import asyncio
import logging
import subprocess
import sys
import os
import uuid
import time
import httpx
from pathlib import Path
from typing import List, Dict
from datetime import datetime
from contextlib import asynccontextmanager

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, model_validator
from typing import Literal, Optional

from demos.cdss_example.cdss import CDSS
from demos.cdss_example.message_bus import message_queue

# Try to import trace replay engine for trace_replay mode (optional for live demo)
try:
    # Add evals/src to path for trace replay engine
    sys.path.insert(0, str(Path(__file__).parent.parent.parent / "evals" / "src"))
    from traces.trace_replay_engine import TraceReplayEngine, ReplayEvent
    TRACE_REPLAY_AVAILABLE = True
except ImportError:
    TRACE_REPLAY_AVAILABLE = False
    logger = logging.getLogger(__name__)
    logger.warning("Trace replay engine not available - trace_replay mode disabled")

from exaim_core.exaim import EXAIM


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup/shutdown events."""
    # Startup: Start background message broadcaster
    broadcaster_task = asyncio.create_task(message_broadcaster())
    
    # Model loading handled by vLLM server (started via supervisord)
    # Backend waits for vLLM readiness via start_backend.sh probe
    
    yield
    # Shutdown: Cancel background task
    broadcaster_task.cancel()
    try:
        await broadcaster_task
    except asyncio.CancelledError:
        pass  # Expected during shutdown - task was cancelled gracefully


# Configure logging
logger = logging.getLogger(__name__)

# Health check cache
_health_cache = {
    "status": "unknown",
    "vllm": "unknown",
    "timestamp": 0.0,
}
_health_cache_ttl = 10.0  # Cache for 10 seconds

app = FastAPI(lifespan=lifespan)

# Add CORS middleware for Next.js frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",  # Next.js dev server default
        "http://localhost:3001",  # Next.js dev server alternate
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Store active WebSocket connections
active_connections: List[WebSocket] = []
# Lock for synchronizing access to active_connections
connections_lock = asyncio.Lock()

# Track active run IDs per agent (agent_id -> run_id)
# This allows us to associate tokens with the correct agent invocation
active_run_ids: Dict[str, str] = {}
run_ids_lock = asyncio.Lock()


async def get_active_connections() -> List[WebSocket]:
    """Get a copy of the active connections list safely."""
    async with connections_lock:
        return active_connections.copy()


async def add_connection(websocket: WebSocket):
    """Add a connection to the active connections list safely."""
    async with connections_lock:
        active_connections.append(websocket)


async def remove_connection(websocket: WebSocket):
    """Remove a connection from the active connections list safely."""
    async with connections_lock:
        if websocket in active_connections:
            active_connections.remove(websocket)


@app.get("/health")
async def health_check():
    """Health check endpoint that verifies vLLM availability.
    
    Returns:
        200 OK if backend and vLLM are both healthy
        503 Service Unavailable if vLLM is not available
    
    Response is cached for 10 seconds to reduce load on vLLM.
    """
    global _health_cache
    
    current_time = time.time()
    
    # Return cached result if still valid
    if current_time - _health_cache["timestamp"] < _health_cache_ttl:
        if _health_cache["status"] == "healthy":
            return {
                "status": _health_cache["status"],
                "vllm": _health_cache["vllm"],
                "cached": True
            }
        else:
            raise HTTPException(
                status_code=503,
                detail={
                    "status": _health_cache["status"],
                    "vllm": _health_cache["vllm"],
                    "cached": True
                }
            )
    
    # Check vLLM availability
    vllm_url = os.environ.get("OPENAI_BASE_URL", "http://localhost:8001/v1")
    models_endpoint = f"{vllm_url}/models"
    
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.get(models_endpoint)
            
            if response.status_code == 200:
                # vLLM is healthy
                _health_cache = {
                    "status": "healthy",
                    "vllm": "ready",
                    "timestamp": current_time
                }
                return {
                    "status": "healthy",
                    "vllm": "ready",
                    "cached": False
                }
            else:
                # vLLM returned non-200 status
                _health_cache = {
                    "status": "unhealthy",
                    "vllm": "unavailable",
                    "timestamp": current_time
                }
                raise HTTPException(
                    status_code=503,
                    detail={
                        "status": "unhealthy",
                        "vllm": "unavailable",
                        "cached": False
                    }
                )
    except Exception as e:
        # vLLM connection failed
        logger.warning(f"vLLM health check failed: {e}")
        _health_cache = {
            "status": "unhealthy",
            "vllm": "unavailable",
            "timestamp": current_time
        }
        raise HTTPException(
            status_code=503,
            detail={
                "status": "unhealthy",
                "vllm": "unavailable",
                "error": str(e),
                "cached": False
            }
        )


# Global CDSS instance for the current request.
# Note: This is intentionally request-scoped - each POST to /api/process-case creates
# a new instance. Access is synchronized using cdss_lock to prevent race conditions
# when multiple requests arrive simultaneously.
cdss_instance: CDSS = None
cdss_lock = asyncio.Lock()
cdss_process_task: asyncio.Task = None
should_stop = False
cancellation_event = asyncio.Event()

# Maximum allowed length for case text to prevent resource abuse
MAX_CASE_LENGTH = 100000


class CaseRequest(BaseModel):
    mode: Literal["live_demo", "trace_replay"] = "live_demo"
    case: Optional[str] = None
    trace_file: Optional[str] = None
    
    @model_validator(mode='after')
    def validate_request(self):
        """Validate request based on mode."""
        if self.mode == "live_demo":
            if not self.case or not self.case.strip():
                raise ValueError("Case text cannot be empty for live_demo mode")
            if len(self.case) > MAX_CASE_LENGTH:
                raise ValueError(f"Case text exceeds maximum length of {MAX_CASE_LENGTH} characters")
            self.case = self.case.strip()
        elif self.mode == "trace_replay":
            if not self.trace_file or not self.trace_file.strip():
                raise ValueError("trace_file is required for trace_replay mode")
            self.trace_file = self.trace_file.strip()
        return self


async def send_token_direct(agent_id: str, token: str):
    """Send token directly to WebSocket clients without queuing.
    
    This bypasses the message queue for immediate token-by-token delivery.
    Includes the active run_id for this agent to ensure tokens are routed
    to the correct agent invocation card.
    """
    connections = await get_active_connections()
    if not connections:
        return
    
    # Look up the active run_id for this agent
    async with run_ids_lock:
        run_id = active_run_ids.get(agent_id)
    
    # If no run_id found, log warning but still send token
    # (frontend will handle gracefully)
    if not run_id:
        logger.warning(f"No active run_id found for agent '{agent_id}'. Token may be routed incorrectly.")
    
    message = {
        "type": "token",
        "agent_id": agent_id,
        "run_id": run_id,  # Include run_id to match tokens to correct card
        "token": token,
        "timestamp": datetime.now().isoformat()
    }
    
    disconnected = []
    for connection in connections:
        try:
            await connection.send_json(message)
        except Exception as e:
            logger.warning(f"Error sending token to client: {e}")
            disconnected.append(connection)
    
    # Remove disconnected clients
    for conn in disconnected:
        await remove_connection(conn)


async def send_summary_direct(summary):
    """Send summary directly to WebSocket clients.
    
    Args:
        summary: AgentSummary Pydantic object to broadcast
    """
    connections = await get_active_connections()
    if not connections:
        return
    
    # Convert Pydantic model to dict
    summary_dict = {
        "status_action": summary.status_action,
        "key_findings": summary.key_findings,
        "differential_rationale": summary.differential_rationale,
        "uncertainty_confidence": summary.uncertainty_confidence,
        "recommendation_next_step": summary.recommendation_next_step,
        "agent_contributions": summary.agent_contributions
    }
    
    message = {
        "type": "summary",
        "summary_data": summary_dict,
        "timestamp": datetime.now().isoformat()
    }
    
    disconnected = []
    for connection in connections:
        try:
            await connection.send_json(message)
        except Exception as e:
            logger.warning(f"Error sending summary to client: {e}")
            disconnected.append(connection)
    
    # Remove disconnected clients
    for conn in disconnected:
        await remove_connection(conn)


async def send_agent_started(agent_id: str):
    """Send agent_started message to create a new card in the UI.
    
    This should be called before each agent invocation to signal that
    a new reasoning session is beginning. Generates a unique run_id
    for this invocation to track tokens correctly.
    
    Args:
        agent_id: The base agent identifier (e.g., 'OrchestratorAgent')
    """
    connections = await get_active_connections()
    if not connections:
        return
    
    # Generate a unique run_id for this agent invocation
    run_id = str(uuid.uuid4())
    
    # Store the active run_id for this agent
    async with run_ids_lock:
        active_run_ids[agent_id] = run_id
    
    message = {
        "type": "agent_started",
        "agent_id": agent_id,
        "run_id": run_id,
        "timestamp": datetime.now().isoformat()
    }
    
    disconnected = []
    for connection in connections:
        try:
            await connection.send_json(message)
        except Exception as e:
            logger.warning(f"Error sending agent_started to client: {e}")
            disconnected.append(connection)
    
    # Remove disconnected clients
    for conn in disconnected:
        await remove_connection(conn)


def _queue_message_fallback(message: dict, context: str = "message"):
    """Helper function to queue a message as fallback when direct WebSocket send fails.
    
    Args:
        message: The message dict to queue
        context: Description of the message context for logging
    """
    try:
        message_queue.put_nowait(message)
    except asyncio.QueueFull:
        logger.warning(f"Message queue full for {context}, message dropped.")
    except Exception as queue_error:
        logger.error(f"Error queueing {context}: {queue_error}")


def trace_callback(agent_id: str, token: str):
    """Callback function to send tokens directly to WebSocket clients.
    
    This is called from async code (token stream), so we can safely schedule
    async operations. We send tokens directly without queuing to ensure
    token-by-token delivery.
    
    Every token from every agent that uses on_new_token() will
    trigger this callback, ensuring all tokens are captured and displayed.
    """
    # Schedule immediate async send without blocking
    # Since we're called from async context, we can safely create a task
    try:
        # Try to get the running event loop - will succeed if called from async context
        asyncio.get_running_loop()
        # Schedule the send immediately as a task
        asyncio.create_task(send_token_direct(agent_id, token))
    except RuntimeError:
        # No running event loop - unexpected but log warning and fallback to queue
        logger.warning(f"No running event loop in trace_callback for agent {agent_id} - using queue fallback")
        # Look up the active run_id for this agent (synchronous access for fallback)
        run_id = None
        try:
            # Try to get run_id synchronously (may not be perfect but better than nothing)
            run_id = active_run_ids.get(agent_id)
        except Exception:
            # Silently ignore any errors accessing active_run_ids dictionary
            # This can occur during cleanup or if the dictionary is being modified concurrently
            pass
        message = {
            "type": "token",
            "agent_id": agent_id,
            "run_id": run_id,  # Include run_id to match tokens to correct card
            "token": token,
            "timestamp": datetime.now().isoformat()
        }
        _queue_message_fallback(message, f"agent {agent_id} token")
    except Exception as e:
        # Fallback to queue if direct send fails
        logger.warning(f"Error sending token directly, falling back to queue: {e}")
        # Look up the active run_id for this agent (synchronous access for fallback)
        run_id = None
        try:
            # Try to get run_id synchronously (may not be perfect but better than nothing)
            run_id = active_run_ids.get(agent_id)
        except Exception:
            # Silently ignore any errors accessing active_run_ids dictionary
            # This can occur during cleanup or if the dictionary is being modified concurrently
            pass
        message = {
            "type": "token",
            "agent_id": agent_id,
            "run_id": run_id,  # Include run_id to match tokens to correct card
            "token": token,
            "timestamp": datetime.now().isoformat()
        }
        _queue_message_fallback(message, f"agent {agent_id} token")


def _summary_to_dict(summary) -> dict:
    """Convert a summary Pydantic model to a dict for serialization."""
    return {
        "status_action": summary.status_action,
        "key_findings": summary.key_findings,
        "differential_rationale": summary.differential_rationale,
        "uncertainty_confidence": summary.uncertainty_confidence,
        "recommendation_next_step": summary.recommendation_next_step,
        "agent_contributions": summary.agent_contributions
    }


def summary_callback(summary):
    """Callback function to send summaries directly to WebSocket clients.
    
    This is called when EXAIM generates a new summary from buffered traces.
    
    Args:
        summary: AgentSummary Pydantic object
    """
    try:
        # Try to get the running event loop - will succeed if called from async context
        asyncio.get_running_loop()
        # Schedule the send immediately as a task
        asyncio.create_task(send_summary_direct(summary))
    except RuntimeError:
        # No running event loop - unexpected but log warning and fallback to queue
        logger.warning("No running event loop in summary_callback - using queue fallback")
        message = {
            "type": "summary",
            "summary_data": _summary_to_dict(summary),
            "timestamp": datetime.now().isoformat()
        }
        _queue_message_fallback(message, "summary")
    except Exception as e:
        # Fallback to queue if direct send fails
        logger.warning(f"Error sending summary directly, falling back to queue: {e}")
        message = {
            "type": "summary",
            "summary_data": _summary_to_dict(summary),
            "timestamp": datetime.now().isoformat()
        }
        _queue_message_fallback(message, "summary")


async def broadcast_message(message: dict):
    """Broadcast a message to all connected WebSocket clients."""
    connections = await get_active_connections()
    if not connections:
        return
    
    disconnected = []
    for connection in connections:
        try:
            await connection.send_json(message)
        except Exception as e:
            logger.warning(f"Error sending message to client: {e}")
            disconnected.append(connection)
    
    # Remove disconnected clients
    for conn in disconnected:
        await remove_connection(conn)


async def message_broadcaster():
    """Background task to process message queue and broadcast to clients.
    
    Processes messages aggressively to ensure smooth streaming without batching.
    """
    while True:
        try:
            # Wait for at least one message
            message = await message_queue.get()
            await broadcast_message(message)
            message_queue.task_done()
            
            # Process any additional messages immediately without waiting
            # This prevents batching and ensures smooth streaming
            while not message_queue.empty():
                try:
                    message = message_queue.get_nowait()
                    await broadcast_message(message)
                    message_queue.task_done()
                except asyncio.QueueEmpty:
                    break
                except Exception as e:
                    logger.error(f"Error processing queued message: {e}")
                    message_queue.task_done()
                    
        except asyncio.CancelledError:
            break
        except Exception as e:
            logger.error(f"Error in message broadcaster: {e}")


@app.get("/api/traces")
async def list_traces():
    """List available trace files from evals/data/traces/ directory.
    
    Returns list of trace files with case_id extracted from filename.
    """
    # Get project root directory
    current_file = Path(__file__).resolve()
    project_root = current_file.parent.parent.parent  # Go up from server.py -> backend -> demos -> project root
    traces_dir = project_root / "evals" / "data" / "traces"
    
    if not traces_dir.exists():
        logger.warning(f"Traces directory not found: {traces_dir}")
        return {"traces": []}
    
    traces = []
    for trace_file in sorted(traces_dir.glob("*.trace.jsonl.gz")):
        # Extract case_id from filename (e.g., "case-33651373.trace.jsonl.gz" -> "case-33651373")
        case_id = trace_file.stem.replace(".trace.jsonl", "")
        traces.append({
            "case_id": case_id,
            "file_path": str(trace_file.relative_to(project_root))
        })
    
    return {"traces": traces}


async def replay_trace_file(trace_file_path: str):
    """Replay a trace file and stream events to WebSocket clients.
    
    Args:
        trace_file_path: Path to trace file (relative to project root)
    """
    # Get project root directory
    current_file = Path(__file__).resolve()
    project_root = current_file.parent.parent.parent
    full_trace_path = project_root / trace_file_path
    
    if not full_trace_path.exists():
        raise HTTPException(status_code=404, detail=f"Trace file not found: {trace_file_path}")
    
    # Initialize EXAIM for summary generation
    exaim = EXAIM()
    exaim.register_trace_callback(trace_callback)
    exaim.register_summary_callback(summary_callback)
    
    # Clear active run IDs when starting a new trace replay
    async with run_ids_lock:
        active_run_ids.clear()
    
    # Initialize trace replay engine
    try:
        engine = TraceReplayEngine(
            full_trace_path,
            strict_stub_guard=False,  # Allow stub traces for demo
            shift_to_zero=False  # Preserve original timing
        )
    except Exception as e:
        logger.error(f"Error initializing trace replay engine: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to initialize trace replay: {str(e)}")
    
    # Track timing per agent
    last_time_ms = None
    
    try:
        # Replay content_plane stream only (excludes control_plane turns like orchestrator summaries)
        # This ensures EXAIM only processes actual agent reasoning traces, not internal MAS coordination
        for event in engine.replay_content_plane():
            # Handle timing preservation
            if last_time_ms is not None:
                delay_ms = max(0, event.virtual_time_ms - last_time_ms)
                if delay_ms > 0:
                    await asyncio.sleep(delay_ms / 1000.0)
            last_time_ms = event.virtual_time_ms
            
            if event.event_type == "turn_start":
                # Send agent_started message (this generates and stores run_id)
                await send_agent_started(event.agent_id)
                
            elif event.event_type == "delta":
                # Send token directly to WebSocket
                await send_token_direct(event.agent_id, event.delta_text)
                
                # Process token through EXAIM immediately to match live streaming behavior
                # This allows TokenGate to accumulate and flush chunks naturally during the stream
                # Summaries will be generated at the same points as in live mode
                try:
                    # Temporarily remove trace_callback to avoid duplicate token sending
                    # (we've already sent tokens via send_token_direct above)
                    had_callback = trace_callback in exaim.trace_callbacks
                    if had_callback:
                        exaim.trace_callbacks.remove(trace_callback)
                    
                    # Process each character through TokenGate
                    for char in event.delta_text:
                        await exaim.on_new_token(event.agent_id, char)
                    
                    # Re-register trace_callback for future deltas
                    if had_callback and trace_callback not in exaim.trace_callbacks:
                        exaim.register_trace_callback(trace_callback)
                except Exception as e:
                    logger.warning(f"Error processing delta token for agent {event.agent_id}: {e}")
                    # Re-register callback even on error
                    if trace_callback not in exaim.trace_callbacks:
                        exaim.register_trace_callback(trace_callback)
                
            elif event.event_type == "turn_end":
                # Flush any remaining TokenGate content for this agent
                # Tail content is parked for later summarization
                try:
                    # Temporarily remove trace_callback to avoid duplicate token sending
                    had_callback = trace_callback in exaim.trace_callbacks
                    if had_callback:
                        exaim.trace_callbacks.remove(trace_callback)
                    
                    # Flush any remaining buffer
                    await exaim.flush_agent(event.agent_id)
                    
                    # Re-register trace_callback for future turns
                    if had_callback and trace_callback not in exaim.trace_callbacks:
                        exaim.register_trace_callback(trace_callback)
                except Exception as e:
                    logger.warning(f"Error flushing agent {event.agent_id} at turn_end: {e}")
                    # Re-register callback even on error
                    if trace_callback not in exaim.trace_callbacks:
                        exaim.register_trace_callback(trace_callback)
    
    except Exception as e:
        logger.error(f"Error during trace replay: {e}")
        raise HTTPException(status_code=500, detail=f"Trace replay failed: {str(e)}")


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time trace updates.
    
    This endpoint maintains a persistent connection for streaming agent traces
    and summaries to the client. Client messages are received to keep the
    connection alive but are not processed - all communication is server-to-client
    via broadcast_message().
    """
    await websocket.accept()
    await add_connection(websocket)
    connections = await get_active_connections()
    logger.info(f"Client connected. Total connections: {len(connections)}")
    
    try:
        # Keep connection alive - just wait for disconnect
        # Messages are sent via broadcast_message() from other parts of the app
        while True:
            # Wait for any message (or disconnect) - messages are received but not
            # processed as this is a server-push only endpoint for trace streaming
            await websocket.receive_text()
    except WebSocketDisconnect:
        await remove_connection(websocket)
        connections = await get_active_connections()
        logger.info(f"Client disconnected. Total connections: {len(connections)}")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        await remove_connection(websocket)


@app.post("/api/process-case")
async def process_case(request: CaseRequest):
    """Process a clinical case through the CDSS system or replay a trace file.
    
    Routes to either:
    - Live Demo mode: CDSS graph execution (existing functionality)
    - Trace Replay mode: Replay frozen trace file
    
    A new CDSS instance is created for each request. The cdss_lock ensures that only
    one request can process at a time, preventing race conditions with the global
    cdss_instance and its callbacks. Requests will be queued and processed sequentially.
    
    Input validation (empty check, length limit) is handled by Pydantic validator
    and will return 422 Unprocessable Entity for invalid input.
    """
    global cdss_instance, cdss_process_task, should_stop, cancellation_event
    
    # Acquire lock to ensure only one request processes at a time
    async with cdss_lock:
        try:
            # Reset stop flag and cancellation event
            should_stop = False
            cancellation_event.clear()
            
            # Clear active run IDs when starting a new case
            async with run_ids_lock:
                active_run_ids.clear()
            
            # Send start message
            await broadcast_message({
                "type": "processing_started",
                "timestamp": datetime.now().isoformat()
            })
            
            # Route based on mode
            if request.mode == "live_demo":
                # Existing CDSS path
                # Create a new CDSS instance for this request
                cdss_instance = CDSS()
                
                # Register trace callback with EXAIM
                cdss_instance.exaim.register_trace_callback(trace_callback)
                logger.debug(f"Trace callback registered. Callbacks: {len(cdss_instance.exaim.trace_callbacks)}")
                
                # Register summary callback with EXAIM
                cdss_instance.exaim.register_summary_callback(summary_callback)
                logger.debug(f"Summary callback registered. Callbacks: {len(cdss_instance.exaim.summary_callbacks)}")
                
                # Process the case (this will stream traces via callbacks)
                # request.case is already validated and trimmed by Pydantic validator
                # Wrap in a cancellable task
                async def process_with_cancellation():
                    try:
                        return await cdss_instance.process_case(request.case)
                    except asyncio.CancelledError:
                        logger.info("Case processing was cancelled during execution")
                        raise
                    except Exception as e:
                        # Check if cancellation was requested
                        if cancellation_event.is_set() or should_stop:
                            logger.info("Case processing was stopped due to cancellation request")
                            raise asyncio.CancelledError("Processing stopped by user")
                        raise
                
                cdss_process_task = asyncio.create_task(process_with_cancellation())
                
                try:
                    await cdss_process_task
                except asyncio.CancelledError:
                    logger.info("Case processing was cancelled")
                    # Send stop message
                    await broadcast_message({
                        "type": "processing_stopped",
                        "timestamp": datetime.now().isoformat()
                    })
                    return {
                        "status": "stopped",
                        "message": "Case processing was stopped"
                    }
                
                # Check if stop was requested
                if should_stop or cancellation_event.is_set():
                    logger.info("Case processing was stopped")
                    await broadcast_message({
                        "type": "processing_stopped",
                        "timestamp": datetime.now().isoformat()
                    })
                    return {
                        "status": "stopped",
                        "message": "Case processing was stopped"
                    }
                
            elif request.mode == "trace_replay":
                # Check if trace replay is available
                if not TRACE_REPLAY_AVAILABLE:
                    raise HTTPException(
                        status_code=501,
                        detail="Trace replay mode is not available in this deployment"
                    )
                # New trace replay path
                await replay_trace_file(request.trace_file)
            else:
                raise HTTPException(status_code=400, detail=f"Invalid mode: {request.mode}")
            
            # Send completion message
            await broadcast_message({
                "type": "processing_complete",
                "timestamp": datetime.now().isoformat()
            })
            
            return {
                "status": "success",
                "message": "Case processed successfully"
            }
        except HTTPException as http_exc:
            # Send error to WebSocket clients
            await broadcast_message({
                "type": "error",
                "message": http_exc.detail,
                "timestamp": datetime.now().isoformat()
            })
            # Re-raise to return proper HTTP error response
            raise
        except Exception as e:
            error_msg = str(e)
            logger.error(f"Error processing case: {error_msg}")
            logger.exception("Full traceback:")
            
            # Send error message to clients
            await broadcast_message({
                "type": "error",
                "message": error_msg,
                "timestamp": datetime.now().isoformat()
            })
            
            raise HTTPException(status_code=500, detail=error_msg)
        finally:
            cdss_process_task = None
            cancellation_event.clear()


@app.post("/api/stop-case")
async def stop_case():
    """Stop the currently running case processing and restart servers.
    
    This will kill both backend and frontend processes and restart them.
    """
    global cdss_instance, cdss_process_task, should_stop, cancellation_event
    
    # Clear all state before restarting
    should_stop = True
    cancellation_event.set()
    
    # Cancel and clear the current task
    async with cdss_lock:
        if cdss_process_task and not cdss_process_task.done():
            cdss_process_task.cancel()
            try:
                await asyncio.wait_for(cdss_process_task, timeout=0.5)
            except (asyncio.CancelledError, asyncio.TimeoutError):
                pass  # Expected if task was cancelled or timed out during shutdown
            cdss_process_task = None
        
        # Reset CDSS instance
        if cdss_instance:
            try:
                cdss_instance.reset()
            except Exception as e:
                logger.warning(f"Error resetting CDSS instance: {e}")
            cdss_instance = None
    
    # Send stop message to frontend before restarting
    try:
        await broadcast_message({
            "type": "processing_stopped",
            "timestamp": datetime.now().isoformat()
        })
    except Exception as e:
        logger.warning(f"Error sending stop message: {e}")
    
    # Get project root directory (parent of demos/)
    current_file = Path(__file__).resolve()
    project_root = current_file.parent.parent.parent  # Go up from server.py -> backend -> demos -> project root
    restart_script = project_root / "restart_servers.ps1"
    
    if not restart_script.exists():
        logger.error(f"Restart script not found at {restart_script}")
        return {
            "status": "error",
            "message": f"Restart script not found at {restart_script}"
        }
    
    # Execute restart script in background
    # Use subprocess.Popen to run in background without blocking
    try:
        if sys.platform == "win32":
            # Windows: Use PowerShell to run the script
            subprocess.Popen(
                ["powershell", "-ExecutionPolicy", "Bypass", "-File", str(restart_script)],
                cwd=str(project_root),
                creationflags=subprocess.CREATE_NEW_CONSOLE if sys.platform == "win32" else 0,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )
            logger.info("Restart script executed - servers will restart shortly")
        else:
            # Unix/Mac: Use bash
            bash_script = project_root / "restart_servers.sh"
            if bash_script.exists():
                # Make script executable
                os.chmod(str(bash_script), 0o755)
                subprocess.Popen(
                    ["bash", str(bash_script)],
                    cwd=str(project_root),
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL
                )
                logger.info("Restart script executed - servers will restart shortly")
            else:
                logger.error(f"Restart script not found at {bash_script}")
                return {
                    "status": "error",
                    "message": f"Restart script not found at {bash_script}"
                }
        
        # Give a moment for the script to start, then this process will be killed
        await asyncio.sleep(1)
        
        return {
            "status": "success",
            "message": "Servers are being restarted"
        }
    except Exception as e:
        logger.error(f"Error executing restart script: {e}")
        return {
            "status": "error",
            "message": f"Failed to restart servers: {str(e)}"
        }


if __name__ == "__main__":
    import uvicorn
    print("Starting EXAIM API server...")
    print("WebSocket endpoint: ws://localhost:8000/ws")
    print("API endpoint: http://localhost:8000/api/process-case")
    uvicorn.run(app, host="0.0.0.0", port=8000)
