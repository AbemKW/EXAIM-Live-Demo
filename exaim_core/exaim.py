from typing import Optional, Callable, List
import asyncio
import uuid
import logging
from exaim_core.buffer_agent.buffer_agent import BufferAgent
from exaim_core.schema.agent_segment import AgentSegment
from exaim_core.summarizer_agent.summarizer_agent import SummarizerAgent
from exaim_core.token_gate.token_gate import TokenGate
from exaim_core.schema.agent_summary import AgentSummary

class EXAIM:
    def __init__(self, history_k: int = 2):
        self.buffer_agent = BufferAgent()
        self.summarizer_agent = SummarizerAgent(max_new_buffer_words=500)
        self.token_gate = TokenGate()
        self.summaries: list[AgentSummary] = []
        self.history_k = history_k
        self.trace_callbacks: List[Callable[[str, str], None]] = []
        self.summary_callbacks: List[Callable[[AgentSummary], None]] = []
        # Task reference tracking to prevent GC
        self.background_tasks: set = set()
        # Lock for thread-safe summary list access
        self.summaries_lock = asyncio.Lock()
    
    def register_trace_callback(self, callback: Callable[[str, str], None]):
        """Register a callback function to be called when trace tokens are received.
        
        Args:
            callback: Function that takes (agent_id: str, token: str) as arguments
        """
        self.trace_callbacks.append(callback)
    
    def register_summary_callback(self, callback: Callable[[AgentSummary], None]):
        """Register a callback function to be called when summaries are created.
        
        Args:
            callback: Function that takes (summary: AgentSummary) as argument
        """
        self.summary_callbacks.append(callback)
    
    def _print_summary(self, summary: AgentSummary):
        if summary is None:
            print(f"\n{'='*60}")
            print(f"Summary Update")
            print(f"{'='*60}")
            print(f"Warning: Received None summary")
            print()
            return
        
        print(f"\n{'='*60}")
        print(f"Summary Update")
        print(f"{'='*60}")
        print(f"Status / Action: {summary.status_action}")
        print(f"Key Findings: {summary.key_findings}")
        print(f"Differential & Rationale: {summary.differential_rationale}")
        print(f"Uncertainty / Confidence: {summary.uncertainty_confidence}")
        print(f"Recommendation / Next Step: {summary.recommendation_next_step}")
        print(f"Agent Contributions: {summary.agent_contributions}")
        print()
    
    async def get_all_summaries(self) -> list[AgentSummary]:
        """Returns all summaries as AgentSummary objects."""
        async with self.summaries_lock:
            return self.summaries.copy()

    async def get_summaries_by_agent(self, agent_id: str) -> list[AgentSummary]:
        """Get all summaries involving a specific agent."""
        async with self.summaries_lock:
            return [s for s in self.summaries if agent_id.lower() in s.agent_contributions.lower()]

    def get_agent_trace_count(self, agent_id: str) -> int:
        return self.buffer_agent.get_trace_count(agent_id)

    def _format_summary_for_history(self, summary: AgentSummary) -> str:
        """Converts an AgentSummary to a string representation for use in history."""
        parts = [
            f"Status/Action: {summary.status_action}",
            f"Key Findings: {summary.key_findings}",
            f"Differential/Rationale: {summary.differential_rationale}",
            f"Uncertainty/Confidence: {summary.uncertainty_confidence}",
            f"Recommendation/Next: {summary.recommendation_next_step}",
            f"Agent Contributions: {summary.agent_contributions}"
        ]
        return " | ".join(parts)
    
    def _format_summaries_history(self, summaries: list[AgentSummary]) -> list[str]:
        """Converts a list of AgentSummary objects to string representations for prompt history."""
        return [self._format_summary_for_history(s) for s in summaries]

    async def _run_background_summary(
        self,
        agent_segments: list[AgentSegment],
        summary_history_strs: list[str],
        latest_summary_str: str,
        trigger_id: str,
    ):
        """Run summarization in background without blocking the main stream.
        
        This is a fire-and-forget task that:
        1. Calls the summarizer agent
        2. Appends the result to summaries
        3. Invokes summary callbacks
        
        Args:
            agent_segments: Segments to summarize
            summary_history_strs: Previous summaries for context
            latest_summary_str: Most recent summary
            trigger_id: Unique ID for this trigger (for debugging/correlation)
        """
        logger = logging.getLogger(__name__)
        
        try:

            logger.info(f"[EXAIM Background] Starting background summarization task (trigger_id={trigger_id})")
            summary = await asyncio.wait_for(
                self.summarizer_agent.summarize(
                    agent_segments,
                    summary_history_strs,
                    latest_summary_str,
                    self.history_k,
                ),
                timeout=180.0,
            )
            
            if summary is not None:
                # Thread-safe append with lock
                async with self.summaries_lock:
                    self.summaries.append(summary)
                
                self._print_summary(summary)
                
                # Emit summary event to callbacks
                for callback in self.summary_callbacks:
                    try:
                        callback(summary)
                    except Exception as e:
                        logger.error(f"Error in summary callback: {e}")
                        

                logger.info(f"[EXAIM Background] Summary completed successfully (trigger_id={trigger_id})")

                logger.warning(f"[EXAIM Background] Summarizer returned None (trigger_id={trigger_id})")
                
        except asyncio.TimeoutError:
            logger.error(f"[EXAIM Background] Summarizer timed out after 120s (trigger_id={trigger_id})")
        except Exception as e:
            logger.error(f"[EXAIM Background] Summarizer failed (trigger_id={trigger_id}): {e}", exc_info=True)
            # Re-add segments to buffer_agent's tail so they are not lost
            self.buffer_agent.park_tail(agent_segments)
            logger.info(f"[EXAIM Background] Re-parked {len(agent_segments)} segments after summarizer failure (trigger_id={trigger_id})")

    def _get_limited_history(self, summaries: list[AgentSummary]) -> list[str]:
        """Get the last k summary entries."""
        limited = summaries[-self.history_k:] if self.history_k > 0 else []
        return self._format_summaries_history(limited)

    async def received_trace(self, agent_id: str, text: str) -> Optional[AgentSummary]:
        """
        Processes a trace for the given agent ID and text, triggering summarization if appropriate.

        Returns:
            AgentSummary: if summarization was triggered.
            None: otherwise.
        """
        # Emit trace text as tokens to callbacks (for non-streaming traces)
        for callback in self.trace_callbacks:
            try:
                # Send the entire text as a single "token" for non-streaming traces
                callback(agent_id, text)
            except Exception as e:
                logger.error(f"Error in trace callback: {e}")
        
        # Prepare previous summaries for buffer agent evaluation
        all_summaries = await self.get_all_summaries()
        previous_summaries = self._get_limited_history(all_summaries)
        
        trigger = await self.buffer_agent.addsegment(
            agent_id,
            text,
            previous_summaries,
            flush_reason="full_trace",
            history_k=self.history_k
        )
        if trigger:
            # Generate unique trigger ID for correlation tracking
            trigger_id = str(uuid.uuid4())
            
            agent_segments = self.buffer_agent.flush()
            all_summaries = await self.get_all_summaries()
            summary_history_strs = self._get_limited_history(all_summaries[:-1])
            latest_summary_str = self._format_summary_for_history(all_summaries[-1]) if all_summaries else "No summaries yet."
            
            logger.info(f"[EXAIM] Trigger {trigger_id}: Flushed {len(agent_segments)} segments from received_trace")
            
            # Create task with reference to prevent GC
            task = asyncio.create_task(self._run_background_summary(
                agent_segments,
                summary_history_strs,
                latest_summary_str,
                trigger_id
            ))
            # Store reference and auto-cleanup on completion
            self.background_tasks.add(task)
            task.add_done_callback(self.background_tasks.discard)
            
            # Return immediately without waiting for summary
            return None
        return None

    async def on_new_token(self, agent_id: str, token: str) -> Optional[AgentSummary]:
        """Process a single token for the given agent."""
        # Emit trace callbacks
        for callback in self.trace_callbacks:
            try:
                callback(agent_id, token)
            except Exception as e:
                logger.error(f"Error in trace callback: {e}")

        # Pass token through TokenGate
        chunk = await self.token_gate.add_token(agent_id, token)

        # Process chunk if complete
        if chunk:
            summaries = await self.get_all_summaries()
            flush_reason = self.token_gate.get_last_flush_reason(agent_id)
            print(f"\033[1;32m[EXAIM DEBUG] Calling _process_chunk for {agent_id}, flush_reason={flush_reason}\033[0m")
            return await self._process_chunk(agent_id, chunk, summaries, flush_reason)

        # Check TokenGate timers
        timer_chunk = await self.token_gate.check_timers(agent_id)
        if timer_chunk:
            summaries = await self.get_all_summaries()
            flush_reason = self.token_gate.get_last_flush_reason(agent_id)
            print(f"\033[1;32m[EXAIM DEBUG] Calling _process_chunk (timer) for {agent_id}, flush_reason={flush_reason}\033[0m")
            return await self._process_chunk(agent_id, timer_chunk, summaries, flush_reason)

        return None

    async def _process_chunk(
        self,
        agent_id: str,
        chunk: str,
        summaries: list[AgentSummary],
        flush_reason: str | None = None
    ) -> Optional[AgentSummary]:
        """Process a chunk of text for summarization."""
        logger = logging.getLogger(__name__)
        

        logger.info(f"[EXAIM] Processing chunk for {agent_id}, flush_reason={flush_reason}, chunk_length={len(chunk)}")
        
        previous_summaries = self._get_limited_history(summaries)
        
        try:
            trigger = await self.buffer_agent.addsegment(
                agent_id,
                chunk,
                previous_summaries,
                flush_reason=flush_reason,
                history_k=self.history_k
            )

            logger.info(f"[EXAIM] Buffer agent returned trigger={trigger} for {agent_id}")
        except Exception as e:
            logger.error(f"[EXAIM] Buffer agent failed for {agent_id}: {e}", exc_info=True)
            return None
            
        if trigger:
            # Generate unique trigger ID for correlation tracking
            trigger_id = str(uuid.uuid4())
            
            # Flush returns deferred tail + current buffer content.
            agent_segments = self.buffer_agent.flush()

            logger.info(f"[EXAIM] Trigger {trigger_id}: Flushed {len(agent_segments)} segments, starting background summarizer...")
            
            summary_history_strs = self._get_limited_history(summaries[:-1])
            latest_summary_str = self._format_summary_for_history(summaries[-1]) if summaries else "No summaries yet."
            
            # Create task with reference to prevent GC
            task = asyncio.create_task(self._run_background_summary(
                agent_segments,
                summary_history_strs,
                latest_summary_str,
                trigger_id
            ))
            # Store reference and auto-cleanup on completion
            self.background_tasks.add(task)
            task.add_done_callback(self.background_tasks.discard)
            

            logger.info(f"[EXAIM] Background summarization task started (trigger_id={trigger_id}), returning immediately")
            # Return immediately without waiting for summary
            return None
        return None

    async def flush_agent(self, agent_id: str) -> None:
        """Flush remaining tokens and park them without triggering summarization."""
        remaining = await self.token_gate.flush(agent_id)
        if not remaining:
            return None

        logger = logging.getLogger(__name__)
        self.buffer_agent.park_tail([AgentSegment(agent_id=agent_id, segment=remaining)])
        logger.debug(
            "Parked 1 segment for %s in tail buffer",
            agent_id
        )
        return None

