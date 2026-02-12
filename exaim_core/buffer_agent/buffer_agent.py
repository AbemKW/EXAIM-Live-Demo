from langchain_core.prompts import ChatPromptTemplate
from infra import get_llm, LLMRole
from pydantic import BaseModel, ValidationError
from typing import Union
import logging
from exaim_core.schema.agent_segment import AgentSegment
from exaim_core.schema.buffer_analysis import BufferAnalysis, BufferAnalysisNoNovelty
from exaim_core.utils.prompts import (
    get_buffer_agent_system_prompt,
    get_buffer_agent_system_prompt_no_novelty,
    get_buffer_agent_user_prompt
)
from exaim_core.utils.json_utils import extract_json_from_text, extract_json_with_cot_fallback

logger = logging.getLogger(__name__)


class TraceData(BaseModel):
    count: int


def compute_trigger(a) -> tuple[bool, str]:
    """Compute trigger deterministically from BufferAnalysis-like object.
    
    Args:
        a: BufferAnalysis or BufferAnalysisNoNovelty object
        
    Returns:
        Tuple of (trigger: bool, path: str) where path is one of:
        - "C" for CRITICAL_ALERT
        - "A" for completed value path
        - "B" for topic shift path
        - "-" for no trigger
    """
    # Path C: Critical alert
    if a.stream_state == "CRITICAL_ALERT":
        return True, "C"
    
    structural_closed = getattr(a, "structural_closed", a.is_complete)
    semantic_complete = getattr(a, "semantic_complete", a.is_complete)
    
    # Path A: Completed value
    if structural_closed and semantic_complete:
        if hasattr(a, "is_novel"):
            if a.is_relevant and a.is_novel:
                return True, "A"
        else:
            # BufferAnalysisNoNovelty: no novelty check
            if a.is_relevant:
                return True, "A"
    
    # Path B: Topic shift
    if a.stream_state == "TOPIC_SHIFT":
        if hasattr(a, "is_novel"):
            if a.is_relevant and a.is_novel:
                return True, "B"
        else:
            # BufferAnalysisNoNovelty: no novelty check
            if a.is_relevant:
                return True, "B"
    
    return False, "-"


class BufferAgent:
    def __init__(self, disable_novelty: bool = False):
        self.buffer: list[AgentSegment] = []
        self.tail_segments: list[AgentSegment] = []  # Deferred segments from non-trigger flushes
        # Use a smarter model if available, or the same base model
        self.base_llm = get_llm(LLMRole.BUFFER_AGENT)
        self.disable_novelty = disable_novelty
        # Conditionally initialize LLM and prompt based on novelty check
        if disable_novelty:
            try:
                self.llm = self.base_llm.with_structured_output(BufferAnalysisNoNovelty)
                self.use_json_fallback = False
            except (AttributeError, NotImplementedError):
                # Model doesn't support structured output, use JSON parsing fallback
                self.llm = self.base_llm
                self.use_json_fallback = True
            self.flag_prompt = ChatPromptTemplate.from_messages([
                ("system", get_buffer_agent_system_prompt_no_novelty()),
                ("user", get_buffer_agent_user_prompt())
            ])
            self.schema_class = BufferAnalysisNoNovelty
        else:
            try:
                self.llm = self.base_llm.with_structured_output(BufferAnalysis)
                self.use_json_fallback = False
            except (AttributeError, NotImplementedError):
                # Model doesn't support structured output, use JSON parsing fallback
                self.llm = self.base_llm
                self.use_json_fallback = True
            self.flag_prompt = ChatPromptTemplate.from_messages([
                ("system", get_buffer_agent_system_prompt()),
                ("user", get_buffer_agent_user_prompt())
            ])
            self.schema_class = BufferAnalysis
        self.traces: dict[str, TraceData] = {}
        self.last_analysis: dict[str, Union[BufferAnalysis, BufferAnalysisNoNovelty, None]] = {}



    def _parse_llm_output(self, response) -> Union[BufferAnalysis, BufferAnalysisNoNovelty]:
        """Parse LLM output into schema, handling both structured and text outputs."""
        if self.use_json_fallback:
            # Extract text content with better error handling
            try:
                if hasattr(response, 'content'):
                    content = response.content
                elif isinstance(response, str):
                    content = response
                else:
                    content = str(response)
                
                # Log debug information
                logger.debug(f"Raw LLM response type: {type(response)}")
                logger.debug(f"Content (first 500 chars): {content[:500]}")
                
                # Try to extract JSON (strip CoT traces first)
                json_data = extract_json_with_cot_fallback(content)
                if json_data:
                    try:
                        return self.schema_class(**json_data)
                    except ValidationError as e:
                        raise ValueError(f"JSON validation failed: {e}\nExtracted JSON: {json_data}")
                else:
                    raise ValueError(f"Could not extract valid JSON from response: {content[:500]}")
            except Exception as e:
                logger.debug(f"Exception in _parse_llm_output: {type(e).__name__}: {str(e)}")
                raise ValueError(f"Error parsing LLM output: {type(e).__name__}: {str(e)}")
        else:
            # Already structured output
            return response

    @staticmethod
    def format_segments_for_prompt(segments: list[AgentSegment]) -> str:
        if not segments:
            return "(Buffer empty)"

        lines = []
        last_agent = None
        acc = []

        def flush():
            nonlocal acc, last_agent
            if acc and last_agent is not None:
                # Simple newline-separated format: agent_id on its own line, then content
                lines.append(f"{last_agent}:")
                lines.append(" ".join(acc))
            acc = []

        for s in segments:
            if s.agent_id != last_agent:
                flush()
                last_agent = s.agent_id
            acc.append(s.segment)

        flush()
        return "\n".join(lines)

    async def addsegment(
        self,
        agent_id: str,
        segment: str,
        previous_summaries: list[str],
        flush_reason: str | None = None,
        history_k: int = 3
    ) -> bool:
        new_text = segment
        if agent_id not in self.traces:
            self.traces[agent_id] = TraceData(count=0)
        self.traces[agent_id].count += 1
        
        # Add to buffer and track agent_id
        self.buffer.append(AgentSegment(agent_id=agent_id, segment=new_text))
        
        # Prepare context
        # We pass the *rest* of the buffer separate from the *new* segment so the LLM sees the flow.
        # Include deferred tail content so trigger decisions account for previously parked segments.
        # Group by agent to keep the prompt compact.
        prior_segments = self.tail_segments + self.buffer[:-1]
        buffer_context = self.format_segments_for_prompt(prior_segments)
        new_trace_block = self.format_segments_for_prompt([self.buffer[-1]])
        
        chain = self.flag_prompt | self.llm
        
        try:
            response = await chain.ainvoke({
                "summaries": previous_summaries,
                "previous_trace": buffer_context,
                "new_trace": new_trace_block,
                "flush_reason": flush_reason or "none",
                "history_k": history_k
            })
            analysis: Union[BufferAnalysis, BufferAnalysisNoNovelty] = self._parse_llm_output(response)
            self.last_analysis[agent_id] = analysis
            
            # Compute trigger deterministically in code (not from model)
            trigger, trigger_path = compute_trigger(analysis)
            
            # DEBUG: Print the LLM's analysis to verify it's working
            novelty_str = f" | Novel={analysis.is_novel}" if hasattr(analysis, "is_novel") else ""
            CYAN = "\033[1;36m"  # Bright cyan
            RESET = "\033[0m"    # Reset color
            print(f"{CYAN}DEBUG [{agent_id}]: State={analysis.stream_state} | Complete={analysis.is_complete} | Relevant={analysis.is_relevant}{novelty_str} | Trigger={trigger} (path={trigger_path}){RESET}")
            print(f"{CYAN}DEBUG [{agent_id}] Rationale: {analysis.rationale}{RESET}")
            
            return trigger

        except Exception as e:
            # Fallback if structured output fails (fail closed/safe to avoid spam)
            RED = "\033[1;31m"   # Bright red
            RESET = "\033[0m"    # Reset color
            print(f"{RED}Buffer decision failed: {e}{RESET}")
            self.last_analysis[agent_id] = None
            return False
    
    def flush(self) -> list[AgentSegment]:
        """Flush tail + live buffer and return segments with their corresponding agent IDs.

        Tail segments are deferred content parked by `park_tail()` when a forced
        flush occurs without a BufferAgent trigger. They are included in the
        `buffer_context` passed to `addsegment()` decisions so trigger logic can
        consider them, and they are prepended here so the next summarization
        includes them with their original agent IDs.
        
        Returns:
            List of AgentSegment items, preserving original agent attribution
        """
        flushed_segments = self.tail_segments + self.buffer
        self.tail_segments.clear()
        self.buffer.clear()
        return flushed_segments

    def park_tail(self, segments: list[AgentSegment]) -> None:
        """Append leftover segments to the tail buffer without summarizing."""
        if not segments:
            return
        self.tail_segments.extend(segments)
        
    def get_trace_count(self, agent_id: str) -> int:
        return self.traces.get(agent_id, TraceData(count=0)).count

    def get_last_analysis(self, agent_id: str) -> Union[BufferAnalysis, BufferAnalysisNoNovelty, None]:
        return self.last_analysis.get(agent_id)
