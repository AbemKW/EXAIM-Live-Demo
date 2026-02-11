from langchain_core.prompts import ChatPromptTemplate
from exaim_core.schema.agent_summary import AgentSummary
from typing import List, Dict, Any
from pydantic import ValidationError
import json
import re
import logging
from infra import get_llm, LLMRole
from exaim_core.utils.prompts import (
    get_summarizer_system_prompt,
    get_summarizer_user_prompt,
)
from exaim_core.schema.agent_segment import AgentSegment
from exaim_core.utils.json_utils import extract_json_from_text, extract_json_with_cot_fallback

class SummarizerAgent:
    def __init__(self):
        # Get LLM with role-specific generation parameters configured in registry
        self.base_llm = get_llm(LLMRole.SUMMARIZER)
        # Prepare the JSON schema for guided decoding (vLLM guided_json)
        try:
            self.guided_json_schema = AgentSummary.model_json_schema()
        except Exception:
            logger = logging.getLogger(__name__)
            logger.warning(
                "Failed to generate guided_json schema from AgentSummary.model_json_schema(); "
                "guided JSON will be disabled.",
                exc_info=True,
            )
            self.guided_json_schema = None
        try:
            self.llm = self.base_llm.with_structured_output(
                    schema=AgentSummary,
                    method="json_schema",
                    strict=True
            )
            self.use_json_fallback = False
        except (AttributeError, NotImplementedError):
            # Model doesn't support structured output, use JSON parsing fallback
            self.llm = self.base_llm
            self.use_json_fallback = True

        self.summarize_prompt = ChatPromptTemplate.from_messages([    
            ("system", get_summarizer_system_prompt()),
            ("user", get_summarizer_user_prompt()),
        ])
        
        # Field limits for validation (word counts, match schema guidance)
        # Values represent maximum allowed words per field
        self.field_limits = {
            'status_action': 25,
            'key_findings': 30,
            'differential_rationale': 35,
            'uncertainty_confidence': 20,
            'recommendation_next_step': 30,
            'agent_contributions': 25
        }
    


    def _validate_and_truncate(self, summary: AgentSummary) -> AgentSummary:
        """Proactively validate and truncate summary fields to ensure limits.
        
        Since 4b models cannot reliably count words, we ALWAYS validate
        and truncate as needed before returning any summary.
        Preserves full text in full_* fields when truncation occurs.
        
        Args:
            summary: AgentSummary that may exceed limits
            
        Returns:
            AgentSummary with fields guaranteed to be within limits
        """
        logger = logging.getLogger(__name__)
        output_dict = summary.model_dump()
        
        def _word_count(s: str) -> int:
            return len(str(s).split()) if s is not None else 0

        for field, max_words in self.field_limits.items():
            value = str(getattr(summary, field, ''))
            wc = _word_count(value)
            if wc > max_words:
                logger.warning(f"Field '{field}' exceeds word limit: {wc} > {max_words}. Truncating and preserving full text.")
                # Preserve full text in full_* field
                output_dict[f"full_{field}"] = value
                # Truncate main field for display
                output_dict[field] = self._truncate_field(value, max_words)
            else:
                # No truncation needed
                output_dict[f"full_{field}"] = None
        
        return AgentSummary(**output_dict)

    def _parse_llm_output(self, response) -> AgentSummary:
        """Parse LLM output into AgentSummary, handling both structured and text outputs."""
        if self.use_json_fallback:
            # Extract text content with better error handling
            try:
                if hasattr(response, 'content'):
                    content = response.content
                elif isinstance(response, str):
                    content = response
                else:
                    content = str(response)
                
                # Use robust extractor that strips CoT traces and code fences
                json_data = extract_json_with_cot_fallback(content)
                if json_data:
                    try:
                        return AgentSummary(**json_data)
                    except ValidationError:
                        # Re-raise to allow higher-level retry logic
                        raise
                else:
                    logger = logging.getLogger(__name__)
                    logger.error(f"Could not extract JSON from response (first 1000 chars): {content[:1000]}")
                    raise ValueError(f"Could not extract valid JSON from response: {content[:500]}")
            except ValidationError:
                # Re-raise ValidationError as-is so retry logic can handle it
                raise
            except Exception as e:
                logger = logging.getLogger(__name__)
                logger.error(f"Error parsing LLM output: {type(e).__name__}: {str(e)}")
                raise ValueError(f"Error parsing LLM output: {type(e).__name__}: {str(e)}")
        else:
            # Already structured output
            return response

    def _extract_max_length_violations(self, validation_error: ValidationError) -> Dict[str, int]:
        """Extract fields that violated length/word constraints from ValidationError.

        Returns:
            Dict mapping field names to their max_word limits
        """
        violations = {}
        for error in validation_error.errors():
            field_path = error.get('loc', ())
            if not field_path:
                continue
            field_name = field_path[-1]
            ctx = error.get('ctx', {}) or {}
            # Prefer explicit max_words in ctx if present
            max_words = ctx.get('max_words')
            if max_words:
                violations[field_name] = max_words
                continue

            # Fallback: parse message for a max_words token (validator may include it)
            msg = error.get('msg', '')
            m = re.search(r'max_words\s*=\s*(\d+)', msg)
            if m:
                violations[field_name] = int(m.group(1))
                continue

            # Last resort: use configured field_limits if present
            if field_name in self.field_limits:
                violations[field_name] = self.field_limits[field_name]

        return violations
    
    def _truncate_field(self, text: str, max_words: int) -> str:
        """Truncate a field to a maximum number of words.

        Args:
            text: Text to truncate
            max_words: Maximum allowed words

        Returns:
            Truncated text (first max_words words)
        """
        words = str(text).split()
        if len(words) <= max_words:
            return text
        return " ".join(words[:max_words])
    
    def _apply_fallback_truncation(self, output_dict: Dict[str, Any]) -> AgentSummary:
        """Apply fallback truncation to fields that exceed word limits.

        This is a last resort when the LLM fails to comply.
        Preserves full text in full_* fields when truncation occurs.

        Args:
            output_dict: Dictionary with field values that may exceed limits

        Returns:
            AgentSummary with truncated fields
        """
        truncated_dict = {}
        for field, max_words in self.field_limits.items():
            value = str(output_dict.get(field, ''))
            if len(value.split()) > max_words:
                # Preserve full text
                truncated_dict[f"full_{field}"] = value
                # Truncate for display
                truncated_dict[field] = self._truncate_field(value, max_words)
            else:
                truncated_dict[field] = value
                truncated_dict[f"full_{field}"] = None

        return AgentSummary(**truncated_dict)
    
    # Local truncation is used for length violations to avoid repeated retries
    # and reduce latency when small models exceed field limits.

    def _extract_validation_error_from_exception(self, e: Exception) -> tuple[ValidationError | None, dict | None]:
        """Extract ValidationError and parsed JSON from LangChain exception.
        
        LangChain wraps ValidationError in OutputParserException. This method:
        1. Checks if exception is ValidationError directly
        2. Checks if ValidationError is in __cause__ or args
        3. Tries to extract JSON from error message string
        
        Returns:
            Tuple of (ValidationError or None, parsed JSON dict or None)
        """
        # Check if it's a ValidationError directly
        if isinstance(e, ValidationError):
            return e, None
        
        # Check __cause__ (common pattern for wrapped exceptions)
        if hasattr(e, '__cause__') and isinstance(e.__cause__, ValidationError):
            validation_error = e.__cause__
            # Try to extract JSON from error message
            error_str = str(e)
            json_dict = self._extract_json_from_error_message(error_str)
            return validation_error, json_dict
        
        # Check args for ValidationError
        for arg in getattr(e, 'args', []):
            if isinstance(arg, ValidationError):
                error_str = str(e)
                json_dict = self._extract_json_from_error_message(error_str)
                return arg, json_dict
        
        # Try to extract JSON from error message and create ValidationError
        error_str = str(e)
        json_dict = self._extract_json_from_error_message(error_str)
        if json_dict:
            try:
                # Try to create the object to get the real ValidationError
                AgentSummary(**json_dict)
            except ValidationError as ve:
                return ve, json_dict
        
        return None, json_dict if json_dict else None
    
    def _extract_json_from_error_message(self, error_str: str) -> dict | None:
        """Extract JSON dict from error message string.
        
        LangChain error messages often include the JSON that failed validation.
        """
        import re
        # Look for JSON object in the error message
        # Pattern: {"key": "value", ...}
        json_match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', error_str)
        if json_match:
            try:
                return json.loads(json_match.group())
            except json.JSONDecodeError:
                pass
        return None

    async def _get_raw_output(
        self,
        summary_history: List[str],
        latest_summary: str,
        new_buffer: str,
        history_k: int = 3,
    ) -> Dict[str, Any]:
        """Get raw LLM output as a dictionary, extracting JSON if needed.

        Returns:
            Dictionary with field values, or None if extraction fails
        """
        try:
            raw_chain = self.summarize_prompt | self.base_llm
            # Ask vLLM to constrain output to the AgentSummary JSON schema when available
            extra_body = {"guided_json": self.guided_json_schema} if self.guided_json_schema is not None else None
            
            # Limit the history to prevent token limit issues
            limited_history = summary_history[-history_k:] if len(summary_history) > history_k else summary_history
            
            if extra_body is not None:
                raw_response = await raw_chain.ainvoke({
                "summary_history": ",\n".join(limited_history),
                "latest_summary": latest_summary,
                "new_buffer": new_buffer,
                "history_k": history_k
                }, extra_body=extra_body)
            else:
                raw_response = await raw_chain.ainvoke({
                    "summary_history": ",\n".join(limited_history),
                    "latest_summary": latest_summary,
                    "new_buffer": new_buffer,
                    "history_k": history_k
                })

            # Extract JSON from raw response using robust extractor
            content = raw_response.content if hasattr(raw_response, 'content') else str(raw_response)
            logger = logging.getLogger(__name__)
            logger.info(f"Summarizer Raw Content Preview: {content[:1000]}")

            json_data = extract_json_with_cot_fallback(content)
            return json_data
        except Exception:
            pass

        return None
    
    @staticmethod
    def format_segments_for_prompt(segments: List[AgentSegment]) -> str:
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

    async def summarize(
        self,
        segments_with_agents: List[AgentSegment],
        summary_history: List[str],
        latest_summary: str,
        history_k: int = 3,
    ) -> AgentSummary:
        """Summarize agent output with automatic retry and fallback truncation.

        This method attempts to get a valid summary and enforces field limits:
        1. Initial attempt with structured output
        2. Fallback truncation on length violations

        Args:
            segments_with_agents: List of AgentSegment items representing agent contributions
            summary_history: List of previous summary strings
            latest_summary: Most recent summary string
            history_k: The number of previous summaries to include in history

        Returns:
            AgentSummary object

        Raises:
            ValidationError: If validation fails for non-length-related reasons
            Exception: For other unexpected errors
        """
        summarize_chain = self.summarize_prompt | self.llm

        new_buffer = self.format_segments_for_prompt(segments_with_agents)

        # Limit the history to prevent token limit issues
        limited_history = summary_history[-history_k:] if len(summary_history) > history_k else summary_history

        # Attempt 1: Initial structured output
        try:
            extra_body = {"guided_json": self.guided_json_schema} if self.guided_json_schema is not None else None
            if extra_body is not None:
                response = await summarize_chain.ainvoke({
                "summary_history": ",\n".join(limited_history),
                "latest_summary": latest_summary,
                "new_buffer": new_buffer,
                "history_k": history_k
                }, extra_body=extra_body)
            else:
                response = await summarize_chain.ainvoke({
                    "summary_history": ",\n".join(limited_history),
                    "latest_summary": latest_summary,
                    "new_buffer": new_buffer,
                    "history_k": history_k
                })
            summary = self._parse_llm_output(response)

            # CRITICAL: Always validate and truncate before returning
            return self._validate_and_truncate(summary)

        except Exception as e:
            # Extract ValidationError from LangChain exception wrapper
            validation_error, previous_output = self._extract_validation_error_from_exception(e)

            if validation_error:
                # Check if this is a max_length violation
                violations = self._extract_max_length_violations(validation_error)

                if violations:
                    if previous_output is None:
                        previous_output = await self._get_raw_output(
                            summary_history,
                            latest_summary,
                            new_buffer,
                            history_k,
                        )

                    if previous_output is None:
                        # Fall back to an empty dict with expected fields so truncation
                        # produces a valid AgentSummary with empty/trimmed values.
                        previous_output = {f: '' for f in self.field_limits.keys()}

                    logger = logging.getLogger(__name__)
                    logger.warning(
                        "Summarizer agent: applying local truncation to fields: %s",
                        list(violations.keys()),
                    )
                    return self._apply_fallback_truncation(previous_output)
                else:
                    # Not a max_length violation, re-raise
                    raise validation_error
            else:
                # Not a ValidationError, re-raise the original exception
                raise
