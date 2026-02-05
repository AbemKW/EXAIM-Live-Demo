from langchain_core.prompts import ChatPromptTemplate
from exaim_core.schema.agent_summary import AgentSummary
from typing import List, Dict, Any
from pydantic import ValidationError
import json
import logging
from infra import get_llm, LLMRole
from exaim_core.utils.prompts import get_summarizer_system_prompt, get_summarizer_user_prompt
from exaim_core.schema.agent_segment import AgentSegment
from exaim_core.utils.json_utils import extract_json_from_text

class SummarizerAgent:
    def __init__(self):
        self.base_llm = get_llm(LLMRole.SUMMARIZER)
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
        
        # Field limits for rewrite prompts
        self.field_limits = {
            'status_action': 150,
            'key_findings': 180,
            'differential_rationale': 210,
            'uncertainty_confidence': 120,
            'recommendation_next_step': 180,
            'agent_contributions': 150
        }
    


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
                
                # Try to extract JSON
                json_data = extract_json_from_text(content)
                if json_data:
                    try:
                        return AgentSummary(**json_data)
                    except ValidationError as e:
                        raise ValueError(f"JSON validation failed: {e}\nExtracted JSON: {json_data}")
                else:
                    raise ValueError(f"Could not extract valid JSON from response: {content[:500]}")
            except Exception as e:
                logger = logging.getLogger(__name__)
                logger.error(f"Error parsing LLM output: {type(e).__name__}: {str(e)}")
                raise ValueError(f"Error parsing LLM output: {type(e).__name__}: {str(e)}")
        else:
            # Already structured output
            return response

    def _extract_max_length_violations(self, validation_error: ValidationError) -> Dict[str, int]:
        """Extract fields that violated max_length constraints from ValidationError.
        
        Returns:
            Dict mapping field names to their max_length limits
        """
        violations = {}
        for error in validation_error.errors():
            if error['type'] == 'string_too_long':
                field_path = error.get('loc', ())
                if field_path:
                    field_name = field_path[-1]
                    # Extract max_length from ctx if available
                    ctx = error.get('ctx', {})
                    max_length = ctx.get('max_length')
                    if max_length:
                        violations[field_name] = max_length
                    elif field_name in self.field_limits:
                        violations[field_name] = self.field_limits[field_name]
        return violations
    
    def _truncate_field(self, text: str, max_length: int) -> str:
        """Truncate a field to max_length, preserving word boundaries when possible.
        
        Args:
            text: Text to truncate
            max_length: Maximum allowed length
            
        Returns:
            Truncated text
        """
        if len(text) <= max_length:
            return text
        
        # Try to truncate at a word boundary
        truncated = text[:max_length]
        # Find the last space before the limit
        last_space = truncated.rfind(' ')
        if last_space > max_length * 0.8:  # Only use word boundary if it's not too far back
            truncated = truncated[:last_space]
        else:
            truncated = truncated[:max_length]
        
        return truncated
    
    def _apply_fallback_truncation(self, output_dict: Dict[str, Any]) -> AgentSummary:
        """Apply fallback truncation to fields that exceed limits.
        
        This is a last resort when the LLM fails to comply after retries.
        
        Args:
            output_dict: Dictionary with field values that may exceed limits
            
        Returns:
            AgentSummary with truncated fields
        """
        truncated_dict = {}
        for field, max_len in self.field_limits.items():
            value = output_dict.get(field, '')
            if len(str(value)) > max_len:
                truncated_dict[field] = self._truncate_field(str(value), max_len)
            else:
                truncated_dict[field] = value
        
        return AgentSummary(**truncated_dict)
    
    def _create_rewrite_prompt(self, previous_output: Dict[str, Any], violations: Dict[str, int]) -> str:
        """Create a targeted rewrite prompt for fields that exceeded max_length.
        
        Args:
            previous_output: The previous output that failed validation (as dict)
            violations: Dict mapping field names to their max_length limits
            
        Returns:
            Rewrite prompt string
        """
        violation_list = []
        for field, max_len in violations.items():
            current_value = previous_output.get(field, '')
            current_len = len(str(current_value))
            violation_list.append(
                f"- {field}: currently {current_len} characters, must be ≤ {max_len} characters (need to remove {current_len - max_len} characters)"
            )
        
        violations_text = '\n'.join(violation_list)
        
        prompt = f"""⚠️ CRITICAL REWRITE REQUEST ⚠️

Your previous output was REJECTED because it exceeded character limits. You MUST shorten the following fields:

{violations_text}

Your previous output:
{json.dumps(previous_output, indent=2)}

MANDATORY INSTRUCTIONS:
1. Shorten ONLY the fields listed above to meet their character limits EXACTLY
2. Count characters as you shorten - verify each field is ≤ its limit
3. Preserve all semantic meaning and clinical information
4. Use abbreviations, remove redundant words, prioritize essential facts
5. Keep all other fields exactly as they are
6. Return the complete output with shortened fields

VERIFY BEFORE SUBMITTING: Count characters in each shortened field to ensure compliance."""
        
        return prompt

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
            raw_response = await raw_chain.ainvoke({
                "summary_history": ",\n".join(summary_history),
                "latest_summary": latest_summary,
                "new_buffer": new_buffer,
                "history_k": history_k
            })
            
            # Extract JSON from raw response
            content = raw_response.content if hasattr(raw_response, 'content') else str(raw_response)
            
            logger = logging.getLogger(__name__)
            logger.info(f"Summarizer Raw Content Preview: {content[:1000]}") # Debug log
            
            # Try to parse JSON from response
            start_idx = content.find('{')
            if start_idx != -1:
                # Find matching closing brace
                brace_count = 0
                in_string = False
                escape_next = False
                
                for i in range(start_idx, len(content)):
                    char = content[i]
                    
                    if escape_next:
                        escape_next = False
                        continue
                    
                    if char == '\\':
                        escape_next = True
                        continue
                    
                    if char == '"' and not escape_next:
                        in_string = not in_string
                        continue
                    
                    if not in_string:
                        if char == '{':
                            brace_count += 1
                        elif char == '}':
                            brace_count -= 1
                            if brace_count == 0:
                                json_str = content[start_idx:i+1]
                                return json.loads(json_str)
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
        
        This method attempts to get a valid summary up to 3 times:
        1. Initial attempt with structured output
        2. Retry with rewrite prompt if character limits exceeded
        3. Fallback truncation if retry still fails
        
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
        
        # Attempt 1: Initial structured output
        try:
            response = await summarize_chain.ainvoke({
                "summary_history": ",\n".join(summary_history),
                "latest_summary": latest_summary,
                "new_buffer": new_buffer,
                "history_k": history_k
            })
            summary = self._parse_llm_output(response)
            return summary
        except Exception as e:
            # Extract ValidationError from LangChain exception wrapper
            validation_error, previous_output = self._extract_validation_error_from_exception(e)
            
            if validation_error:
                # Check if this is a max_length violation
                violations = self._extract_max_length_violations(validation_error)
                
                if violations:
                    # Attempt 2: Retry with rewrite prompt
                    try:
                        # Get raw output if not already extracted
                        if previous_output is None:
                            previous_output = await self._get_raw_output(
                                summary_history,
                                latest_summary,
                                new_buffer,
                                history_k,
                            )
                        
                        if previous_output is None:
                            # Can't extract output, use fallback truncation on a minimal dict
                            # This shouldn't happen, but we'll handle it gracefully
                            raise ValueError("Could not extract previous output for rewrite")
                        
                        # Create rewrite prompt
                        rewrite_prompt_text = self._create_rewrite_prompt(previous_output, violations)
                        
                        # Create rewrite prompt template
                        rewrite_prompt = ChatPromptTemplate.from_messages([
                            ("system", get_summarizer_system_prompt()),
                            ("user", rewrite_prompt_text),
                        ])
                        
                        # Retry with rewrite prompt
                        rewrite_chain = rewrite_prompt | self.llm
                        response = await rewrite_chain.ainvoke({})
                        summary = self._parse_llm_output(response)
                        return summary
                    
                    except Exception as retry_error:
                        # Attempt 3: Fallback truncation
                        # Get the raw output if we don't have it
                        if previous_output is None:
                            previous_output = await self._get_raw_output(
                                summary_history,
                                latest_summary,
                                new_buffer,
                                history_k,
                            )
                        
                        if previous_output is None:
                            # Last resort: re-raise the original validation error
                            raise validation_error
                        
                        # Apply fallback truncation
                        logger = logging.getLogger(__name__)
                        logger.warning(
                            f"Summarizer agent failed to comply with character limits after retry. "
                            f"Applying fallback truncation to fields: {list(violations.keys())}"
                        )
                        return self._apply_fallback_truncation(previous_output)
                else:
                    # Not a max_length violation, re-raise
                    raise validation_error
            else:
                # Not a ValidationError, re-raise the original exception
                raise
