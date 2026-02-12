from typing import Optional, Dict, Protocol
from datetime import datetime, timezone, timedelta
import re


class Clock(Protocol):
    """Protocol for clock/time provider interface.
    
    Used for deterministic timing in trace replay calibration.
    """
    def now(self) -> datetime:
        """Return current time (UTC-aware datetime)."""
        ...


class ManualClock:
    """Manual clock for deterministic testing/calibration.
    
    Allows setting and advancing time manually for reproducible
    trace replay scenarios.
    
    Example:
        clock = ManualClock(datetime(2024, 1, 1, tzinfo=timezone.utc))
        clock.advance(15.0)  # Advance by 15 seconds
        now = clock.now()
    """
    def __init__(self, initial_time: Optional[datetime] = None):
        """Initialize manual clock.
        
        Args:
            initial_time: Starting time (UTC-aware). If None, uses current UTC time.
        """
        # CRITICAL: Keep UTC-aware for both real clock and ManualClock
        self._time = initial_time or datetime.now(timezone.utc)
        if self._time.tzinfo is None:
            # Ensure UTC-aware if provided without timezone
            self._time = self._time.replace(tzinfo=timezone.utc)
    
    def now(self) -> datetime:
        """Return current time (UTC-aware datetime)."""
        return self._time
    
    def set_time(self, dt: datetime):
        """Set current time (must be UTC-aware).
        
        Args:
            dt: New time (will be made UTC-aware if not already)
        """
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        self._time = dt
    
    def advance(self, seconds: float):
        """Advance time by seconds.
        
        Args:
            seconds: Number of seconds to advance
        """
        self._time += timedelta(seconds=seconds)


class TokenGate:
    """A lightweight, syntax-aware pre-buffer that regulates token flow into BufferAgent.
    
    The Token Gate does not interpret meaning - it only decides when enough structure
    has accumulated to pass tokens upstream to the BufferAgent for semantic evaluation.
    
    Note: This class counts whitespace-delimited words (not model tokenizer tokens)
    for its min/max thresholds. The class name "TokenGate" refers to its role in
    gating streaming tokens from the LLM, not to the counting method.
    
    Clock Injection:
        TokenGate supports clock injection for deterministic timing in trace replay
        calibration. Pass a `clock` parameter (implementing the `Clock` protocol) to
        use virtual time instead of real wall-clock time. See `ManualClock` for a
        simple implementation usable in evals.
    
    Boundary Cue Strategy:
        Under Strategy 1, boundary cues are only checked after min_words is reached,
        using a "near-end" detection rule. This rule handles sentence-ending
        punctuation (. ? !) followed by optional closers (quotes, brackets, parentheses),
        making it more robust than checking only the last character. Flushing occurs
        at min_words ONLY if a boundary cue is detected near the end; otherwise the
        buffer waits for max_words or timer triggers.
        
        Boundary cues are hardcoded to `.?!\n` (period, question mark, exclamation,
        newline). Tabs are excluded as formatting noise.
        
        Examples that trigger flush:
        - "Hello world." → True (ends with period)
        - "He said 'Yes.'" → True (period + closing quote)
        - "Hello world.)" → True (period + closing paren)
        - "Hello world\n" → True (ends with newline)
        
        Examples that do NOT trigger:
        - "Hello. More text" → False (punctuation in middle)
        - "Hello world" → False (no punctuation)
        - "Hello world\t" → False (tab is formatting, not a boundary)
    """
    
    def __init__(
        self,
        min_words: int = 60,
        max_words: int = 100,
        silence_timer: float = 1.0,
        max_wait_timeout: float = 4.0,
        clock: Optional[Clock] = None
    ):
        """Initialize TokenGate with configurable flush triggers.
        
        Args:
            min_words: Minimum word threshold (whitespace-delimited) before flushing (default: 60)
            max_words: Maximum word cap (whitespace-delimited) to force flush (default: 100)
            silence_timer: Seconds of inactivity before flush (default: 1.0)
            max_wait_timeout: Maximum seconds before forced flush (default: 4.0)
            clock: Optional clock/time provider for deterministic timing. If None, uses real
                wall-clock time. Use `ManualClock` for trace replay calibration.
        
        Note: Balanced settings for quality summaries with reasonable responsiveness.
        
        Boundary cues are hardcoded to `.?!\n` (period, question mark, exclamation, newline)
        and cannot be configured.
        """
        self.min_words = min_words
        self.max_words = max_words
        self.silence_timer = silence_timer
        self.max_wait_timeout = max_wait_timeout
        self.clock = clock
        
        # Per-agent text buffers
        self.buffers: Dict[str, str] = {}
        
        # Track when each agent's buffer started (for max wait timeout)
        self.buffer_start_times: Dict[str, datetime] = {}
        
        # Track when last token was received (for silence timer)
        self.last_token_times: Dict[str, datetime] = {}
        
        # Track flush reasons and times for calibration
        self.last_flush_reason: Dict[str, str] = {}
        self.last_flush_time: Dict[str, datetime] = {}
    
    def _get_now(self) -> datetime:
        """Get current time using injected clock or real time.
        
        Returns:
            Current UTC-aware datetime
        """
        if self.clock is not None:
            return self.clock.now()
        return datetime.now(timezone.utc)
    
    def get_last_flush_reason(self, agent_id: str) -> Optional[str]:
        """Get the reason for the last flush for the given agent.
        
        Useful for calibration to track flush trigger types.
        
        Args:
            agent_id: Agent identifier
            
        Returns:
            Flush reason string (e.g., "silence_timer", "max_words", "boundary_cue")
            or None if no flush has occurred yet
        """
        return self.last_flush_reason.get(agent_id)
    
    def get_last_flush_time(self, agent_id: str) -> Optional[datetime]:
        """Get the timestamp of the last flush for the given agent.
        
        Args:
            agent_id: Agent identifier
            
        Returns:
            Flush timestamp (UTC-aware datetime) or None if no flush has occurred yet
        """
        return self.last_flush_time.get(agent_id)
    
    def _count_words(self, text: str) -> int:
        """Count whitespace-delimited words in text.
        
        This counts words by splitting on whitespace, providing a simple and fast
        method for flow control without external tokenizer dependencies.
        
        Args:
            text: Text to count words in
            
        Returns:
            Number of whitespace-delimited words
        """
        if not text:
            return 0
        # Split by whitespace and count non-empty words
        words = text.split()
        return len(words)
    
    def _has_boundary_cue_at_end(self, text: str) -> bool:
        """Check if buffer ends with boundary cue near the end (after min_words threshold).
        
        Uses a "near-end" detection rule that handles sentence-ending punctuation
        followed by optional closers (quotes, brackets, parentheses). This is more
        robust than checking only the last character, as it correctly identifies
        boundaries like "Hello world.)" or "He said 'Yes.'"
        
        Why "near-end": Sentence-final punctuation may be followed by closing
        quotes, brackets, or parentheses. We check for punctuation followed by
        zero or more closers at the end of the stripped text.
        
        CRITICAL: Include \n as cue, exclude \t (formatting noise).
        Tabs are excluded because they are formatting characters that don't
        indicate semantic boundaries - they can appear anywhere in text.
        
        Args:
            text: Buffer text to check
            
        Returns:
            True if buffer ends with boundary cue near end, False otherwise
        """
        if not text:
            return False
        
        # Strip trailing spaces/tabs only (NOT newlines)
        # This preserves newlines while removing formatting whitespace
        text2 = text.rstrip(" \t")
        if not text2:
            return False
        
        # Check for newline first (common case)
        if text2.endswith("\n") or text2.endswith("\r\n"):
            return True
        
        # Use regex to match sentence-ending punctuation followed by optional closers
        # Pattern: punctuation [.?!] at end, then zero or more closers: ) ] } ' "
        # This handles cases like: "Hello world.)" or "He said 'Yes.'"
        # Note: ] must be escaped as \] in character class
        boundary_pattern = r"[.?!][)\]\}'\"]*$"
        return bool(re.search(boundary_pattern, text2))
    
    def _should_flush(self, agent_id: str) -> Optional[str]:
        """Check if any flush condition is met for the given agent.
        
        Under Strategy 1:
        - Flush on max_words (always)
        - Flush on min_words ONLY if boundary cue present at end
        - Otherwise wait for max_words or timers
        
        Args:
            agent_id: Agent identifier
            
        Returns:
            Flush reason string if should flush, None otherwise
        """
        if agent_id not in self.buffers:
            return None
        
        buffer_text = self.buffers[agent_id]
        word_count = self._count_words(buffer_text)
        
        # max_words always forces flush
        if word_count >= self.max_words:
            return "max_words"
        
        # min_words reached - ONLY flush if boundary cue present
        if word_count >= self.min_words:
            if self._has_boundary_cue_at_end(buffer_text):
                return "boundary_cue"
            # Don't flush - wait for max_words or timers
            return None
        
        return None
    
    def _check_timer_conditions(self, agent_id: str) -> Optional[str]:
        """Check if timer-based flush conditions are met.
        
        Args:
            agent_id: Agent identifier
            
        Returns:
            Flush reason string ("silence_timer" or "max_wait_timeout") if timer expired,
            None otherwise
        """
        if agent_id not in self.buffers or not self.buffers[agent_id]:
            return None
        
        now = self._get_now()
        
        # Check silence timer - if no token received for silence_timer seconds
        if agent_id in self.last_token_times:
            silence_elapsed = (now - self.last_token_times[agent_id]).total_seconds()
            if silence_elapsed >= self.silence_timer:
                return "silence_timer"
        
        # Check max wait timeout - if buffer has existed for max_wait_timeout seconds
        if agent_id in self.buffer_start_times:
            max_wait_elapsed = (now - self.buffer_start_times[agent_id]).total_seconds()
            if max_wait_elapsed >= self.max_wait_timeout:
                return "max_wait_timeout"
        
        return None
    
    async def add_token(self, agent_id: str, token: str) -> Optional[str]:
        """Add a token or chunk to the buffer for the given agent.
        
        The parameter name "token" is historical - this method accepts any text string,
        whether it's a single token, a multi-token chunk, or delta text from trace replay.
        TokenGate is agnostic to input granularity and accumulates text regardless.
        
        If flush conditions are met, returns the buffered text and clears the buffer.
        Otherwise, returns None.
        
        Args:
            agent_id: Agent identifier
            token: Text string to add (may be a single token, chunk, or delta_text)
            
        Returns:
            Flushed chunk text if flush triggered, None otherwise
        """
        now = self._get_now()
        
        # CRITICAL: Check silence timer BEFORE adding token (preserves gap)
        # This handles the case where there's a pause in the stream
        silence_flush = None
        if agent_id in self.buffers and agent_id in self.last_token_times:
            silence_elapsed = (now - self.last_token_times[agent_id]).total_seconds()
            if silence_elapsed >= self.silence_timer and self.buffers[agent_id]:
                # Silence timer expired - flush old buffer
                silence_flush = await self.flush(agent_id, reason="silence_timer")
        
        # Initialize buffer if needed (after potential flush)
        if agent_id not in self.buffers:
            self.buffers[agent_id] = ""
            self.buffer_start_times[agent_id] = now
        
        # Add token to buffer
        self.buffers[agent_id] += token
        
        # Update last token time (silence timer resets on each token)
        self.last_token_times[agent_id] = now
        
        # If silence timer triggered, return the flushed chunk
        # The new token is now in a fresh buffer
        if silence_flush:
            return silence_flush
        
        # Check max wait timeout
        if agent_id in self.buffer_start_times:
            max_wait_elapsed = (now - self.buffer_start_times[agent_id]).total_seconds()
            if max_wait_elapsed >= self.max_wait_timeout:
                return await self.flush(agent_id, reason="max_wait_timeout")
        
        # Check structural flush conditions
        flush_reason = self._should_flush(agent_id)
        if flush_reason:
            return await self.flush(agent_id, reason=flush_reason)
        
        return None
    
    async def flush(self, agent_id: str, reason: Optional[str] = None) -> Optional[str]:
        """Force flush the buffer for the given agent.
        
        Args:
            agent_id: Agent identifier
            reason: Optional reason for the flush (e.g., "silence_timer", "max_words", "boundary_cue")
            
        Returns:
            Flushed buffer text, or None if buffer is empty
        """
        if agent_id not in self.buffers or not self.buffers[agent_id]:
            return None
        
        # Get buffered text
        flushed_text = self.buffers[agent_id]
        
        # Track flush reason and time for calibration
        if reason:
            self.last_flush_reason[agent_id] = reason
        self.last_flush_time[agent_id] = self._get_now()
        
        # Clear buffer
        self.buffers[agent_id] = ""
        
        # Reset timers
        if agent_id in self.buffer_start_times:
            del self.buffer_start_times[agent_id]
        if agent_id in self.last_token_times:
            del self.last_token_times[agent_id]
        
        return flushed_text
    
    async def check_timers(self, agent_id: str) -> Optional[str]:
        """Check if timers have expired and flush if needed.
        
        This should be called periodically or after async operations to check
        if silence timer or max wait timeout has expired.
        
        Args:
            agent_id: Agent identifier
            
        Returns:
            Flushed chunk if timer expired, None otherwise
        """
        reason = self._check_timer_conditions(agent_id)
        if reason:
            return await self.flush(agent_id, reason=reason)
        return None
