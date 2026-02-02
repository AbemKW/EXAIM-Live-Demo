"""Compatibility shim: exaid_core re-exports from exaim_core."""

# Explicit imports (exaim_core uses lazy imports, so import * doesn't work)
from exaim_core import EXAIM, AgentSummary, TokenGate, BufferAgent, SummarizerAgent

# Alias for backward compatibility
EXAID = EXAIM

__all__ = ['EXAID', 'EXAIM', 'AgentSummary', 'TokenGate', 'BufferAgent', 'SummarizerAgent']
