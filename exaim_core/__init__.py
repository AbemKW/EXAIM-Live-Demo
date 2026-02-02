"""EXAIM Core - Explainable AI Middleware Framework

This module contains the core EXAIM research artifact including:
- TokenGate: Token streaming pre-buffer with syntax awareness
- BufferAgent: Intelligent trace buffer with LLM-based trigger
- SummarizerAgent: LLM-based summarization with structured output
- EXAIM: Main orchestrator class
- AgentSummary: Structured summary schema

"""

__all__ = ['EXAIM', 'AgentSummary', 'TokenGate', 'BufferAgent', 'SummarizerAgent']

# Lazy imports using PEP 562 __getattr__
# This allows importing exaim_core without pulling in heavy dependencies
# like LangChain unless the specific classes are accessed


def __getattr__(name: str):
    """Lazy import for module-level attributes (PEP 562)."""
    if name == 'TokenGate':
        from exaim_core.token_gate.token_gate import TokenGate
        return TokenGate
    elif name == 'AgentSummary':
        from exaim_core.schema.agent_summary import AgentSummary
        return AgentSummary
    elif name == 'EXAIM':
        from exaim_core.exaim import EXAIM
        return EXAIM
    elif name == 'BufferAgent':
        from exaim_core.buffer_agent.buffer_agent import BufferAgent
        return BufferAgent
    elif name == 'SummarizerAgent':
        from exaim_core.summarizer_agent.summarizer_agent import SummarizerAgent
        return SummarizerAgent
    else:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

