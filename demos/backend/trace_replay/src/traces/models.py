"""Shared data models and exceptions for trace replay."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Optional


class TraceReplayError(Exception):
    """Base exception for trace replay errors."""


class TraceValidationError(TraceReplayError):
    """Raised when trace validation fails."""


class StubTraceError(TraceReplayError):
    """Raised when attempting to replay a stub trace without override."""


@dataclass(frozen=True)
class ReplayEvent:
    """
    Single replay event yielded by the engine.

    Represents either a delta (stream content) or a boundary (turn start/end).
    """

    event_type: Literal["delta", "turn_start", "turn_end"]
    virtual_time_ms: int
    stream_mode: Literal["full", "content_plane"]
    seq: int
    turn_id: int
    agent_id: str

    # Delta-specific (None for boundaries)
    delta_text: Optional[str] = None

    # Boundary-specific (None for deltas)
    boundary: Optional[Literal["start", "end"]] = None
    content_hash: Optional[str] = None  # Only on turn_end


@dataclass
class TurnClassification:
    """
    Classification result for a single turn.

    Every turn is classified as either content_plane or control_plane,
    with an auditable classification_reason.
    """

    turn_id: int
    turn_type: Literal["content_plane", "control_plane"]
    agent_id: str
    turn_text: str
    classification_reason: str
    delta_count: int
    start_seq: int
    end_seq: int


@dataclass
class AuditFlag:
    """
    Flags for reviewer visibility on suspicious but unfiltered turns.

    When a turn looks label-like but doesn't match derived labels,
    it's kept as content_plane but flagged for audit.
    """

    turn_id: int
    flag_type: str
    details: str


@dataclass
class TraceMeta:
    """Parsed trace metadata from trace_meta record."""

    schema_version: str
    case_id: str
    mas_run_id: str
    mac_commit: str
    model: str
    created_at: str
    t0_emitted_ms: int
    stub_mode: bool
    total_turns: Optional[int]
    total_deltas: Optional[int]
    decoding: Optional[dict] = None


@dataclass
class _TurnData:
    """Internal: Accumulated data for a turn during reconstruction."""

    turn_id: int
    agent_id: str
    deltas: list = field(default_factory=list)  # List of (seq, delta_text, t_rel_ms)
    start_seq: Optional[int] = None
    end_seq: Optional[int] = None
    start_t_rel_ms: Optional[int] = None
    end_t_rel_ms: Optional[int] = None
    content_hash: Optional[str] = None
