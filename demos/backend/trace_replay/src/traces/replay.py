"""Deterministic trace replay engine with conservative classification."""

from __future__ import annotations

import hashlib
import warnings
from pathlib import Path
from typing import Iterator, Literal, Optional

from .classifier import classify_turn
from .models import (
    AuditFlag,
    ReplayEvent,
    StubTraceError,
    TraceMeta,
    TraceValidationError,
    TurnClassification,
    _TurnData,
)
from .parser import derive_agent_labels, iter_trace_records


# Boundary timestamp tolerance (milliseconds)
# Due to millisecond resolution and async timing, boundary timestamps may be
# off by 1-2ms from delta timestamps. This is cosmetic and does not affect
# evaluation semantics.
BOUNDARY_TIME_EPSILON_MS = 2


def compute_time_shift(records: list[dict]) -> int:
    min_t_rel = 0
    for record in records:
        t_rel = record.get("t_rel_ms")
        if t_rel is not None:
            min_t_rel = min(min_t_rel, t_rel)
    return min_t_rel


class TraceReplayEngine:
    def __init__(
        self,
        trace_path: Path,
        *,
        strict_stub_guard: bool = True,
        strict_validation: bool = True,
        shift_to_zero: bool = False,
    ):
        self._trace_path = Path(trace_path)
        self._strict_stub_guard = strict_stub_guard
        self._strict_validation = strict_validation
        self._shift_to_zero = shift_to_zero

        self._records: Optional[list[dict]] = None
        self._metadata: Optional[TraceMeta] = None
        self._derived_labels: Optional[frozenset[str]] = None
        self._classifications: Optional[dict[int, TurnClassification]] = None
        self._audit_flags: Optional[list[AuditFlag]] = None
        self._turn_data: Optional[dict[int, _TurnData]] = None
        self._time_shift: int = 0
        self._initialized: bool = False

    def _ensure_initialized(self) -> None:
        if self._initialized:
            return
        self._derived_labels = derive_agent_labels(self._trace_path)
        if not self._derived_labels:
            warnings.warn(f"No agent labels derived from {self._trace_path}.")
        self._records = list(iter_trace_records(self._trace_path))
        if not self._records:
            raise TraceValidationError(f"Trace file is empty: {self._trace_path}")
        self._parse_metadata()
        if self._strict_stub_guard and self._metadata.stub_mode:
            raise StubTraceError("Stub trace not allowed")
        self._validate_trace()
        if self._shift_to_zero:
            self._time_shift = compute_time_shift(self._records)
        self._reconstruct_turns()
        self._classify_all_turns()
        self._initialized = True

    def _parse_metadata(self) -> None:
        first_record = self._records[0]
        if first_record.get("record_type") != "trace_meta":
            raise TraceValidationError("First record must be trace_meta")
        self._metadata = TraceMeta(
            schema_version=first_record.get("schema_version", "unknown"),
            case_id=first_record.get("case_id", "unknown"),
            mas_run_id=first_record.get("mas_run_id", "unknown"),
            mac_commit=first_record.get("mac_commit", "unknown"),
            model=first_record.get("model", "unknown"),
            created_at=first_record.get("created_at", "unknown"),
            t0_emitted_ms=first_record.get("t0_emitted_ms", 0),
            stub_mode=first_record.get("stub_mode", False),
            total_turns=first_record.get("total_turns"),
            total_deltas=first_record.get("total_deltas"),
            decoding=first_record.get("decoding"),
        )

    # (Truncated: the rest of the implementation mirrors the original replay.py)

    def replay_content_plane(self) -> Iterator[ReplayEvent]:
        return self._iter_events("content_plane")

    def replay_full(self) -> Iterator[ReplayEvent]:
        return self._iter_events("full")

    def get_metadata(self) -> TraceMeta:
        self._ensure_initialized()
        return self._metadata

    def get_derived_agent_labels(self) -> frozenset[str]:
        self._ensure_initialized()
        return self._derived_labels

    def get_turn_classifications(self) -> dict[int, TurnClassification]:
        self._ensure_initialized()
        return self._classifications.copy()

    def get_audit_flags(self) -> list[AuditFlag]:
        self._ensure_initialized()
        return self._audit_flags.copy()


def replay_trace(trace_path: Path, stream: Literal["full", "content_plane"] = "full", **kwargs):
    engine = TraceReplayEngine(trace_path, **kwargs)
    if stream == "full":
        yield from engine.replay_full()
    else:
        yield from engine.replay_content_plane()
