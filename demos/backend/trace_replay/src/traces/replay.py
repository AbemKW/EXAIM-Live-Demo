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
    """
    Find minimum t_rel_ms across ALL records (deltas AND boundaries).

    Important: turn_boundary.t_rel_ms may be negative!

    Args:
        records: List of all trace records

    Returns:
        Minimum t_rel_ms value (will be negative if any boundary is before t0)
    """
    min_t_rel = 0  # Default if no records with t_rel_ms

    for record in records:
        t_rel = record.get("t_rel_ms")
        if t_rel is not None:
            min_t_rel = min(min_t_rel, t_rel)

    return min_t_rel


class TraceReplayEngine:
    """
    Deterministic trace replay with conservative classification.

    Two-pass architecture:
        Pass 1: Derive agent_labels from turn_boundary records only (authoritative source)
        Pass 2: Classify turns and yield replay events
    """

    def __init__(
        self,
        trace_path: Path,
        *,
        strict_stub_guard: bool = True,
        strict_validation: bool = True,
        shift_to_zero: bool = False,
    ):
        """
        Initialize the replay engine.

        Args:
            trace_path: Path to trace JSONL(.gz) file
            strict_stub_guard: If True, raise error on stub_mode traces
            strict_validation: If True, raise error on validation failures
            shift_to_zero: If True, shift timeline so minimum virtual_time = 0
        """
        self._trace_path = Path(trace_path)
        self._strict_stub_guard = strict_stub_guard
        self._strict_validation = strict_validation
        self._shift_to_zero = shift_to_zero

        # Lazy-loaded state
        self._records: Optional[list[dict]] = None
        self._metadata: Optional[TraceMeta] = None
        self._derived_labels: Optional[frozenset[str]] = None
        self._classifications: Optional[dict[int, TurnClassification]] = None
        self._audit_flags: Optional[list[AuditFlag]] = None
        self._turn_data: Optional[dict[int, _TurnData]] = None
        self._time_shift: int = 0
        self._initialized: bool = False

    def _ensure_initialized(self) -> None:
        """Load and process trace if not already done."""
        if self._initialized:
            return

        # Pass 1: Derive agent labels from boundaries only
        self._derived_labels = derive_agent_labels(self._trace_path)

        if not self._derived_labels:
            warnings.warn(
                f"No agent labels derived from {self._trace_path}. "
                "This may indicate a data quality issue."
            )

        # Load all records for Pass 2
        self._records = list(iter_trace_records(self._trace_path))

        if not self._records:
            raise TraceValidationError(f"Trace file is empty: {self._trace_path}")

        # Parse metadata
        self._parse_metadata()

        # Check stub guard
        if self._strict_stub_guard and self._metadata.stub_mode:
            raise StubTraceError(
                f"Trace {self._trace_path} is a stub trace (stub_mode=true). "
                "Stub traces must not be used for evaluation. "
                "Pass strict_stub_guard=False to override."
            )

        # Validate trace
        self._validate_trace()

        # Compute time shift if needed
        if self._shift_to_zero:
            self._time_shift = compute_time_shift(self._records)

        # Reconstruct turns and classify
        self._reconstruct_turns()
        self._classify_all_turns()

        self._initialized = True

    def _parse_metadata(self) -> None:
        """Parse trace_meta record."""
        first_record = self._records[0]

        if first_record.get("record_type") != "trace_meta":
            raise TraceValidationError(
                f"First record must be trace_meta, got: {first_record.get('record_type')}"
            )

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

    def _validate_trace(self) -> None:
        """
        Validate trace integrity.

        Checks:
            - seq strictly increasing
            - t_emitted_ms non-decreasing for deltas
            - Boundary start/end pairs match
            - Per-record t_rel_ms consistency (critical for virtual time)
                * stream_delta: t_rel_ms == t_emitted_ms - t0
                * turn_boundary: t_rel_ms == t_ms - t0
            - Per-turn boundary-time containment
                * turn_start.t_ms <= first_delta.t_emitted_ms + epsilon
                * turn_end.t_ms >= last_delta.t_emitted_ms - epsilon
                * Skips turns with boundaries but zero deltas
            - Turns with deltas but no boundaries (schema violation)
            - content_hash matches (warning only)

        Behavior:
            - strict_validation=True: Violations → errors → raises TraceValidationError
            - strict_validation=False: Violations → warnings → continues execution
        """
        errors = []
        warnings_list = []

        t0 = self._metadata.t0_emitted_ms
        prev_seq = -1
        prev_t_emitted = -1
        open_turns: dict[int, dict] = {}  # turn_id -> start boundary
        turn_deltas: dict[int, list[str]] = {}  # turn_id -> list of delta_text
        turn_delta_times: dict[int, list[int]] = {}  # turn_id -> list of t_emitted_ms
        turn_boundaries: dict[int, dict] = {}  # turn_id -> {start: record, end: record}
        turns_with_deltas: set[int] = set()  # Track which turns have deltas
        turns_with_boundaries: set[int] = set()  # Track which turns have boundaries

        for idx, record in enumerate(self._records):
            if idx == 0 and record.get("record_type") == "trace_meta":
                continue

            record_type = record.get("record_type")
            seq = record.get("seq", -1)

            # Check seq strictly increasing
            if seq <= prev_seq:
                errors.append(f"seq {seq} not > prev {prev_seq} at index {idx}")
            prev_seq = seq

            if record_type == "stream_delta":
                t_emitted = record.get("t_emitted_ms", 0)
                t_rel = record.get("t_rel_ms")
                turn_id = record.get("turn_id", 0)
                delta_text = record.get("delta_text", "")

                # Check t_emitted non-decreasing
                if t_emitted < prev_t_emitted:
                    errors.append(
                        f"t_emitted {t_emitted} < prev {prev_t_emitted} at index {idx}"
                    )
                prev_t_emitted = t_emitted

                # Check t_rel_ms consistency for deltas
                if t_rel is not None:
                    expected_t_rel = t_emitted - t0
                    if t_rel != expected_t_rel:
                        msg = (
                            f"stream_delta at index {idx}: t_rel_ms {t_rel} != expected {expected_t_rel} "
                            f"(t_emitted_ms {t_emitted} - t0 {t0})"
                        )
                        if self._strict_validation:
                            errors.append(msg)
                        else:
                            warnings_list.append(msg)

                # Accumulate deltas for content_hash verification
                if turn_id not in turn_deltas:
                    turn_deltas[turn_id] = []
                    turn_delta_times[turn_id] = []
                turn_deltas[turn_id].append(delta_text)
                turn_delta_times[turn_id].append(t_emitted)
                turns_with_deltas.add(turn_id)

            elif record_type == "turn_boundary":
                turn_id = record.get("turn_id", 0)
                agent_id = record.get("agent_id", "unknown")
                boundary = record.get("boundary")
                t_ms = record.get("t_ms", 0)
                t_rel = record.get("t_rel_ms")

                # Check t_rel_ms consistency for boundaries
                if t_rel is not None:
                    expected_t_rel = t_ms - t0
                    if t_rel != expected_t_rel:
                        msg = (
                            f"turn_boundary at index {idx} (turn {turn_id}, {boundary}): "
                            f"t_rel_ms {t_rel} != expected {expected_t_rel} "
                            f"(t_ms {t_ms} - t0 {t0})"
                        )
                        if self._strict_validation:
                            errors.append(msg)
                        else:
                            warnings_list.append(msg)

                if boundary == "start":
                    open_turns[turn_id] = record
                    if turn_id not in turn_boundaries:
                        turn_boundaries[turn_id] = {}
                    turn_boundaries[turn_id]["start"] = record
                    turns_with_boundaries.add(turn_id)
                elif boundary == "end":
                    if turn_id not in open_turns:
                        errors.append(f"turn {turn_id} end without start at index {idx}")
                    else:
                        del open_turns[turn_id]

                    if turn_id not in turn_boundaries:
                        turn_boundaries[turn_id] = {}
                    turn_boundaries[turn_id]["end"] = record

                    # Verify content_hash
                    content_hash = record.get("content_hash")
                    if content_hash and turn_id in turn_deltas:
                        deltas = turn_deltas[turn_id]
                        content = "".join(deltas)
                        expected_hash = (
                            f"sha256:{hashlib.sha256(content.encode('utf-8')).hexdigest()}"
                        )
                        if content_hash != expected_hash:
                            warnings_list.append(
                                f"content_hash mismatch for turn {turn_id}"
                            )

        # Check for unclosed turns
        for turn_id in open_turns:
            errors.append(f"turn {turn_id} has start but no end")

        # Check for turns with deltas but no boundaries (schema violation)
        turns_with_deltas_no_boundaries = turns_with_deltas - turns_with_boundaries
        if turns_with_deltas_no_boundaries:
            msg = (
                "Turns with deltas but no boundaries (schema violation): "
                f"{sorted(turns_with_deltas_no_boundaries)}"
            )
            if self._strict_validation:
                errors.append(msg)
            else:
                warnings_list.append(msg)

        # Per-turn boundary-time containment checks
        for turn_id, boundaries in turn_boundaries.items():
            if turn_id not in turn_delta_times or len(turn_delta_times[turn_id]) == 0:
                # Turn has boundaries but zero deltas - skip boundary-time check
                continue

            first_delta_t = min(turn_delta_times[turn_id])
            last_delta_t = max(turn_delta_times[turn_id])

            start_boundary = boundaries.get("start")
            end_boundary = boundaries.get("end")

            if start_boundary:
                start_t_ms = start_boundary.get("t_ms", 0)
                if start_t_ms > first_delta_t:
                    diff_ms = start_t_ms - first_delta_t
                    msg = (
                        f"turn {turn_id}: turn_start.t_ms {start_t_ms} > first_delta.t_emitted_ms {first_delta_t} "
                        f"(diff={diff_ms}ms, exceeds epsilon={BOUNDARY_TIME_EPSILON_MS}ms)"
                    )
                    if diff_ms > BOUNDARY_TIME_EPSILON_MS:
                        if self._strict_validation:
                            errors.append(msg)
                        else:
                            warnings_list.append(msg)
                    else:
                        warnings_list.append(
                            f"turn {turn_id}: turn_start.t_ms {start_t_ms} > first_delta.t_emitted_ms {first_delta_t} "
                            f"(diff={diff_ms}ms, within epsilon={BOUNDARY_TIME_EPSILON_MS}ms)"
                        )

            if end_boundary:
                end_t_ms = end_boundary.get("t_ms", 0)
                if end_t_ms < last_delta_t:
                    diff_ms = last_delta_t - end_t_ms
                    msg = (
                        f"turn {turn_id}: turn_end.t_ms {end_t_ms} < last_delta.t_emitted_ms {last_delta_t} "
                        f"(diff={diff_ms}ms, exceeds epsilon={BOUNDARY_TIME_EPSILON_MS}ms)"
                    )
                    if diff_ms > BOUNDARY_TIME_EPSILON_MS:
                        if self._strict_validation:
                            errors.append(msg)
                        else:
                            warnings_list.append(msg)
                    else:
                        warnings_list.append(
                            f"turn {turn_id}: turn_end.t_ms {end_t_ms} < last_delta.t_emitted_ms {last_delta_t} "
                            f"(diff={diff_ms}ms, within epsilon={BOUNDARY_TIME_EPSILON_MS}ms)"
                        )

        # Report warnings
        for warning in warnings_list:
            warnings.warn(warning)

        # Handle errors
        if errors and self._strict_validation:
            raise TraceValidationError(
                f"Trace validation failed with {len(errors)} error(s):\n"
                + "\n".join(f"  - {e}" for e in errors[:10])
            )

    def _reconstruct_turns(self) -> None:
        """Reconstruct turn data by accumulating deltas between boundaries."""
        self._turn_data = {}

        for record in self._records:
            record_type = record.get("record_type")

            if record_type == "turn_boundary":
                turn_id = record.get("turn_id", 0)
                agent_id = record.get("agent_id", "unknown")
                boundary = record.get("boundary")
                seq = record.get("seq", 0)
                t_rel_ms = record.get("t_rel_ms", 0)

                if turn_id not in self._turn_data:
                    self._turn_data[turn_id] = _TurnData(
                        turn_id=turn_id,
                        agent_id=agent_id,
                    )

                if boundary == "start":
                    self._turn_data[turn_id].start_seq = seq
                    self._turn_data[turn_id].start_t_rel_ms = t_rel_ms
                elif boundary == "end":
                    self._turn_data[turn_id].end_seq = seq
                    self._turn_data[turn_id].end_t_rel_ms = t_rel_ms
                    self._turn_data[turn_id].content_hash = record.get("content_hash")

            elif record_type == "stream_delta":
                turn_id = record.get("turn_id", 0)
                seq = record.get("seq", 0)
                delta_text = record.get("delta_text", "")
                t_rel_ms = record.get("t_rel_ms", 0)

                if turn_id not in self._turn_data:
                    # Delta without boundary start - create turn data
                    self._turn_data[turn_id] = _TurnData(
                        turn_id=turn_id,
                        agent_id=record.get("agent_id", "unknown"),
                    )

                self._turn_data[turn_id].deltas.append((seq, delta_text, t_rel_ms))

    def _classify_all_turns(self) -> None:
        """Classify all turns using conservative rules."""
        self._classifications = {}
        self._audit_flags = []

        for turn_id, turn_data in self._turn_data.items():
            # Reconstruct turn text from deltas (sorted by seq)
            sorted_deltas = sorted(turn_data.deltas, key=lambda x: x[0])
            turn_text = "".join(d[1] for d in sorted_deltas)

            classification = classify_turn(
                turn_id=turn_id,
                turn_text=turn_text,
                delta_count=len(sorted_deltas),
                agent_id=turn_data.agent_id,
                start_seq=turn_data.start_seq
                or (sorted_deltas[0][0] if sorted_deltas else 0),
                end_seq=turn_data.end_seq
                or (sorted_deltas[-1][0] if sorted_deltas else 0),
                derived_agent_labels=self._derived_labels,
                audit_flags=self._audit_flags,
            )

            self._classifications[turn_id] = classification

    def _compute_virtual_time(self, t_rel_ms: int) -> int:
        """Compute virtual time from t_rel_ms, applying shift if configured."""
        return t_rel_ms - self._time_shift

    def _iter_events(self, stream_mode: Literal["full", "content_plane"]) -> Iterator[ReplayEvent]:
        """
        Internal iterator that yields replay events.

        Args:
            stream_mode: "full" for all events, "content_plane" for content only
        """
        self._ensure_initialized()

        # Get set of control_plane turn IDs for filtering
        control_plane_turns = {
            turn_id
            for turn_id, cls in self._classifications.items()
            if cls.turn_type == "control_plane"
        }

        for record in self._records:
            record_type = record.get("record_type")

            if record_type == "trace_meta":
                continue

            turn_id = record.get("turn_id", 0)

            # Skip control_plane turns in content_plane mode
            if stream_mode == "content_plane" and turn_id in control_plane_turns:
                continue

            if record_type == "stream_delta":
                yield ReplayEvent(
                    event_type="delta",
                    virtual_time_ms=self._compute_virtual_time(record.get("t_rel_ms", 0)),
                    stream_mode=stream_mode,
                    seq=record.get("seq", 0),
                    turn_id=turn_id,
                    agent_id=record.get("agent_id", "unknown"),
                    delta_text=record.get("delta_text", ""),
                )

            elif record_type == "turn_boundary":
                boundary = record.get("boundary")
                yield ReplayEvent(
                    event_type="turn_start" if boundary == "start" else "turn_end",
                    virtual_time_ms=self._compute_virtual_time(record.get("t_rel_ms", 0)),
                    stream_mode=stream_mode,
                    seq=record.get("seq", 0),
                    turn_id=turn_id,
                    agent_id=record.get("agent_id", "unknown"),
                    boundary=boundary,
                    content_hash=record.get("content_hash") if boundary == "end" else None,
                )

    # =========================================================================
    # Public API: Streams
    # =========================================================================

    def replay_full(self) -> Iterator[ReplayEvent]:
        """
        Yield ALL events (deltas + boundaries) in seq order.

        Includes both content_plane and control_plane turns.
        """
        return self._iter_events("full")

    def replay_content_plane(self) -> Iterator[ReplayEvent]:
        """
        Yield content_plane events only (excludes control_plane turns).

        Timing gaps from excluded turns are preserved - virtual time is
        NOT compressed. This ensures TokenGate sees realistic delays.
        """
        return self._iter_events("content_plane")

    # =========================================================================
    # Public API: Metadata
    # =========================================================================

    def get_metadata(self) -> TraceMeta:
        """Return parsed trace_meta record."""
        self._ensure_initialized()
        return self._metadata

    def get_derived_agent_labels(self) -> frozenset[str]:
        """
        Return the agent label set derived from this trace.

        Labels are derived from turn_boundary records only (authoritative source).
        """
        self._ensure_initialized()
        return self._derived_labels

    def get_turn_classifications(self) -> dict[int, TurnClassification]:
        """
        Return turn_id -> classification mapping.

        Every turn has a classification_reason for auditability.
        """
        self._ensure_initialized()
        return self._classifications.copy()

    # =========================================================================
    # Public API: Audit
    # =========================================================================

    def get_audit_flags(self) -> list[AuditFlag]:
        """
        Return list of audit flags for suspicious turns.

        Suspicious turns are classified as content_plane (conservative)
        but flagged for reviewer visibility.
        """
        self._ensure_initialized()
        return self._audit_flags.copy()


def replay_trace(
    trace_path: Path,
    stream: Literal["full", "content_plane"] = "full",
    **kwargs,
) -> Iterator[ReplayEvent]:
    """
    Convenience function to replay a trace.

    Args:
        trace_path: Path to trace file
        stream: Which stream to replay ("full" or "content_plane")
        **kwargs: Passed to TraceReplayEngine

    Yields:
        ReplayEvent objects
    """
    engine = TraceReplayEngine(trace_path, **kwargs)

    if stream == "full":
        yield from engine.replay_full()
    else:
        yield from engine.replay_content_plane()
