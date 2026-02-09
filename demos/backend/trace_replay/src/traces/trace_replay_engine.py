#!/usr/bin/env python3
"""
Trace replay engine shim for demos/backend local package.
"""

from pathlib import Path

from .classifier import classify_turn, is_suspicious_label_like
from .models import (
    AuditFlag,
    ReplayEvent,
    StubTraceError,
    TraceReplayError,
    TraceValidationError,
)
from .parser import derive_agent_labels, iter_trace_records
from .replay import TraceReplayEngine, replay_trace

__all__ = [
    "AuditFlag",
    "ReplayEvent",
    "StubTraceError",
    "TraceReplayEngine",
    "TraceReplayError",
    "TraceValidationError",
    "classify_turn",
    "derive_agent_labels",
    "is_suspicious_label_like",
    "iter_trace_records",
    "replay_trace",
]


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python -m evals.cli.replay_trace <trace_file.jsonl.gz>")
        sys.exit(1)

    trace_path = Path(sys.argv[1])

    try:
        engine = TraceReplayEngine(trace_path, strict_stub_guard=False)
        meta = engine.get_metadata()
        labels = engine.get_derived_agent_labels()
        classifications = engine.get_turn_classifications()

        print("=== Trace Metadata ===")
        print(f"case_id: {meta.case_id}")
        print(f"schema_version: {meta.schema_version}")
        print(f"stub_mode: {meta.stub_mode}")
        print(f"derived_agent_labels: {labels}")
        print()

        print("=== Turn Classifications ===")
        content_count = sum(
            1 for c in classifications.values() if c.turn_type == "content_plane"
        )
        control_count = sum(
            1 for c in classifications.values() if c.turn_type == "control_plane"
        )
        print(
            f"Total: {len(classifications)} turns ({content_count} content_plane, {control_count} control_plane)"
        )

        for turn_id in sorted(classifications.keys()):
            cls = classifications[turn_id]
            text_preview = (
                cls.turn_text[:50].replace("\n", " ") + "..."
                if len(cls.turn_text) > 50
                else cls.turn_text.replace("\n", " ")
            )
            print(
                f"  turn={turn_id}: {cls.turn_type:14} | {cls.classification_reason:30} | \"{text_preview}\""
            )

        flags = engine.get_audit_flags()
        if flags:
            print()
            print(f"=== Audit Flags ({len(flags)}) ===")
            for flag in flags:
                print(f"  turn={flag.turn_id}: {flag.flag_type} - {flag.details[:80]}")

    except TraceReplayError as e:
        print(f"ERROR: {e}")
        sys.exit(1)
