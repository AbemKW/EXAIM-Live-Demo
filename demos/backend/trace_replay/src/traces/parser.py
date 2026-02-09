"""Trace parsing helpers for replay workflows."""

from __future__ import annotations

import gzip
import json
from pathlib import Path
from typing import Iterator

from .models import TraceValidationError


# Agent IDs to exclude from derived label set
EXCLUDED_AGENT_IDS = frozenset({"unknown", "null", "none", ""})


def iter_trace_records(trace_path: Path) -> Iterator[dict]:
    """
    Iterate over ALL records from a trace file.

    Handles both .jsonl and .jsonl.gz files.
    """
    open_fn = gzip.open if str(trace_path).endswith(".gz") else open
    with open_fn(trace_path, "rt", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as e:
                raise TraceValidationError(
                    f"Invalid JSON at {trace_path}:{line_num}: {e}"
                )


def derive_agent_labels(trace_path: Path) -> frozenset[str]:
    """
    Pass 1: Scan turn_boundary records to collect agent labels.

    Deterministic:
        - Processes records in file order
        - Collects agent_id from turn_boundary records ONLY (authoritative)
        - Normalizes to lowercase
        - Excludes null/empty/"unknown"

    Args:
        trace_path: Path to trace JSONL file

    Returns:
        Frozen set of normalized labels (e.g., {"doctor0", "supervisor"})
    """
    labels = set()

    for record in iter_trace_records(trace_path):
        # Only consider turn_boundary records (authoritative source)
        if record.get("record_type") != "turn_boundary":
            continue

        agent_id = record.get("agent_id")
        if agent_id:
            normalized = agent_id.strip().lower()
            if normalized and normalized not in EXCLUDED_AGENT_IDS:
                labels.add(normalized)

    return frozenset(labels)
