"""Turn classification helpers for trace replay."""

from __future__ import annotations

from .models import AuditFlag, TurnClassification


# Sentinel values that trigger control_plane classification
TERMINATE_SENTINELS = frozenset({"TERMINATE"})


def is_suspicious_label_like(turn_text: str, derived_labels: frozenset[str]) -> bool:
    """
    Detect turns that LOOK like speaker-selection but don't match derived labels.

    Suspicious if:
        - Single word (after strip)
        - Looks identifier-ish (alphanumeric, possibly with digits)
        - NOT in derived_agent_labels

    Args:
        turn_text: Reconstructed turn text
        derived_labels: Set of known agent labels

    Returns:
        True if suspicious (should be flagged for audit)
    """
    text = turn_text.strip()
    words = text.split()

    if len(words) != 1:
        return False

    word = words[0].lower()

    # Already matched -> not suspicious (it was filtered)
    if word in derived_labels:
        return False

    # Looks identifier-ish? (contains letters, possibly digits, no spaces)
    if word.replace("_", "").isalnum() and any(c.isalpha() for c in word):
        return True

    return False


def classify_turn(
    turn_id: int,
    turn_text: str,
    delta_count: int,
    agent_id: str,
    start_seq: int,
    end_seq: int,
    derived_agent_labels: frozenset[str],
    audit_flags: list[AuditFlag],
) -> TurnClassification:
    """
    Conservative classification: exact match only.

    Classification rules (in order):
    1. Exact agent label match → control_plane
    2. TERMINATE sentinel → control_plane
    3. Empty/whitespace-only turn → control_plane
    4. Default → content_plane

    Suspicious turns are kept as content_plane and flagged.

    Args:
        turn_id: Turn identifier
        turn_text: Reconstructed text from deltas
        delta_count: Number of deltas in this turn
        agent_id: Agent that produced this turn
        start_seq: First seq in turn
        end_seq: Last seq in turn
        derived_agent_labels: Set of known agent labels
        audit_flags: List to append audit flags to (mutated)

    Returns:
        TurnClassification with type and reason
    """
    normalized = turn_text.strip().lower()

    # Rule 1: Exact agent label match
    if normalized in derived_agent_labels:
        return TurnClassification(
            turn_id=turn_id,
            turn_type="control_plane",
            agent_id=agent_id,
            turn_text=turn_text,
            classification_reason=f"exact_label_match:{normalized}",
            delta_count=delta_count,
            start_seq=start_seq,
            end_seq=end_seq,
        )

    # Rule 2: TERMINATE sentinel (exact, case-insensitive)
    if turn_text.strip().upper() in TERMINATE_SENTINELS:
        return TurnClassification(
            turn_id=turn_id,
            turn_type="control_plane",
            agent_id=agent_id,
            turn_text=turn_text,
            classification_reason="terminate_sentinel",
            delta_count=delta_count,
            start_seq=start_seq,
            end_seq=end_seq,
        )

    # Rule 3: Empty turn (whitespace-only)
    if not turn_text.strip():
        return TurnClassification(
            turn_id=turn_id,
            turn_type="control_plane",
            agent_id=agent_id,
            turn_text=turn_text,
            classification_reason="empty_turn",
            delta_count=delta_count,
            start_seq=start_seq,
            end_seq=end_seq,
        )

    # Default: content_plane
    classification = TurnClassification(
        turn_id=turn_id,
        turn_type="content_plane",
        agent_id=agent_id,
        turn_text=turn_text,
        classification_reason="default_content",
        delta_count=delta_count,
        start_seq=start_seq,
        end_seq=end_seq,
    )

    # Check for suspicious label-like (add audit flag, don't filter)
    if is_suspicious_label_like(turn_text, derived_agent_labels):
        audit_flags.append(
            AuditFlag(
                turn_id=turn_id,
                flag_type="suspicious_label_like_unmatched",
                details=(
                    f"turn_text='{turn_text.strip()}' not in "
                    f"derived_agent_labels={derived_agent_labels}"
                ),
            )
        )

    return classification
