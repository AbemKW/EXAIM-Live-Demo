from pydantic import BaseModel, Field
from typing import Literal


class BufferAnalysis(BaseModel):
    """Structured analysis of stream state for buffer agent decision-making.

    Uses a three-state machine to classify stream completeness based on topic continuity,
    then independently evaluates clinical relevance and novelty to determine if summarization
    should be triggered.
    """
    rationale: str = Field(
        description="Brief justification (<=240 chars) for the decision, referencing completeness/relevance/novelty/stream_state and trigger path if triggered."
    )
    stream_state: Literal["SAME_TOPIC_CONTINUING", "TOPIC_SHIFT", "CRITICAL_ALERT"] = Field(
        description="State machine for stream completeness based on topic continuity. "
        "SAME_TOPIC_CONTINUING: Agent is still refining, listing, or explaining the same specific clinical issue "
        "(e.g., mid-list with markers like '1.', 'First,', reasoning loops, adding detail). WAIT - do not trigger. "
        "TOPIC_SHIFT: Agent moves to a distinctly different clinical subproblem, workup branch, plan section, organ system, problem-list item, "
        "or new reasoning phase (e.g., moving from assessment to plan, from one differential branch to another, from one organ system to another, "
        "from data gathering to interpretation, from interpretation to action). Includes explicit transitions, implicit shifts, conclusions of one section, "
        "and new problem list items. PROCEED to relevance/novelty checks. "
        "CRITICAL_ALERT: Immediate life-safety notification (e.g., 'V-Fib detected', 'Code Blue'). PROCEED immediately."
    )
    is_relevant: bool = Field(
        description="Is this update INTERRUPTION-WORTHY for the clinician right now? "
        "Set true ONLY for HIGH-VALUE deltas: new/changed clinical action/plan (start/stop/order/monitor/consult/dose/contraindication), "
        "new/changed interpretation or diagnostic stance (favored dx, deprioritized dx, rationale, confidence shift), "
        "new/changed abnormal finding that materially changes the mental model (new imaging result, new lab abnormality, notable value change), "
        "or safety-critical content. "
        "Set false for isolated facts not clearly new/changed/abnormal, minor elaboration/repetition/narrative filler, "
        "or 'thinking out loud'/workflow chatter. "
        "Atomic finding or action stated as a complete sentence/list item, when clinically meaningful (e.g., 'MRI shows cerebellar atrophy.') is relevant."
    )
    is_novel: bool = Field(
        description="STRICT: Is this TRULY new vs previous summaries? Does it introduce something substantively different "
        "not already covered in prior summaries? Novel: NEW values/changes (e.g., 'Creatinine rose to 2.2' when previously 1.8), "
        "NEW actions not previously mentioned (e.g., 'Start Amiodarone' when not in prior summaries), "
        "NEW insights with changed reasoning (e.g., 'Diagnosis upgraded from possible to likely'). "
        "NOT Novel: Rephrasing same findings, restating same differentials, confirming existing plans, "
        "adding minor details to already-summarized content, continuing the same line of reasoning without new conclusions, "
        "same clinical content with only added adjectives or restated rationale that does not change the plan or diagnostic stance. "
        "If new_trace is a single sentence and does not introduce a new action, a new dx shift, or a new abnormal/value change, default is_novel = false."
    )
    is_complete: bool = Field(
        description="Has the stream reached a CLOSED unit (phrase-level structural closure) when evaluating CONCAT = previous_trace + new_trace? "
        "Set true if the latest content completes a meaningful phrase-level unit: "
        "- finishes a complete clause (subject-verb-object or subject-verb-complement structure with resolved meaning), "
        "- completes an action statement as a full inference unit (e.g., 'Start Amiodarone' or 'MRI shows cerebellar atrophy'), "
        "- completes a diagnostic inference as a full unit (e.g., 'Likely diagnosis is X because Y'), "
        "- finishes a list item that forms a complete thought (not just scaffolding), "
        "- ends with sentence-final punctuation (., !, ?) that closes a complete thought. "
        "Set false if: "
        "- ends mid-clause with unresolved dependencies, "
        "- contains incomplete reasoning chains (e.g., 'because' without conclusion, 'consider' without resolution), "
        "- ends with forward references that lack resolution ('also consider...', 'next...' without completion), "
        "- list scaffolding without a completed meaningful item. "
        "Focus on phrase-level closure (finished clauses/inference/action units), not just word-level end tokens. "
        "Evaluate completeness on previous_trace + new_trace concatenation, not new_trace alone."
    )


class BufferAnalysisNoNovelty(BaseModel):
    """Structured analysis of stream state for buffer agent decision-making without novelty check.

    Uses a three-state machine to classify stream completeness based on topic continuity,
    then independently evaluates clinical relevance to determine if summarization should be triggered.
    """
    rationale: str = Field(
        description="Brief justification (<=240 chars) for the decision, referencing completeness/relevance/stream_state and trigger path if triggered."
    )
    stream_state: Literal["SAME_TOPIC_CONTINUING", "TOPIC_SHIFT", "CRITICAL_ALERT"] = Field(
        description="State machine for stream completeness based on topic continuity. "
        "SAME_TOPIC_CONTINUING: Agent is still refining, listing, or explaining the same specific clinical issue "
        "(e.g., mid-list with markers like '1.', 'First,', reasoning loops, adding detail). WAIT - do not trigger. "
        "TOPIC_SHIFT: Agent explicitly moves to a distinctly different organ system, problem, or section "
        "(explicit transitions, implicit shifts, conclusions, new problem list items). PROCEED to relevance checks. "
        "CRITICAL_ALERT: Immediate life-safety notification (e.g., 'V-Fib detected', 'Code Blue'). PROCEED immediately."
    )
    is_relevant: bool = Field(
        description="Is this update INTERRUPTION-WORTHY for the clinician right now? "
        "Set true ONLY for HIGH-VALUE deltas: new/changed clinical action/plan (start/stop/order/monitor/consult/dose/contraindication), "
        "new/changed interpretation or diagnostic stance (favored dx, deprioritized dx, rationale, confidence shift), "
        "new/changed abnormal finding that materially changes the mental model (new imaging result, new lab abnormality, notable value change), "
        "or safety-critical content. "
        "Set false for isolated facts not clearly new/changed/abnormal, minor elaboration/repetition/narrative filler, "
        "or 'thinking out loud'/workflow chatter. "
        "Atomic finding or action stated as a complete sentence/list item, when clinically meaningful (e.g., 'MRI shows cerebellar atrophy.') is relevant."
    )
    is_complete: bool = Field(
        description="Has the stream reached a CLOSED unit (phrase-level structural closure) when evaluating CONCAT = previous_trace + new_trace? "
        "Set true if the latest content completes a meaningful phrase-level unit: "
        "- finishes a complete clause (subject-verb-object or subject-verb-complement structure with resolved meaning), "
        "- completes an action statement as a full inference unit (e.g., 'Start Amiodarone' or 'MRI shows cerebellar atrophy'), "
        "- completes a diagnostic inference as a full unit (e.g., 'Likely diagnosis is X because Y'), "
        "- finishes a list item that forms a complete thought (not just scaffolding), "
        "- ends with sentence-final punctuation (., !, ?) that closes a complete thought. "
        "Set false if: "
        "- ends mid-clause with unresolved dependencies, "
        "- contains incomplete reasoning chains (e.g., 'because' without conclusion, 'consider' without resolution), "
        "- ends with forward references that lack resolution ('also consider...', 'next...' without completion), "
        "- list scaffolding without a completed meaningful item. "
        "Focus on phrase-level closure (finished clauses/inference/action units), not just word-level end tokens. "
        "Evaluate completeness on previous_trace + new_trace concatenation, not new_trace alone."
    )
