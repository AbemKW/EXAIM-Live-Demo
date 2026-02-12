from pydantic import BaseModel, Field, validator
from typing import Optional

class AgentSummary(BaseModel):
    """Structured summary for medical multi-agent reasoning, optimized for physician understanding.
    
    Aligned with SBAR/SOAP documentation standards and XAI-CDSS principles for clinical decision support.
    
    Character limits are evidence-based, grounded in:
    - EHR alert & medication-alert usability research (concise, structured alerts reduce cognitive load)
    - Clinical summarization length choices (15-word limits for patient questions, "few words" for problem lists)
    - XAI explanation length and complexity studies (shorter explanations prevent cognitive overload)
    - Cognitive load & readability in clinical text (less clutter improves performance)
    """
    status_action: str = Field(
        # max_words used by summarizer enforcement (15-25 words; we enforce max 25)
        max_words=25,
        description="Concise description of what the system or agents have just done or are currently doing in the reasoning process. "
        "Plays a role similar to SBAR 'Situation', orienting the clinician to the current point in the workflow. "
        "Captures high-level multi-agent activity (e.g., 'retrieval completed, differential updated, uncertainty agent invoked'). "
        "Limit: ~15-25 words (~90-150 chars). Evidence: Alert style guides emphasize title brevity and minimal introductory text "
        "to reduce cognitive burden (Pourian et al., medication alert principles)."
    )
    key_findings: str = Field(
        max_words=30,
        description="The minimal set of clinical facts that are driving the current reasoning step, such as key symptoms, "
        "vital signs, lab results, imaging findings, or relevant history. Corresponds to SBAR 'Background' and SOAP 'Subjective/Objective'. "
        "Must link recommendations to concrete evidence so clinicians can verify or contest them. "
        "Limit: ~20-30 words (~120-180 chars). Evidence: Clinical summarization tasks explicitly require short outputs "
        "(Van Veen et al.: 15 words or less for patient questions, 'few words' for problem lists)."
    )
    differential_rationale: str = Field(
        max_words=35,
        description="A brief statement of the leading diagnostic hypotheses and why certain diagnoses are favored or deprioritized, "
        "expressed in clinical language. Aligns with the 'Assessment' section in SBAR and SOAP, which captures clinical interpretation. "
        "Gives clinicians a way to compare the system's thinking against their own mental model of the case. "
        "Limit: ~25-35 words (~150-210 chars). Evidence: XAI research (Lage et al.) shows longer explanations overload cognitive abilities; "
        "humans prefer explanations with 1-2 central causes (Molnar, Vilone & Longo)."
    )
    uncertainty_confidence: str = Field(
        max_words=20,
        description="A concise representation of model or system uncertainty, which may be probabilistic (e.g., class probabilities) "
        "or qualitative (e.g., 'high uncertainty', 'moderate confidence'). Essential for calibrated trust and safer human-AI collaboration, "
        "especially in ambiguous cases. "
        "Limit: ~10-20 words (~60-120 chars). Evidence: Trust calibration work shows complex uncertainty presentation can confuse users "
        "and harm trust; too much detail leads to cognitive overload and miscalibration (Lage et al., Kaur et al.)."
    )
    recommendation_next_step: str = Field(
        max_words=30,
        description="The specific diagnostic, therapeutic, or follow-up step that EXAIM suggests at this point, usually a short phrase or sentence. "
        "Corresponds to SBAR 'Recommendation' and SOAP 'Plan'. Provides clinicians with immediately actionable information in their workflow. "
        "Limit: ~15-30 words (~90-180 chars). Evidence: Alert-fatigue literature (Marcilly et al.) emphasizes concise, actionable alerts "
        "with clear response options; XAI evaluations show simpler, action-linked explanations are preferred."
    )
    agent_contributions: str = Field(
        max_words=25,
        description="A short list of which agents contributed to this step and how their outputs were used "
        "(e.g., 'Retrieval agent: latest PE guidelines; Differential agent: ranked CAP vs PE; Uncertainty agent: confidence estimates'). "
        "Addresses transparency in multi-agent systems, enabling fine-grained debugging and feedback. "
        "Limit: ~15-25 words (~90-150 chars). Evidence: Human-centered XAI design patterns recommend high-level, filtered explanation "
        "of pipelines; proof-based frameworks (SeXAI, Eccher et al.) explicitly omit intermediate steps to keep explanations short."
    )

    # --- Full Content Fields (populated when truncation occurs) ---
    full_status_action: Optional[str] = Field(default=None, description="Untruncated version of status_action")
    full_key_findings: Optional[str] = Field(default=None, description="Untruncated version of key_findings")
    full_differential_rationale: Optional[str] = Field(default=None, description="Untruncated version of differential_rationale")
    full_uncertainty_confidence: Optional[str] = Field(default=None, description="Untruncated version of uncertainty_confidence")
    full_recommendation_next_step: Optional[str] = Field(default=None, description="Untruncated version of recommendation_next_step")
    full_agent_contributions: Optional[str] = Field(default=None, description="Untruncated version of agent_contributions")

    @validator('status_action')
    def _status_action_word_limit(cls, v: str) -> str:
        max_words = 25
        if v is None:
            return v
        if len(str(v).split()) > max_words:
            raise ValueError(f"too_many_words, max_words={max_words}")
        return v

    @validator('key_findings')
    def _key_findings_word_limit(cls, v: str) -> str:
        max_words = 30
        if v is None:
            return v
        if len(str(v).split()) > max_words:
            raise ValueError(f"too_many_words, max_words={max_words}")
        return v

    @validator('differential_rationale')
    def _differential_rationale_word_limit(cls, v: str) -> str:
        max_words = 35
        if v is None:
            return v
        if len(str(v).split()) > max_words:
            raise ValueError(f"too_many_words, max_words={max_words}")
        return v

    @validator('uncertainty_confidence')
    def _uncertainty_confidence_word_limit(cls, v: str) -> str:
        max_words = 20
        if v is None:
            return v
        if len(str(v).split()) > max_words:
            raise ValueError(f"too_many_words, max_words={max_words}")
        return v

    @validator('recommendation_next_step')
    def _recommendation_next_step_word_limit(cls, v: str) -> str:
        max_words = 30
        if v is None:
            return v
        if len(str(v).split()) > max_words:
            raise ValueError(f"too_many_words, max_words={max_words}")
        return v

    @validator('agent_contributions')
    def _agent_contributions_word_limit(cls, v: str) -> str:
        max_words = 25
        if v is None:
            return v
        if len(str(v).split()) > max_words:
            raise ValueError(f"too_many_words, max_words={max_words}")
        return v
