"""Agent message schemas for CDSS.

Simplified schema - removed consultation, challenge, and debate structures.
Agents now only provide domain reasoning; orchestrator handles all coordination.
"""

from pydantic import BaseModel, Field


class ClinicalCase(BaseModel):
    """Clinical case input for the CDSS system"""
    
    case_text: str = Field(
        description="The clinical case description including patient history, symptoms, and test results"
    )

