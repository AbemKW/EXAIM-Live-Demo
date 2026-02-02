"""Simplified CDSS - orchestrator-driven multi-agent clinical decision support"""

from typing import Union
from exaim_core.exaim import EXAIM
from exaim_core.schema.agent_summary import AgentSummary
from demos.cdss_example.schema.clinical_case import ClinicalCase
from demos.cdss_example.graph.cdss_graph import build_cdss_graph
from demos.cdss_example.schema.graph_state import CDSSGraphState


class CDSS:
    """Clinical Decision Support System with simplified orchestrator-driven workflow"""
    
    def __init__(self):
        """Initialize CDSS with EXAIM and simplified graph workflow"""
        self.exaim = EXAIM()
        self.graph = build_cdss_graph(self.exaim)
    
    async def process_case(
        self, 
        case: Union[ClinicalCase, str],
        use_streaming: bool = True
    ) -> dict:
        """Process a clinical case through the simplified multi-agent system
        
        Args:
            case: ClinicalCase object or free-text case description
            use_streaming: Whether to use streaming (always True in this implementation)
            
        Returns:
            Dictionary containing final synthesis and summaries
        """
        # Convert case to clinical summary if it's a ClinicalCase object
        if isinstance(case, ClinicalCase):
            case_text = case.to_clinical_summary()
        else:
            case_text = str(case)
        
        # Initialize MAC-inspired graph state with message history
        initial_state: CDSSGraphState = {
            "case_text": case_text,
            "messages": [
                {
                    "role": "user",
                    "name": "system",
                    "content": case_text
                }
            ],
            "running_summary": "",
            "recent_delta": "",
            "recent_agent": "none",
            "next_specialist_to_call": None,
            "task_instruction": "",
            "specialists_called": [],
            "iteration_count": 0,
            "max_iterations": 13,  # MAC pattern: reduced from 20 to 13
            "final_synthesis": None
        }
        
        # Run the graph workflow
        final_state = await self.graph.ainvoke(initial_state)
        
        # Get all summaries after graph execution (for UI display only)
        all_summaries = self.get_all_summaries()
        
        # Compile results
        result = {
            "case_summary": case_text,
            "final_synthesis": final_state.get("final_synthesis", ""),
            "running_summary": final_state.get("running_summary", ""),
            "specialists_called": final_state.get("specialists_called", []),
            "iteration_count": final_state.get("iteration_count", 0),
            # EXAIM summaries for UI display only - MAS workflow does not depend on these
            "agent_summaries": [
                {
                    "status_action": s.status_action,
                    "key_findings": s.key_findings,
                    "differential_rationale": s.differential_rationale,
                    "uncertainty_confidence": s.uncertainty_confidence,
                    "recommendation_next_step": s.recommendation_next_step,
                    "agent_contributions": s.agent_contributions
                }
                for s in all_summaries
            ]
        }
        
        return result
    
    def get_all_summaries(self) -> list[AgentSummary]:
        """Get all summaries from EXAIM
        
        NOTE: EXAIM summaries are for UI display only.
        The MAS workflow does not depend on these summaries.
        Orchestrator maintains its own running_summary for workflow decisions.
        """
        return self.exaim.get_all_summaries()
    
    def get_summaries_by_agent(self, agent_id: str) -> list[AgentSummary]:
        """Get summaries for a specific agent
        
        NOTE: EXAIM summaries are for UI display only.
        The MAS workflow does not depend on these summaries.
        """
        return self.exaim.get_summaries_by_agent(agent_id)
    
    def reset(self):
        """Reset the CDSS system (create new EXAIM instance)"""
        self.exaim = EXAIM()

