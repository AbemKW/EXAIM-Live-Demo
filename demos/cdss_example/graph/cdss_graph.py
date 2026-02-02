"""Simplified CDSS graph - orchestrator-driven workflow with specialist reasoning"""

from functools import partial
from langgraph.graph import StateGraph, END
from exaim_core.exaim import EXAIM
from demos.cdss_example.schema.graph_state import CDSSGraphState
from demos.cdss_example.agents.orchestrator_agent import OrchestratorAgent
from demos.cdss_example.agents.laboratory_agent import LaboratoryAgent
from demos.cdss_example.agents.cardiology_agent import CardiologyAgent
from demos.cdss_example.agents.internal_medicine_agent import InternalMedicineAgent
from demos.cdss_example.agents.radiology_agent import RadiologyAgent
from .nodes import (
    orchestrator_node,
    laboratory_node,
    cardiology_node,
    internal_medicine_node,
    radiology_node,
    synthesis_node
)
from .edges import (
    route_from_orchestrator,
    route_to_orchestrator
)


def build_cdss_graph(exaim: EXAIM):
    """Build and compile the simplified CDSS LangGraph workflow
    
    Instantiates all agents once with EXAIM reference and passes them to node functions.
    
    Args:
        exaim: EXAIM instance for token streaming and summarization
    
    Workflow:
    - Orchestrator compresses context and decides next specialist
    - Specialists provide domain reasoning based on case + summary + recent_delta
    - All specialists route back to orchestrator
    - Orchestrator eventually routes to synthesis
    - Synthesis ends the workflow
    """
    
    # Instantiate all agents once with EXAIM reference
    orchestrator = OrchestratorAgent(exaim=exaim)
    laboratory = LaboratoryAgent(exaim=exaim)
    cardiology = CardiologyAgent(exaim=exaim)
    internal_medicine = InternalMedicineAgent(exaim=exaim)
    radiology = RadiologyAgent(exaim=exaim)
    
    # Create the graph
    workflow = StateGraph(CDSSGraphState)
    
    # Add nodes with agent instances passed via partial functions (preserves async)
    workflow.add_node("orchestrator", partial(orchestrator_node, agent=orchestrator))
    workflow.add_node("laboratory", partial(laboratory_node, agent=laboratory))
    workflow.add_node("cardiology", partial(cardiology_node, agent=cardiology))
    workflow.add_node("internal_medicine", partial(internal_medicine_node, agent=internal_medicine))
    workflow.add_node("radiology", partial(radiology_node, agent=radiology))
    workflow.add_node("synthesis", partial(synthesis_node, agent=orchestrator))
    
    # Set entry point
    workflow.set_entry_point("orchestrator")
    
    # Add conditional edge from orchestrator to specialists or synthesis
    workflow.add_conditional_edges(
        "orchestrator",
        route_from_orchestrator,
        {
            "laboratory": "laboratory",
            "cardiology": "cardiology",
            "internal_medicine": "internal_medicine",
            "radiology": "radiology",
            "synthesis": "synthesis"
        }
    )
    
    # Add edges from all specialists back to orchestrator
    workflow.add_conditional_edges(
        "laboratory",
        route_to_orchestrator,
        {"orchestrator": "orchestrator"}
    )
    
    workflow.add_conditional_edges(
        "cardiology",
        route_to_orchestrator,
        {"orchestrator": "orchestrator"}
    )
    
    workflow.add_conditional_edges(
        "internal_medicine",
        route_to_orchestrator,
        {"orchestrator": "orchestrator"}
    )
    
    workflow.add_conditional_edges(
        "radiology",
        route_to_orchestrator,
        {"orchestrator": "orchestrator"}
    )
    
    # Add edge from synthesis to END
    workflow.add_edge("synthesis", END)
    
    # Compile the graph
    app = workflow.compile()
    
    return app



