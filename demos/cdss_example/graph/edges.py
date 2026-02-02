"""Simplified CDSS graph edges - orchestrator-driven routing"""

from typing import Literal
from demos.cdss_example.schema.graph_state import CDSSGraphState


def route_from_orchestrator(state: CDSSGraphState) -> Literal["laboratory", "cardiology", "internal_medicine", "radiology", "synthesis"]:
    """Route from orchestrator based on next_specialist_to_call
    
    MAC-inspired routing: Orchestrator detects TERMINATE keyword and sets next_specialist_to_call to "synthesis".
    Otherwise, routes to the specialist selected by the supervisor's LLM decision.
    
    Args:
        state: Current graph state
        
    Returns:
        Specialist name or "synthesis"
    """
    import logging
    logger = logging.getLogger(__name__)
    
    next_specialist = state.get("next_specialist_to_call", "synthesis")
    logger.info(f"[ROUTING] Reading next_specialist_to_call from state: '{next_specialist}'")
    
    # Validate and return
    valid_options = ["laboratory", "cardiology", "internal_medicine", "radiology", "synthesis"]
    if next_specialist in valid_options:
        logger.info(f"[ROUTING] Valid specialist found, routing to: '{next_specialist}'")
        return next_specialist
    
    # Default to synthesis if invalid
    logger.warning(f"[ROUTING] Invalid specialist '{next_specialist}', defaulting to synthesis")
    return "synthesis"


def route_to_orchestrator(state: CDSSGraphState) -> Literal["orchestrator"]:
    """Route specialist nodes back to orchestrator
    
    All specialists return to orchestrator for compression and routing decisions.
    
    Args:
        state: Current graph state
        
    Returns:
        Always "orchestrator"
    """
    return "orchestrator"


