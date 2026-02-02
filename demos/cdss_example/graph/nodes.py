"""Simplified CDSS graph nodes - orchestrator-driven workflow with specialist reasoning"""

import logging
import re
from typing import Dict, Any
from demos.cdss_example.schema.graph_state import CDSSGraphState
from demos.cdss_example.agents.orchestrator_agent import OrchestratorAgent
from demos.cdss_example.agents.laboratory_agent import LaboratoryAgent
from demos.cdss_example.agents.cardiology_agent import CardiologyAgent
from demos.cdss_example.agents.internal_medicine_agent import InternalMedicineAgent
from demos.cdss_example.agents.radiology_agent import RadiologyAgent

logger = logging.getLogger(__name__)

_send_agent_started_import_warned = False

def get_send_agent_started():
    """Safely import send_agent_started to avoid circular imports"""
    global _send_agent_started_import_warned
    try:
        from demos.backend.server import send_agent_started
        return send_agent_started
    except ImportError:
        if not _send_agent_started_import_warned:
            logger.warning(
                "send_agent_started is not available: could not import from 'demos.backend.server'. "
                "Agent started UI notifications will be skipped."
            )
            _send_agent_started_import_warned = True
        return None


def _build_specialist_context(state: CDSSGraphState, specialist_name: str) -> str:
    """Build context string for specialist
    
    Args:
        state: Current graph state
        specialist_name: Name of the specialist receiving this context
        
    Returns:
        Formatted context string with case, summary, recent update, and task
    """
    case_text = state["case_text"]
    running_summary = state.get("running_summary", "")
    recent_delta = state.get("recent_delta", "")
    recent_agent = state.get("recent_agent", "none")
    task_instruction = state.get("task_instruction", "")
    
    # Build context
    context_parts = [f"CLINICAL CASE:\n{case_text}"]
    
    if running_summary:
        context_parts.append(f"\n\nSUMMARY SO FAR:\n{running_summary}")
    
    if recent_delta and recent_agent != "none":
        context_parts.append(f"\n\nRECENT UPDATE FROM {recent_agent.upper()}:\n{recent_delta}")
    
    if task_instruction:
        context_parts.append(f"\n\nYOUR TASK:\n{task_instruction}")
    
    return "".join(context_parts)


async def orchestrator_node(state: CDSSGraphState, agent: OrchestratorAgent) -> Dict[str, Any]:
    """Orchestrator node: compresses context from full message history, decides next specialist, generates task instruction
    
    MAC-inspired supervisor pattern:
    - Sees full message history (all specialist outputs) like MAC's GroupChat
    - Compresses into running_summary for specialist bounded context
    - Detects TERMINATE keyword for ending conversation
    
    Agent instance is injected via closure from graph builder.
    
    Workflow:
    1. If recent_delta exists, compress it using full messages history context (streams to UI)
    2. Decide next specialist or TERMINATE (streams to UI)
    3. If not TERMINATE, generate task instruction for next specialist (streams to UI)
    4. Update state and return
    """
    case_text = state["case_text"]
    running_summary = state.get("running_summary", "")
    recent_delta = state.get("recent_delta", "")
    recent_agent = state.get("recent_agent", "none")
    specialists_called = state.get("specialists_called", [])
    iteration_count = state.get("iteration_count", 0)
    max_iterations = state.get("max_iterations", 13)
    messages = state.get("messages", [])
    
    # Check for max iterations to prevent infinite loops
    if iteration_count >= max_iterations:
        logger.warning(f"Max iterations ({max_iterations}) reached. Forcing synthesis.")
        return {
            "running_summary": running_summary,
            "next_specialist_to_call": "synthesis",
            "task_instruction": "",
            "iteration_count": iteration_count + 1
        }
    
    # Step 1: Compress recent specialist output into running_summary using full message history
    if recent_delta and recent_agent != "none":
        # Build message history context for compression (last 5 messages for context)
        message_history_text = ""
        if messages:
            recent_messages = messages[-5:]  # Last 5 messages for bounded context
            message_history_text = "\n\n".join([
                f"[{msg.get('role', 'unknown').upper()} - {msg.get('name', 'unknown')}]:\n{msg.get('content', '')}"
                for msg in recent_messages
            ])
        
        # Build compression prompt with full conversation context
        if not running_summary:
            compression_input = (
                f"CONVERSATION HISTORY:\n{message_history_text}\n\n"
                f"A specialist ({recent_agent}) has provided analysis of a clinical case.\n\n"
                f"Specialist Output:\n{recent_delta}\n\n"
                f"Create a concise summary capturing:\n"
                f"- Key clinical findings\n"
                f"- Differential diagnoses\n"
                f"- Recommended tests or interventions\n"
                f"- Critical concerns or urgent issues\n\n"
                f"Keep it focused and actionable."
            )
        else:
            compression_input = (
                f"CONVERSATION HISTORY:\n{message_history_text}\n\n"
                f"Previous Summary:\n{running_summary}\n\n"
                f"New Findings from {recent_agent.upper()}:\n{recent_delta}\n\n"
                f"Generate an updated concise summary that:\n"
                f"- Integrates the new findings from the conversation history\n"
                f"- Maintains key information from previous summary\n"
                f"- Removes redundant or superseded information\n"
                f"- Keeps focus on differential diagnosis and clinical decision-making\n"
                f"- Stays within ~300-500 tokens\n\n"
                f"Provide only the updated summary."
            )
        
        # Send agent_started UI event before streaming
        send_fn = get_send_agent_started()
        if send_fn:
            await send_fn(agent.agent_id)
        
        # Stream compression - agent.stream() handles EXAIM internally
        collected_tokens = []
        async for token in agent.stream(compression_input):
            collected_tokens.append(token)
        
        running_summary = "".join(collected_tokens)
        
        # Append supervisor's compression to message history
        messages.append({
            "role": "supervisor",
            "name": "Orchestrator",
            "content": running_summary
        })
    
    # Step 2: Decide next specialist or TERMINATE
    available_specialists = ['laboratory', 'cardiology', 'internal_medicine', 'radiology']
    not_called = [s for s in available_specialists if s not in specialists_called]
    
    decision_input = (
        f"Clinical Case:\n{case_text}\n\n"
        f"Running Summary:\n{running_summary}\n\n"
        f"Recent Contributor: {recent_agent}\n"
        f"Specialists Called: {', '.join(specialists_called) if specialists_called else 'none yet'}\n"
        f"Available Specialists: {', '.join(not_called) if not_called else 'all have contributed'}\n\n"
        f"Decide the next action:\n"
        f"- If critical questions remain that a specific specialist should address, "
        f"respond with ONLY the specialist name: 'laboratory' OR 'cardiology' OR 'internal_medicine' OR 'radiology'\n"
        f"- If ALL termination criteria are met (all specialists agree, no further discussion needed, "
        f"all diagnostic possibilities explored, all tests justified), respond with ONLY: 'TERMINATE'\n\n"
        f"Consider:\n"
        f"- What clinical questions remain unanswered?\n"
        f"- What specialist expertise would help clarify the diagnosis or treatment?\n"
        f"- Have the key specialists for this case contributed?\n"
        f"- Is there sufficient information and consensus for a final recommendation?\n\n"
        f"Respond with ONLY ONE WORD and NOTHING ELSE: one of: laboratory | cardiology | internal_medicine | radiology | TERMINATE.\n"
        f"Do not add punctuation, explanation, or quotes."
    )
    
    # Send agent_started UI event before streaming
    send_fn = get_send_agent_started()
    if send_fn:
        await send_fn(agent.agent_id)
    
    # Stream decision - agent.stream() handles EXAIM internally
    collected_decision = []
    async for token in agent.stream(decision_input):
        collected_decision.append(token)
    
    # Extract and validate decision (log raw response for debugging)
    raw_decision_text = "".join(collected_decision)
    logger.info(f"Raw specialist decision from LLM: '{raw_decision_text}'")
    
    # Check for TERMINATE keyword (case-insensitive, exact match after stripping)
    if raw_decision_text.strip().upper() == "TERMINATE":
        next_specialist = "synthesis"
        logger.info("TERMINATE detected in supervisor decision. Routing to synthesis.")
    else:
        # Try to extract specialist name from the response
        # First, check if the entire response (stripped and lowercased) is a valid option
        next_specialist = raw_decision_text.strip().lower()
        valid_options = available_specialists
        
        if next_specialist not in valid_options:
            # Fallback: The LLM response may contain the specialist name embedded in text 
            # (e.g., "DECIDE THE NEXT SPECIALIST: radiology" or "I think cardiology would be best")
            # Search for any valid specialist name in the raw text
            combined_pattern = r"\b(" + "|".join(re.escape(opt) for opt in valid_options) + r")\b"
            match = re.search(combined_pattern, raw_decision_text.lower())  # Search in the original text, not the variable
            if match:
                next_specialist = match.group(1)
                logger.info(f"Extracted specialist '{next_specialist}' from response: '{raw_decision_text[:100]}'")
            else:
                logger.warning(f"Invalid specialist response '{raw_decision_text[:100]}...'. Defaulting to 'synthesis'.")
                next_specialist = 'synthesis'
        else:
            logger.info(f"Direct match: specialist is '{next_specialist}'")
    
    logger.info(f"Final next_specialist decision: '{next_specialist}'")
    
    # Step 3: Generate task instruction if not synthesis
    task_instruction = ""
    if next_specialist != "synthesis":
        # Map specialist internal name to agent display name
        specialist_display_names = {
            'laboratory': 'Laboratory',
            'cardiology': 'Cardiology', 
            'internal_medicine': 'Internal Medicine',
            'radiology': 'Radiology'
        }
        specialist_display = specialist_display_names.get(next_specialist, next_specialist.upper())
        
        task_input = (
            f"Clinical Case:\n{case_text}\n\n"
            f"Running Summary:\n{running_summary}\n\n"
            f"Recent Update from {recent_agent.upper()}:\n{recent_delta}\n\n"
            f"You are preparing a task for the {specialist_display} specialist.\n\n"
            f"Generate a specific, focused task instruction that tells them:\n"
            f"- What aspect of the case they should focus on\n"
            f"- What questions they should address\n"
            f"- What prior findings they should consider or verify\n"
            f"- What their analysis should contribute to the diagnosis/treatment plan\n\n"
            f"Keep it concise (2-4 sentences) and actionable.\n"
            f"Do NOT add any headers or formatting - provide only the task description itself."
        )
        
        # Send agent_started UI event before streaming
        send_fn = get_send_agent_started()
        if send_fn:
            await send_fn(agent.agent_id)
        
        # Stream task generation - agent.stream() handles EXAIM internally
        collected_task = []
        async for token in agent.stream(task_input):
            collected_task.append(token)
        
        task_instruction = "".join(collected_task)
    
    logger.info(f"[ORCHESTRATOR] Returning state update with next_specialist_to_call='{next_specialist}'")
    
    return {
        "running_summary": running_summary,
        "next_specialist_to_call": next_specialist,
        "task_instruction": task_instruction,
        "iteration_count": iteration_count + 1,
        "messages": messages
    }


async def laboratory_node(state: CDSSGraphState, agent: LaboratoryAgent) -> Dict[str, Any]:
    """Laboratory specialist node: analyze labs and provide domain reasoning
    
    Agent instance is injected via closure from graph builder.
    """
    # Build context for specialist
    context = _build_specialist_context(state, "laboratory")
    
    # Send agent_started UI event before streaming
    send_fn = get_send_agent_started()
    if send_fn:
        await send_fn(agent.agent_id)
    
    # Stream specialist reasoning - agent.stream() handles EXAIM internally
    collected_tokens = []
    async for token in agent.stream(context):
        collected_tokens.append(token)
    
    # Collect raw output as recent_delta
    recent_delta = "".join(collected_tokens)
    
    # Update specialists_called
    specialists_called = state.get("specialists_called", [])
    if "laboratory" not in specialists_called:
        specialists_called = specialists_called + ["laboratory"]
    
    # Append to message history
    messages = state.get("messages", [])
    messages.append({
        "role": "specialist",
        "name": "Laboratory",
        "content": recent_delta
    })
    
    return {
        "recent_delta": recent_delta,
        "recent_agent": "laboratory",
        "specialists_called": specialists_called,
        "messages": messages
    }


async def cardiology_node(state: CDSSGraphState, agent: CardiologyAgent) -> Dict[str, Any]:
    """Cardiology specialist node: assess cardiovascular findings and provide domain reasoning
    
    Agent instance is injected via closure from graph builder.
    """
    # Build context for specialist
    context = _build_specialist_context(state, "cardiology")
    
    # Send agent_started UI event before streaming
    send_fn = get_send_agent_started()
    if send_fn:
        await send_fn(agent.agent_id)
    
    # Stream specialist reasoning - agent.stream() handles EXAIM internally
    collected_tokens = []
    async for token in agent.stream(context):
        collected_tokens.append(token)
    
    # Collect raw output as recent_delta
    recent_delta = "".join(collected_tokens)
    
    # Update specialists_called
    specialists_called = state.get("specialists_called", [])
    if "cardiology" not in specialists_called:
        specialists_called = specialists_called + ["cardiology"]
    
    # Append to message history
    messages = state.get("messages", [])
    messages.append({
        "role": "specialist",
        "name": "Cardiology",
        "content": recent_delta
    })
    
    return {
        "recent_delta": recent_delta,
        "recent_agent": "cardiology",
        "specialists_called": specialists_called,
        "messages": messages
    }


async def internal_medicine_node(state: CDSSGraphState, agent: InternalMedicineAgent) -> Dict[str, Any]:
    """Internal Medicine specialist node: comprehensive clinical assessment
    
    Agent instance is injected via closure from graph builder.
    """
    # Build context for specialist
    context = _build_specialist_context(state, "internal_medicine")
    
    # Send agent_started UI event before streaming
    send_fn = get_send_agent_started()
    if send_fn:
        await send_fn(agent.agent_id)
    
    # Stream specialist reasoning - agent.stream() handles EXAIM internally
    collected_tokens = []
    async for token in agent.stream(context):
        collected_tokens.append(token)
    
    # Collect raw output as recent_delta
    recent_delta = "".join(collected_tokens)
    
    # Update specialists_called
    specialists_called = state.get("specialists_called", [])
    if "internal_medicine" not in specialists_called:
        specialists_called = specialists_called + ["internal_medicine"]
    
    # Append to message history
    messages = state.get("messages", [])
    messages.append({
        "role": "specialist",
        "name": "InternalMedicine",
        "content": recent_delta
    })
    
    return {
        "recent_delta": recent_delta,
        "recent_agent": "internal_medicine",
        "specialists_called": specialists_called,
        "messages": messages
    }


async def radiology_node(state: CDSSGraphState, agent: RadiologyAgent) -> Dict[str, Any]:
    """Radiology specialist node: interpret imaging findings and provide domain reasoning
    
    Agent instance is injected via closure from graph builder.
    """
    # Build context for specialist
    context = _build_specialist_context(state, "radiology")
    
    # Send agent_started UI event before streaming
    send_fn = get_send_agent_started()
    if send_fn:
        await send_fn(agent.agent_id)
    
    # Stream specialist reasoning - agent.stream() handles EXAIM internally
    collected_tokens = []
    async for token in agent.stream(context):
        collected_tokens.append(token)
    
    # Collect raw output as recent_delta
    recent_delta = "".join(collected_tokens)
    
    # Update specialists_called
    specialists_called = state.get("specialists_called", [])
    if "radiology" not in specialists_called:
        specialists_called = specialists_called + ["radiology"]
    
    # Append to message history
    messages = state.get("messages", [])
    messages.append({
        "role": "specialist",
        "name": "Radiology",
        "content": recent_delta
    })
    
    return {
        "recent_delta": recent_delta,
        "recent_agent": "radiology",
        "specialists_called": specialists_called,
        "messages": messages
    }


async def synthesis_node(state: CDSSGraphState, agent: OrchestratorAgent) -> Dict[str, Any]:
    """Synthesis node: orchestrator generates final clinical recommendations
    
    Uses full message history for comprehensive context (MAC-inspired).
    
    Agent instance is injected via closure from graph builder.
    """
    case_text = state["case_text"]
    running_summary = state.get("running_summary", "")
    messages = state.get("messages", [])
    
    # Build message history context (last 5 messages for bounded context)
    message_history_text = ""
    if messages:
        recent_messages = messages[-5:]
        message_history_text = "\n\n".join([
            f"[{msg.get('role', 'unknown').upper()} - {msg.get('name', 'unknown')}]:\n{msg.get('content', '')}"
            for msg in recent_messages
        ])
    
    # Build synthesis prompt with full conversation context
    synthesis_prompt = (
        f"CLINICAL CASE:\n{case_text}\n\n"
        f"RUNNING SUMMARY OF SPECIALIST FINDINGS:\n{running_summary}\n\n"
        f"RECENT CONVERSATION HISTORY:\n{message_history_text}\n\n"
        f"As the coordinating physician, synthesize all findings into a comprehensive final assessment. Provide:\n"
        f"1. Clinical Summary: Brief overview of the patient and key findings\n"
        f"2. Differential Diagnosis: Most likely diagnoses ranked by probability\n"
        f"3. Critical Findings: Any urgent concerns requiring immediate attention\n"
        f"4. Recommended Management: Specific treatment recommendations and interventions\n"
        f"5. Follow-up Plan: Monitoring and additional testing recommendations\n\n"
        f"Provide clear, actionable clinical recommendations."
    )
    
    # Send agent_started UI event before streaming
    send_fn = get_send_agent_started()
    if send_fn:
        await send_fn(agent.agent_id)
    
    # Stream synthesis - agent.stream() handles EXAIM internally
    collected_tokens = []
    async for token in agent.stream(synthesis_prompt):
        collected_tokens.append(token)
    
    # Collect final synthesis
    final_synthesis = "".join(collected_tokens)
    
    return {
        "final_synthesis": final_synthesis
    }
