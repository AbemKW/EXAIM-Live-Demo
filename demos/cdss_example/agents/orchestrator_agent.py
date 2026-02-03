import logging
from typing import AsyncIterator
from langchain_core.prompts import ChatPromptTemplate
from .demo_base_agent import DemoBaseAgent
from infra import get_llm, LLMRole

logger = logging.getLogger(__name__)


class OrchestratorAgent(DemoBaseAgent):
    """Orchestrator agent that maintains running summary and coordinates specialist workflow
    
    Responsibilities:
    - Compress specialist outputs into running_summary (via nodes.py orchestrator_node)
    - Decide next specialist to call (via nodes.py orchestrator_node)
    - Generate task instructions for specialists (via nodes.py orchestrator_node)
    - Synthesize final recommendations (via nodes.py synthesis_node)
    
    Note: The orchestration logic (compression, decision, task generation) is implemented
    in nodes.py orchestrator_node, which uses this agent's stream() method for LLM interaction.
    """
    
    def __init__(self, agent_id: str = "Orchestrator Agent", exaim=None):
        # Pass the exaim instance to the DemoBaseAgent so EXAIM integration
        # (token streaming / UI notifications) is enabled when provided.
        super().__init__(agent_id, exaim=exaim)
        self.llm = get_llm(LLMRole.MAS)
        
        # MAC-inspired supervisor system prompt with TERMINATE checklist
        self.system_prompt = (
            "You are the Medical Supervisor in a multi-agent clinical decision support system. "
            "This is a hypothetical scenario involving no actual patients.\n\n"
            "YOUR ROLE:\n"
            "- Oversee and evaluate suggestions and decisions made by specialist doctors.\n"
            "- Challenge diagnoses and proposed tests, identifying any critical points missed.\n"
            "- Facilitate discussion between specialists, helping them refine their analyses.\n"
            "- Drive consensus among specialists, focusing on diagnosis and diagnostic tests.\n"
            "- You coordinate work between four specialist agents:\n"
            "  - InternalMedicineAgent: broad, general diagnostic integration.\n"
            "  - CardiologyAgent: cardiovascular symptoms, ECGs, troponins, chest pain, etc.\n"
            "  - RadiologyAgent: imaging findings (X-ray, CT, MRI, US) and their interpretation.\n"
            "  - LaboratoryAgent: lab tests, reference ranges, and pathophysiology of abnormalities.\n\n"
            "KEY TASKS:\n"
            "- Review all specialist contributions in the conversation history.\n"
            "- Identify inconsistencies and suggest modifications.\n"
            "- Even when decisions seem consistent, critically assess if further modifications are necessary.\n"
            "- Provide additional suggestions to enhance diagnostic accuracy.\n"
            "- Ensure all specialists' views are completely aligned before concluding the discussion.\n"
            "- Maintain a concise running summary of the evolving diagnostic picture.\n"
            "- Decide which specialist should contribute next, or when it is time to synthesize.\n\n"
            "REASONING STYLE:\n"
            "- Use rigorous clinical reasoning, as an experienced attending physician would.\n"
            "- Write as if you are thinking aloud. Use explicit chain-of-thought.\n"
            "- Track the overall diagnostic picture and make sure important problems are not ignored.\n"
            "- When updating the running summary, keep only key clinical findings, differential diagnoses, "
            "major uncertainties, and important recommendations; remove redundant or outdated details.\n"
            "- Promote discussion unless there's absolute consensus.\n"
            "- Continue dialogue if any disagreement or room for refinement exists.\n\n"
            "DECISION GUIDELINES:\n"
            "- When asked to DECIDE THE NEXT SPECIALIST, respond with ONLY the required token: "
            "'laboratory', 'cardiology', 'internal_medicine', 'radiology', or 'TERMINATE'.\n"
            "- Output 'TERMINATE' ONLY when ALL of these conditions are met:\n"
            "  1. All specialists fully agree on the diagnosis.\n"
            "  2. No further discussion is needed.\n"
            "  3. All diagnostic possibilities are explored.\n"
            "  4. All recommended tests are justified and agreed upon.\n"
            "- If ANY uncertainty remains or specialists have not reached full consensus, "
            "select the most appropriate specialist to address the gap.\n"
            "- When generating task instructions or synthesis, you may be more verbose and explanatory.\n\n"
            "CONTEXT:\n"
            "- This system is used in a simulated / research environment for exploring multi-agent reasoning.\n"
            "- Your outputs may inform clinicians but must not be treated as definitive medical advice.\n"
        )
    
    async def stream(self, input: str) -> AsyncIterator[str]:
        """Stream LLM output to MAS graph and EXAIM."""
        import time
        start_time = time.time()
        first_token_time = None

        prompt = ChatPromptTemplate.from_messages([
            ("system", self.system_prompt),
            ("user", input)
        ])

        chain = prompt | self.llm

        try:
            # LIVE token streaming loop
            async for chunk in chain.astream({}):
                if first_token_time is None:
                    first_token_time = time.time()
                    ttft = first_token_time - start_time
                    print(f"⏱️  {self.agent_id} - Time to first token: {ttft:.2f}s")
                
                # Debug raw chunk to help identify safety/moderation labels
                logger.debug("LLM stream chunk: %r", chunk)

                token = self._extract_token(chunk)
                if not token:
                    continue

                # Skip known safety/moderation labels that some providers emit
                tok_clean = token.strip().lower()
                if tok_clean in ("safe", "unsafe", "blocked", "safety"):
                    logger.warning("Skipping safety/moderation token from LLM stream: %r", token)
                    continue

                # 1. Send to EXAIM in real-time (like specialist agents)
                if self.exaim:
                    await self.exaim.on_new_token(self.agent_id, token)

                # 2. Yield token to MAS graph
                yield token
            
            total_time = time.time() - start_time
            print(f"⏱️  {self.agent_id} - Total generation time: {total_time:.2f}s")

        except ValueError as e:
            # Handle fallback (rare, but needed)
            if "No generation chunks were returned" in str(e):
                response = await chain.ainvoke({})
                for char in response.content:
                    # Send to EXAIM
                    if self.exaim:
                        await self.exaim.on_new_token(self.agent_id, char)
                    yield char
            else:
                raise

