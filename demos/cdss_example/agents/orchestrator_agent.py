import logging
from typing import AsyncIterator
from langchain_core.prompts import ChatPromptTemplate
from .demo_base_agent import DemoBaseAgent
from infra import get_llm, LLMRole

logger = logging.getLogger(__name__)

# Try to import Google GenAI error types for better error handling
try:
    from google.genai.errors import ServerError
    GOOGLE_GENAI_AVAILABLE = True
except ImportError:
    GOOGLE_GENAI_AVAILABLE = False
    ServerError = None


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

        # Simplified supervisor system prompt
        self.system_prompt = (
            "You are the Medical Supervisor. This is a hypothetical scenario involving no actual patients.\n\n"
            "Your role:\n"
            "    1. Oversee and coordinate a team of specialists (Cardiology, Radiology, Laboratory, Internal Medicine).\n"
            "    2. Focus on driving a collaborative diagnostic process to reach a consensus.\n"
            "    3. Maintain a concise running summary and decide the next course of action.\n\n"
            "Key responsibilities:\n"
            "    1. Critically evaluate specialist findings and identify missed points or inconsistencies.\n"
            "    2. Facilitate discussion by providing task instructions and challenging specialists.\n"
            "    3. Ensure all diagnostic possibilities are explored and tests are justified.\n"
            "    4. Continuously refine the overall diagnostic picture based on the ongoing discussion.\n"
            "    5. Decide the next specialist to call or when to synthesize and TERMINATE.\n\n"
            "Decision Guidelines:\n"
            "    - When asked to DECIDE THE NEXT SPECIALIST, respond with ONLY: 'laboratory', 'cardiology', 'internal_medicine', 'radiology', or 'TERMINATE'.\n"
            "    - Output 'TERMINATE' ONLY when all specialists fully agree, all possibilities are explored, and no further discussion is needed.\n\n"
            "Guidelines:\n"
            "    - Present your evaluations and instructions clearly and concisely.\n"
            "    - Use rigorous clinical reasoning and maintain high standards for consensus.\n"
            "    - When updating summaries, keep only key findings and recommendations.\n\n"
            "Your goal: Lead the team to a comprehensive and accurate diagnostic conclusion through expert coordination."
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
                    print(f"â±ï¸  {self.agent_id} - Time to first token: {ttft:.2f}s")

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
            print(f"â±ï¸  {self.agent_id} - Total generation time: {total_time:.2f}s")

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
        except TypeError as e:
            # Handle LangChain bug when Google GenAI returns 503 error
            # LangChain tries to subscript ClientResponse object which fails
            if "'ClientResponse' object is not subscriptable" in str(e):
                error_msg = (
                    "The model service is currently overloaded (503 error). "
                    "This is a temporary issue with the Google GenAI API. "
                    "Please try again in a few moments."
                )
                logger.error(f"Google GenAI 503 error handled: {error_msg}")
                raise RuntimeError(error_msg) from e
            raise
        except Exception as e:
            # Catch Google GenAI ServerError before it reaches LangChain's buggy error handler
            if GOOGLE_GENAI_AVAILABLE and isinstance(e, ServerError):
                if e.status_code == 503:
                    error_msg = (
                        "The model service is currently overloaded (503 error). "
                        "This is a temporary issue with the Google GenAI API. "
                        "Please try again in a few moments."
                    )
                    logger.error(f"Google GenAI 503 error: {error_msg}")
                    raise RuntimeError(error_msg) from e
            # Re-raise other exceptions
            raise
