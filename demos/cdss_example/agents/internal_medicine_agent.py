import logging
from typing import AsyncIterator
from langchain_core.prompts import ChatPromptTemplate
from .demo_base_agent import DemoBaseAgent
from infra import get_llm, LLMRole
from exaim_core.exaim import EXAIM

logger = logging.getLogger(__name__)

# Try to import Google GenAI error types for better error handling
try:
    from google.genai.errors import ServerError
    GOOGLE_GENAI_AVAILABLE = True
except ImportError:
    GOOGLE_GENAI_AVAILABLE = False
    ServerError = None


class InternalMedicineAgent(DemoBaseAgent):
    """Internal Medicine specialist agent for comprehensive clinical assessment"""
    
    def __init__(self, agent_id: str = "Internal Medicine Agent", exaim: EXAIM = None):
        super().__init__(agent_id, exaim)
        self.llm = get_llm(LLMRole.MAS)
        self.system_prompt = (
            "You are an Internal Medicine specialist. This is a hypothetical scenario involving no actual patients.\n\n"
            "YOUR ROLE:\n"
            "- Analyze the patient's condition described in the case and supervisor's summary.\n"
            "- Focus solely on diagnosis and diagnostic tests, avoiding discussion of management, treatment, or prognosis.\n"
            "- Use your internal medicine expertise to formulate:\n"
            "  • One most likely diagnosis\n"
            "  • Several differential diagnoses\n"
            "  • Recommended diagnostic tests\n\n"
            "YOUR EXPERTISE:\n"
            "You think and reason like an experienced internal medicine attending. Your expertise includes:\n"
            "- Comprehensive patient assessment and differential diagnosis.\n"
            "- Integration of multi-system findings.\n"
            "- Infectious diseases, endocrinology, nephrology, gastroenterology, pulmonology, and systemic syndromes.\n"
            "- Risk stratification and evidence-informed diagnostic strategies.\n\n"
            "KEY RESPONSIBILITIES:\n"
            "1. Thoroughly analyze the case information and other specialists' input from the running summary.\n"
            "2. Offer valuable insights based on your specific expertise.\n"
            "3. Actively engage with other specialists' findings, sharing your thoughts and deductions.\n"
            "4. Provide constructive comments on others' opinions, supporting or challenging them with reasoned arguments.\n"
            "5. Continuously refine your diagnostic approach based on the supervisor's guidance and other specialists' contributions.\n\n"
            "GUIDELINES:\n"
            "- Present your analysis clearly using explicit chain-of-thought reasoning.\n"
            "- Support your diagnoses and test recommendations with relevant reasoning.\n"
            "- Be open to adjusting your view based on compelling arguments from other specialists.\n"
            "- Build on the running summary and recent updates – do NOT restart analysis from scratch.\n"
            "- Respond directly to the ideas and findings presented by other specialists.\n"
            "- Write as if you are thinking aloud in a collaborative diagnostic discussion.\n\n"
            "YOUR GOAL:\n"
            "Contribute to a comprehensive, collaborative diagnostic process, leveraging your unique expertise "
            "to reach the most accurate diagnosis possible.\n"
        )
    
    async def stream(self, input: str) -> AsyncIterator[str]:
        """Stream LLM output while sending tokens live to EXAIM and UI."""

        prompt = ChatPromptTemplate.from_messages([
            ("system", self.system_prompt),
            ("user", input)
        ])

        chain = prompt | self.llm

        try:
            # LIVE token streaming loop
            async for chunk in chain.astream({}):
                token = self._extract_token(chunk)
                if not token:
                    continue

                # 1. Send to EXAIM in real-time
                if self.exaim:
                    await self.exaim.on_new_token(self.agent_id, token)

                # 2. Yield token to MAS graph
                yield token

        except ValueError as e:
            # Handle fallback (rare, but needed)
            if "No generation chunks were returned" in str(e):
                response = await chain.ainvoke({})
                for char in response.content:
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

        # 3. After stream ends: flush remaining TokenGate content (parks tail content for later)
        if self.exaim:
            await self.exaim.flush_agent(self.agent_id)
