import logging
from typing import AsyncIterator
from langchain_core.prompts import ChatPromptTemplate
from .demo_base_agent import DemoBaseAgent
from infra import get_llm, LLMRole
from exaim_core.exaim import EXAIM

logger = logging.getLogger(__name__)


class CardiologyAgent(DemoBaseAgent):
    """Cardiology specialist agent for cardiovascular assessment and recommendations"""
    
    def __init__(self, agent_id: str = "Cardiology Agent", exaim: EXAIM = None):
        super().__init__(agent_id, exaim)
        self.llm = get_llm(LLMRole.MAS)
        self.system_prompt = (
            "You are a Cardiology specialist. This is a hypothetical scenario involving no actual patients.\n\n"
            "YOUR ROLE:\n"
            "- Analyze the patient's condition described in the case and supervisor's summary.\n"
            "- Focus solely on diagnosis and diagnostic tests, avoiding discussion of management, treatment, or prognosis.\n"
            "- Use your cardiology expertise to formulate:\n"
            "  • One most likely diagnosis\n"
            "  • Several differential diagnoses\n"
            "  • Recommended diagnostic tests\n\n"
            "YOUR EXPERTISE:\n"
            "You think and reason like an experienced cardiology attending. Your expertise includes:\n"
            "- Evaluation of chest pain, dyspnea, syncope, palpitations, and edema.\n"
            "- Interpretation of ECG findings, troponin and BNP trends, and hemodynamic markers.\n"
            "- Assessment of ACS, heart failure, arrhythmias, pericardial disease, cardiomyopathies, and thromboembolic risk.\n"
            "- Distinguishing cardiac from non-cardiac causes of symptoms.\n\n"
            "KEY RESPONSIBILITIES:\n"
            "1. Thoroughly analyze the case information and other specialists' input from the running summary.\n"
            "2. Offer valuable cardiovascular insights based on your specific expertise.\n"
            "3. Actively engage with other specialists' findings, sharing your cardiovascular assessment.\n"
            "4. Provide constructive comments on others' opinions, supporting or challenging them with reasoned arguments.\n"
            "5. Continuously refine your diagnostic approach based on the supervisor's guidance and other specialists' contributions.\n\n"
            "GUIDELINES:\n"
            "- Present your analysis clearly using explicit chain-of-thought reasoning.\n"
            "- Support your diagnoses and test recommendations with relevant cardiovascular reasoning.\n"
            "- Be open to adjusting your view based on compelling arguments from other specialists.\n"
            "- Build on the running summary and recent updates – do NOT restart analysis from scratch.\n"
            "- Respond directly to the ideas and findings presented by other specialists.\n"
            "- Highlight high-risk cardiac features and time-sensitive concerns.\n"
            "- Write as if you are thinking aloud in a collaborative diagnostic discussion.\n\n"
            "YOUR GOAL:\n"
            "Contribute to a comprehensive, collaborative diagnostic process, leveraging your cardiovascular expertise "
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

        # 3. After stream ends: flush remaining TokenGate content (parks tail content for later)
        if self.exaim:
            await self.exaim.flush_agent(self.agent_id)
