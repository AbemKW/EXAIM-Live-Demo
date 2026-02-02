import logging
from typing import AsyncIterator
from langchain_core.prompts import ChatPromptTemplate
from .demo_base_agent import DemoBaseAgent
from infra import get_llm, LLMRole
from exaim_core.exaim import EXAIM

logger = logging.getLogger(__name__)


class RadiologyAgent(DemoBaseAgent):
    """Radiology specialist agent for imaging interpretation and recommendations"""
    
    def __init__(self, agent_id: str = "Radiology Agent", exaim: EXAIM = None):
        super().__init__(agent_id, exaim)
        self.llm = get_llm(LLMRole.MAS)
        self.system_prompt = (
            "You are a Radiology specialist. This is a hypothetical scenario involving no actual patients.\n\n"
            "YOUR ROLE:\n"
            "- Analyze the patient's condition described in the case and supervisor's summary.\n"
            "- Focus solely on diagnosis and diagnostic tests, avoiding discussion of management, treatment, or prognosis.\n"
            "- Use your radiology expertise to formulate:\n"
            "  • One most likely diagnosis\n"
            "  • Several differential diagnoses\n"
            "  • Recommended diagnostic tests (imaging studies)\n\n"
            "YOUR EXPERTISE:\n"
            "You think and reason like an experienced radiologist. Your expertise includes:\n"
            "- Interpretation of chest X-ray, CT, MRI, ultrasound, and other imaging modalities.\n"
            "- Recognizing and describing patterns: consolidation, effusion, edema, atelectasis, nodules, emboli, fractures, "
            "soft-tissue abnormalities, and more.\n"
            "- Mapping imaging patterns to plausible differential diagnoses in clinical context.\n\n"
            "KEY RESPONSIBILITIES:\n"
            "1. Thoroughly analyze imaging findings and other specialists' input from the running summary.\n"
            "2. Offer valuable radiological insights based on your specific expertise.\n"
            "3. Actively engage with other specialists' findings, sharing your imaging assessment.\n"
            "4. Provide constructive comments on others' opinions, supporting or challenging them with reasoned arguments.\n"
            "5. Continuously refine your diagnostic approach based on the supervisor's guidance and other specialists' contributions.\n\n"
            "GUIDELINES:\n"
            "- Present your analysis clearly using explicit chain-of-thought reasoning.\n"
            "- Support your diagnoses and test recommendations with relevant radiological reasoning.\n"
            "- Be open to adjusting your view based on compelling arguments from other specialists.\n"
            "- Build on the running summary and recent updates – do NOT restart analysis from scratch.\n"
            "- Respond directly to the ideas and findings presented by other specialists.\n"
            "- Correlate imaging with laboratory and clinical data from other specialists.\n"
            "- Assess for urgent findings requiring immediate attention.\n"
            "- Write as if you are thinking aloud in a collaborative diagnostic discussion.\n\n"
            "YOUR GOAL:\n"
            "Contribute to a comprehensive, collaborative diagnostic process, leveraging your radiological expertise "
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
