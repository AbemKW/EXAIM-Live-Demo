"""LLM Registry for EXAIM Infrastructure"""
import os
import yaml
import logging
from enum import Enum
from pathlib import Path
from typing import Optional, Union
from dotenv import find_dotenv, load_dotenv

load_dotenv(find_dotenv())

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI

logger = logging.getLogger(__name__)

class LLMRole(str, Enum):
    MAS = "mas"
    SUMMARIZER = "summarizer"
    BUFFER_AGENT = "buffer_agent"

_CONFIG_PATH = Path(__file__).parent / "model_configs.yaml"
_DEFAULT_CONFIGS = None

def _load_default_configs():
    global _DEFAULT_CONFIGS
    if _DEFAULT_CONFIGS is None:
        _DEFAULT_CONFIGS = {}
        if _CONFIG_PATH.exists():
            with open(_CONFIG_PATH, 'r') as f:
                _DEFAULT_CONFIGS = yaml.safe_load(f) or {}
    return _DEFAULT_CONFIGS

def _create_llm_instance(provider: str, model: Optional[str] = None, streaming: bool = True, temperature: Optional[float] = None, role: str = ""):
    provider = provider.lower()
    model_name = model or "google/medgemma-1.5-4b-it"

    # OpenAI provider now handles vLLM via OpenAI-compatible API
    if provider == "openai":
        base_url = os.getenv("OPENAI_BASE_URL", None)
        api_key = os.getenv("OPENAI_API_KEY", "EMPTY")
        
        # Model-specific kwargs for vLLM extra_body parameters
        model_kwargs = {}
        
        # Guided JSON for buffer agent (vLLM-specific)
        if role == LLMRole.BUFFER_AGENT:
            try:
                from exaim_core.schema.buffer_analysis import BufferAnalysis
                guided_json_schema = BufferAnalysis.model_json_schema()
                model_kwargs["extra_body"] = {"guided_json": guided_json_schema}
                logger.info(f"Using guided JSON for buffer agent with schema: {list(guided_json_schema.get('properties', {}).keys())}")
            except Exception as e:
                logger.warning(f"Could not generate guided JSON schema for buffer agent: {e}")
        
        # Summarizer generation parameters + guided JSON for character limit enforcement
        # Guided JSON uses vLLM's constrained decoding to enforce schema at token level
        if role == LLMRole.SUMMARIZER:
            try:
                from exaim_core.schema.agent_summary import AgentSummary
                guided_json_schema = AgentSummary.model_json_schema()
                model_kwargs["extra_body"] = {
                    "guided_json": guided_json_schema,
                    "repetition_penalty": 1.15
                }
                logger.info(f"Using guided JSON for summarizer with character limits enforced")
            except Exception as e:
                logger.warning(f"Could not generate guided JSON schema for summarizer: {e}")
                # Fallback to repetition penalty only
                extra_body = model_kwargs.get("extra_body", {})
                extra_body["repetition_penalty"] = 1.15
                model_kwargs["extra_body"] = extra_body
            
            return ChatOpenAI(
                model=model_name,
                base_url=base_url,
                api_key=api_key,
                temperature=0.0,  # Deterministic for consistent retry behavior
                top_p=0.9,        # Slightly constrain sampling
                max_tokens=800,   # Safety cap (valid responses ~250-350 tokens)
                model_kwargs=model_kwargs,
            )
        
        return ChatOpenAI(
            model=model_name,
            base_url=base_url,
            api_key=api_key,
            temperature=temperature if temperature is not None else 0.0,
            model_kwargs=model_kwargs,
        )
    
    if provider == "google":
        # Use temperature override for deterministic behavior
        actual_temp = temperature if temperature is not None else (0.0 if role in [LLMRole.SUMMARIZER, LLMRole.BUFFER_AGENT] else 0.7)
        return ChatGoogleGenerativeAI(
            model=model or "gemini-2.5-flash-lite",
            google_api_key=os.getenv("GOOGLE_API_KEY"),
            temperature=actual_temp,
        )
    
    if provider == "groq":
        return ChatGroq(
            model=model or "llama-3.3-70b-versatile",
            api_key=os.getenv("GROQ_API_KEY"),
        )
    
    raise ValueError(f"Unsupported provider: {provider}. Use 'openai', 'google', or 'groq'.")

class LLMRegistry:
    def __init__(self):
        self._instances = {}
        self._configs = {}
        self._load_configs()
    
    def _load_configs(self):
        load_dotenv(find_dotenv())
        default_configs = _load_default_configs()
        for role, default_config in default_configs.items():
            config = default_config.copy()
            self._configs[role] = config

    def get_llm(self, role: Union[str, LLMRole]):
        role_str = role.value if isinstance(role, LLMRole) else role
        if role_str in self._instances:
            return self._instances[role_str]
        
        config = self._configs[role_str]
        temperature = 0.0 if role_str in ["summarizer", "buffer_agent"] else None
        
        instance = _create_llm_instance(
            provider=config["provider"],
            model=config.get("model"),
            streaming=config.get("streaming", True),
            temperature=temperature,
            role=role_str
        )
        self._instances[role_str] = instance
        return instance

_registry: Optional[LLMRegistry] = None

def get_llm(role):
    global _registry
    if _registry is None:
        _registry = LLMRegistry()
    return _registry.get_llm(role)
