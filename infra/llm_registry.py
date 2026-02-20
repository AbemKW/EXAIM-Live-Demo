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

def _create_llm_instance(provider: str, model: Optional[str] = None, streaming: bool = True, temperature: Optional[float] = None, role: str = "", base_url: Optional[str] = None):
    provider = provider.lower()
    model_name = model or "medgemma-27b-text-it"

    # OpenAI provider handles vLLM or LM Studio via OpenAI-compatible API
    if provider == "openai":
        # Prioritize OPENAI_BASE_URL from environment for dynamic GPU routing
        # This ensures we use the Lambda instance IP provisioned during startup
        env_base_url = os.getenv("OPENAI_BASE_URL")
        if env_base_url:
            base_url = env_base_url
        elif base_url is None:
            # Fallback to local default if no environment variable or config is set
            base_url = "http://localhost:1234/v1"

        api_key = os.getenv("OPENAI_API_KEY", "EMPTY")

        # Model-specific kwargs for extra_body parameters
        model_kwargs = {}
        if role == LLMRole.SUMMARIZER:
            model_kwargs = {"max_tokens": 2048}
            return ChatOpenAI(
                model=model_name,
                base_url=base_url,
                api_key=api_key,
                model_kwargs=model_kwargs,
                streaming=streaming
            )
        
        if role == LLMRole.BUFFER_AGENT:
            model_kwargs = {"max_tokens": 1200}
            return ChatOpenAI(
                model=model_name,
                base_url=base_url,
                api_key=api_key,
                model_kwargs=model_kwargs,
                streaming=streaming
            )

        return ChatOpenAI(
            model=model_name,
            base_url=base_url,
            api_key=api_key,
            temperature=temperature if temperature is not None else 0.0,
            model_kwargs=model_kwargs,
            streaming=streaming
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
            role=role_str,
            base_url=config.get("base_url")
        )
        self._instances[role_str] = instance
        return instance

_registry: Optional[LLMRegistry] = None

def get_llm(role):
    global _registry
    if _registry is None:
        _registry = LLMRegistry()
    return _registry.get_llm(role)
