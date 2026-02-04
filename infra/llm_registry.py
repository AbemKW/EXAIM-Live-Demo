"""LLM Registry for EXAIM Infrastructure

Centralized LLM management with role-based configuration.
Supports YAML configuration with environment variable overrides.
"""
import os
import yaml
import logging
import warnings
from enum import Enum
from pathlib import Path
from typing import Optional, Union
from dotenv import find_dotenv, load_dotenv
load_dotenv(find_dotenv())
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage, AIMessage, HumanMessage, SystemMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.callbacks.manager import CallbackManagerForLLMRun
from typing import Any, List, Mapping, Optional

logger = logging.getLogger(__name__)

class LLMRole(str, Enum):
    """Enumeration of LLM roles for type-safe role specification."""
    MAS = "mas"
    SUMMARIZER = "summarizer"
    BUFFER_AGENT = "buffer_agent"


class HuggingFacePipelineLLM(BaseChatModel):
    """LangChain-compatible wrapper for Hugging Face pipelines."""
    
    pipeline: Any = None
    model_name: str = ""
    temperature: float = 0.0
    
    class Config:
        arbitrary_types_allowed = True
    
    def __init__(self, pipeline, model_name: str = "", temperature: float = 0.0, **kwargs):
        super().__init__(**kwargs)
        self.pipeline = pipeline
        self.model_name = model_name
        self.temperature = temperature
    
    @property
    def _llm_type(self) -> str:
        return "huggingface_pipeline"
    
    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        """Generate response using Hugging Face pipeline."""
        hf_messages = []
        system_instruction = ""

        # Normalize input messages list
        if not messages:
            return ChatResult(generations=[ChatGeneration(message=AIMessage(content=""))])

        # If first message is SystemMessage, capture and skip it
        if isinstance(messages[0], SystemMessage):
            system_instruction = messages[0].content
            remaining = messages[1:]
        else:
            remaining = messages

        for i, msg in enumerate(remaining):
            # Map LangChain roles to HF chat roles expected by Gemma
            if isinstance(msg, AIMessage):
                role = "assistant"
            else:
                # Treat HumanMessage and any unknown as user
                role = "user"

            content = msg.content

            # Prepend system instruction to the first user message
            if i == 0 and role == "user" and system_instruction:
                if isinstance(content, str):
                    content = f"{system_instruction}\n\n{content}"
                elif isinstance(content, list):
                    # For multimodal, insert a text block at start
                    content.insert(0, {"type": "text", "text": system_instruction})

            # Ensure content is in a HF-serializable form
            if not isinstance(content, (str, list)):
                content = str(content)

            hf_messages.append({"role": role, "content": content})

        # Edge case: only system instruction present
        if system_instruction and not hf_messages:
            hf_messages.append({"role": "user", "content": system_instruction})

        # --- CORRECTION 2: Pipeline Invocation and Extraction ---
        try:
            gen_kwargs = {
                "max_new_tokens": 2048,
            }

            if self.temperature is not None and self.temperature > 0:
                gen_kwargs["temperature"] = self.temperature
                gen_kwargs["do_sample"] = True
            else:
                gen_kwargs["do_sample"] = False

            # Call the pipeline. The 'image-text-to-text' pipeline accepts
            # a list of chat dicts for Gemma-style models.
            result = self.pipeline(hf_messages, **gen_kwargs)

            # Robust extraction of generated text
            text_output = ""
            if isinstance(result, list) and len(result) > 0:
                item = result[0]
                if isinstance(item, dict) and "generated_text" in item:
                    gen_text = item["generated_text"]
                    if isinstance(gen_text, list) and len(gen_text) > 0:
                        # Try to get last assistant message
                        for msg in reversed(gen_text):
                            if isinstance(msg, dict) and msg.get("role") in ("assistant", "model"):
                                text_output = msg.get("content", str(msg))
                                break
                        else:
                            # Fallback to last element
                            last = gen_text[-1]
                            text_output = last.get("content", str(last)) if isinstance(last, dict) else str(last)
                    else:
                        # generated_text is a string
                        text_output = gen_text if isinstance(gen_text, str) else str(gen_text)
                elif isinstance(item, dict) and "text" in item:
                    text_output = item.get("text", "")
                else:
                    text_output = str(item)
            else:
                text_output = str(result)

            message = AIMessage(content=text_output)
            return ChatResult(generations=[ChatGeneration(message=message)])
        except Exception as e:
            logger.error(f"Error in HuggingFacePipelineLLM._generate: {e}")
            logger.exception("Full traceback:")
            warnings.warn(f"HuggingFace pipeline error: {e}")
            return ChatResult(generations=[ChatGeneration(message=AIMessage(content=f"Error: {str(e)}"))])
    
    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        """Get identifying parameters."""
        return {"model_name": self.model_name, "temperature": self.temperature}


# Lazy loading: Don't load configs or instantiate registry at import time
_CONFIG_PATH = Path(__file__).parent / "model_configs.yaml"
_DEFAULT_CONFIGS = None
# Cache for HuggingFace pipelines to prevent double loading (OOM)
_HF_PIPELINE_CACHE = {}


def _load_default_configs():
    """Lazy load default configs from YAML."""
    global _DEFAULT_CONFIGS
    if _DEFAULT_CONFIGS is None:
        _DEFAULT_CONFIGS = {}
        if _CONFIG_PATH.exists():
            with open(_CONFIG_PATH, 'r') as f:
                _DEFAULT_CONFIGS = yaml.safe_load(f) or {}
    return _DEFAULT_CONFIGS


def _create_llm_instance(provider: str, model: Optional[str] = None, streaming: bool = True, temperature: Optional[float] = None):
    """Factory function to create LLM instances based on provider type.
    
    Args:
        provider: LLM provider name (google, groq, openai)
        model: Model name to use (overrides environment defaults)
        streaming: Whether to enable streaming
        temperature: Temperature parameter for LLM (None uses provider default)
        
    Returns:
        Configured LLM instance
        
    Environment variables:
        For Google (Gemini):
            - GOOGLE_API_KEY: Google API key (required)
            - GOOGLE_MODEL_NAME: Default model name (default: gemini-2.5-flash-lite)
            
        For Groq:
            - GROQ_API_KEY: Groq API key
            - GROQ_MODEL: Default model name (default: llama-3.3-70b-versatile)
            
        For OpenAI (or OpenAI-compatible):
            - OPENAI_API_KEY: API key
            - OPENAI_BASE_URL: Base URL for API (optional, for compatible endpoints)
            - OPENAI_MODEL: Default model name (optional)
        For Ollama (self-hosted Ollama HTTP API):
            - OLLAMA_BASE_URL: Base URL for Ollama endpoint (e.g. https://...)
            - OLLAMA_MODEL: Default model name (optional)
            
        For HuggingFace (local transformers pipeline):
            - HUGGINGFACE_MODEL: Model name (default: google/medgemma-1.5-4b-it)
            - HUGGINGFACE_TASK: Pipeline task (default: text-generation, auto-set to image-text-to-text for medgemma)
    """
    provider = provider.lower()
    
    if provider == "google":
        kwargs = {
            "model": model or os.getenv("GOOGLE_MODEL_NAME", "gemini-2.5-flash-lite"),
            "google_api_key": os.getenv("GOOGLE_API_KEY"),
            "streaming": streaming,
            "timeout": 30,  # 30 second timeout
            "max_retries": 2  # Reduce retries for faster failures
        }
        if temperature is not None:
            kwargs["temperature"] = temperature
        return ChatGoogleGenerativeAI(**kwargs)
    elif provider == "groq":
        kwargs = {
            "api_key": os.getenv("GROQ_API_KEY"),
            "model": model or os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile"),
            "streaming": streaming
        }
        if temperature is not None:
            kwargs["temperature"] = temperature
        return ChatGroq(**kwargs)
    elif provider == "openai":
        kwargs = {
            "api_key": os.getenv("OPENAI_API_KEY", "NONE"),
            "streaming": streaming
        }
        if base_url := os.getenv("OPENAI_BASE_URL"):
            kwargs["base_url"] = base_url
        if model:
            kwargs["model"] = model
        elif openai_model := os.getenv("OPENAI_MODEL"):
            kwargs["model"] = openai_model
        if temperature is not None:
            kwargs["temperature"] = temperature
        return ChatOpenAI(**kwargs)
    elif provider == "ollama":
        # Use ChatOpenAI as an OpenAI-compatible client for Ollama-compatible endpoints
        # Note: Ollama's HTTP API can be exposed via an OpenAI-compatible gateway.
        kwargs = {
            "api_key": os.getenv("OLLAMA_API_KEY", "NONE"),
            "streaming": streaming
        }
        # Allow explicit base URL for Ollama instances
        if base_url := os.getenv("OLLAMA_BASE_URL"):
            kwargs["base_url"] = base_url
        # Model selection: prefer explicit model argument, then env var
        if model:
            kwargs["model"] = model
        elif ollama_model := os.getenv("OLLAMA_MODEL"):
            kwargs["model"] = ollama_model
        if temperature is not None:
            kwargs["temperature"] = temperature
        return ChatOpenAI(**kwargs)
    elif provider == "huggingface":
        # Use Hugging Face transformers pipeline
        try:
            from transformers import pipeline as hf_pipeline
        except ImportError:
            raise ImportError(
                "transformers package is required for huggingface provider. "
                "Install with: pip install transformers torch"
            )
        
        # Model selection: prefer explicit model argument, then env var, default to medgemma
        model_name = model or os.getenv("HUGGINGFACE_MODEL", "google/medgemma-1.5-4b-it")
        
        # MedGemma / Gemma-3 architecture is multimodal and requires
        # the 'image-text-to-text' task even for text-only inputs.
        task = os.getenv("HUGGINGFACE_TASK", "image-text-to-text")
        
        # Check cache explicitly
        cache_key = f"{model_name}:{task}"
        
        if cache_key in _HF_PIPELINE_CACHE:
            logger.info(f"Using cached HuggingFace pipeline for: {cache_key}")
            pipe = _HF_PIPELINE_CACHE[cache_key]
        else:
            # Create pipeline with device_map for automatic GPU support
            pipe_kwargs = {
                "model": model_name,
                "device_map": "auto",
                "torch_dtype": "auto",
                "trust_remote_code": True,
            }
            
            try:
                logger.info(f"Loading HuggingFace pipeline: task={task}, model={model_name}")
                pipe = hf_pipeline(task, **pipe_kwargs)
                _HF_PIPELINE_CACHE[cache_key] = pipe
                logger.info("Successfully loaded HuggingFace pipeline")
            except Exception as e:
                logger.error(f"Error loading HuggingFace pipeline: {e}")
                raise
        
        return HuggingFacePipelineLLM(
            pipeline=pipe,
            model_name=model_name,
            temperature=temperature if temperature is not None else 0.0
        )
    else:
        raise ValueError(
            f"Unknown LLM provider: {provider}. "
            f"Supported providers: google, groq, openai, ollama, huggingface"
        )


class LLMRegistry:
    """Registry for managing LLM instances by role."""
    
    def __init__(self):
        self._instances = {}
        self._configs = {}
        self._load_configs()
    
    def _load_configs(self):
        """Load configurations from YAML and environment variables."""
        # Search for a .env file in parent directories (works when working_dir=/app/evals)
        load_dotenv(find_dotenv())  # Load env vars when registry is first created
        default_configs = _load_default_configs()
        for role, default_config in default_configs.items():
            # Start with YAML defaults
            config = default_config.copy()
            
            # Override with environment variables if present
            provider_env = f"{role.upper()}_LLM_PROVIDER"
            model_env = f"{role.upper()}_LLM_MODEL"
            
            if provider := os.getenv(provider_env):
                config["provider"] = provider
            if model := os.getenv(model_env):
                config["model"] = model
            
            self._configs[role] = config
    
    def get_llm(self, role: Union[str, LLMRole]):
        """Get or create an LLM instance for the specified role.
        
        Args:
            role: Role name (mas, summarizer, buffer_agent) or LLMRole enum
            
        Returns:
            Configured LLM instance
            
        Raises:
            ValueError: If role is not configured
        """
        # Convert enum to string if needed
        role_str = role.value if isinstance(role, LLMRole) else role
        
        if role_str not in self._configs:
            raise ValueError(
                f"Unknown LLM role: {role}. "
                f"Available roles: {list(self._configs.keys())}"
            )
        
        # Return cached instance if available
        if role_str in self._instances:
            return self._instances[role_str]
        
        # Create new instance
        config = self._configs[role_str]
        
        # Set temperature to 0 for summarizer and buffer_agent
        temperature = None
        if role_str in ["summarizer", "buffer_agent"]:
            temperature = 0.0
        
        instance = _create_llm_instance(
            provider=config["provider"],
            model=config.get("model"),
            streaming=config.get("streaming", True),
            temperature=temperature
        )
        
        # Cache and return
        self._instances[role_str] = instance
        return instance


# Lazy registry: Don't instantiate at import time
_registry: Optional[LLMRegistry] = None


def get_llm(role: Union[str, LLMRole]):
    """Convenience function to get an LLM instance for a role.
    
    Args:
        role: Role name (mas, summarizer, buffer_agent) or LLMRole enum
        
    Returns:
        Configured LLM instance
    """
    global _registry
    if _registry is None:
        _registry = LLMRegistry()
    return _registry.get_llm(role)

