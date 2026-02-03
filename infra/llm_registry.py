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
        
        # Convert LangChain messages to HF pipeline format
        hf_messages = []
        for msg in messages:
            if isinstance(msg, SystemMessage):
                role = "system"
            elif isinstance(msg, HumanMessage):
                role = "user"
            elif isinstance(msg, AIMessage):
                role = "assistant"
            else:
                role = "user"
            
            # Handle both text and image content
            if isinstance(msg.content, str):
                hf_messages.append({"role": role, "content": msg.content})
            elif isinstance(msg.content, list):
                # Multi-modal content
                hf_messages.append({"role": role, "content": msg.content})
            else:
                hf_messages.append({"role": role, "content": str(msg.content)})
        
        # Call the pipeline
        try:
            # Prepare generation kwargs
            gen_kwargs = {
                "return_full_text": False,  # Only return generated text
                "max_new_tokens": 1024,  # Allow longer responses for complete JSON output
            }
            
            # Add temperature and sampling if specified
            if self.temperature is not None and self.temperature > 0:
                gen_kwargs["temperature"] = self.temperature
                gen_kwargs["do_sample"] = True
            
            # Determine how to call the pipeline based on task type
            task = getattr(self.pipeline, 'task', 'text-generation')
            
            if task == 'image-text-to-text':
                # Image-text-to-text pipeline - use text parameter
                result = self.pipeline(text=hf_messages, **gen_kwargs)
            else:
                # Standard text-generation pipeline - pass messages directly
                # The pipeline expects either a string or list of chat messages
                result = self.pipeline(hf_messages, **gen_kwargs)
            
            # Log debug information
            logger.debug(f"HF Pipeline Task: {task}")
            logger.debug(f"HF Pipeline Result Type: {type(result)}")
            if isinstance(result, list) and len(result) > 0:
                logger.debug(f"HF Result[0] Type: {type(result[0])}")
                logger.debug(f"HF Result[0] Keys: {result[0].keys() if isinstance(result[0], dict) else 'N/A'}")
            
            # Extract text from result
            # For chat/conversational models, result is typically:
            # [{'generated_text': [{'role': 'user', 'content': '...'}, {'role': 'assistant', 'content': '...'}]}]
            # or [{'generated_text': 'some text'}]
            if isinstance(result, list) and len(result) > 0:
                item = result[0]
                
                if isinstance(item, dict):
                    generated = item.get('generated_text', item)
                    
                    # If generated_text is a list of messages (chat format)
                    if isinstance(generated, list) and len(generated) > 0:
                        # Find the last assistant message
                        for msg in reversed(generated):
                            if isinstance(msg, dict) and msg.get('role') == 'assistant':
                                text = msg.get('content', str(msg))
                                break
                        else:
                            # No assistant message found, use last message
                            if isinstance(generated[-1], dict):
                                text = generated[-1].get('content', str(generated[-1]))
                            else:
                                text = str(generated[-1])
                    elif isinstance(generated, str):
                        text = generated
                    else:
                        text = str(generated)
                elif isinstance(item, str):
                    text = item
                else:
                    text = str(item)
            else:
                text = str(result)
            
            logger.debug(f"HF Extracted Text (first 200 chars): {text[:200]}")
            
            message = AIMessage(content=text)
            generation = ChatGeneration(message=message)
            return ChatResult(generations=[generation])
        except Exception as e:
            logger.error(f"Error in HuggingFacePipelineLLM._generate: {e}")
            logger.exception("Full traceback:")
            warnings.warn(f"HuggingFace pipeline error: {e}")
            # Return error message
            message = AIMessage(content=f"Error: {str(e)}")
            generation = ChatGeneration(message=message)
            return ChatResult(generations=[generation])
    
    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        """Get identifying parameters."""
        return {"model_name": self.model_name, "temperature": self.temperature}


# Lazy loading: Don't load configs or instantiate registry at import time
_CONFIG_PATH = Path(__file__).parent / "model_configs.yaml"
_DEFAULT_CONFIGS = None


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
        
        # Determine task type - for text-only use, we should use text-generation
        # image-text-to-text requires image inputs
        task = os.getenv("HUGGINGFACE_TASK", "text-generation")
        if "medgemma" in model_name.lower() and task == "image-text-to-text":
            # MedGemma supports multimodal, but for text-only tasks use text-generation
            logger.warning(f"Using text-generation task for {model_name} (image-text-to-text requires images)")
            task = "text-generation"
        
        # Create pipeline with device_map for automatic GPU support
        pipe_kwargs = {
            "model": model_name,
            "device_map": "auto",
        }
        
        try:
            logger.info(f"Loading HuggingFace pipeline: task={task}, model={model_name}")
            pipe = hf_pipeline(task, **pipe_kwargs)
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

