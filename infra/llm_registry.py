"""LLM Registry for EXAIM Infrastructure

Centralized LLM management with role-based configuration.
Supports YAML configuration with environment variable overrides.
"""
# Fix CUDA memory fragmentation before any torch imports
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True,max_split_size_mb:128"
import gc
import yaml
import logging
import warnings
from enum import Enum
from pathlib import Path
from typing import Optional, Union
from dotenv import find_dotenv, load_dotenv
load_dotenv(find_dotenv())
import torch
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

        # Normalize input messages list
        if not messages:
            return ChatResult(generations=[ChatGeneration(message=AIMessage(content=""))])

        # Extract system message if present (always prepend to first user message)
        # MedGemma models can produce empty outputs if a 'system' role is passed
        system_instruction = ""
        start_idx = 0
        if isinstance(messages[0], SystemMessage):
            system_instruction = messages[0].content
            start_idx = 1

        # Convert messages to HF format (only 'user' and 'assistant' roles)
        for msg in messages[start_idx:]:
            # Map LangChain roles to HF chat roles
            if isinstance(msg, AIMessage):
                role = "assistant"
            else:
                # Treat HumanMessage, SystemMessage (if any remain), and unknown as user
                role = "user"

            content = msg.content

            # Ensure content is in a HF-serializable form
            if not isinstance(content, (str, list)):
                content = str(content)

            hf_messages.append({"role": role, "content": content})

        # Always prepend system instruction to first user message (safe fallback)
        if system_instruction:
            # Find the first user message
            first_user_idx = None
            for i, msg in enumerate(hf_messages):
                if msg["role"] == "user":
                    first_user_idx = i
                    break
            
            if first_user_idx is not None:
                # Prepend system instruction to first user message
                content = hf_messages[first_user_idx]["content"]
                if isinstance(content, str):
                    hf_messages[first_user_idx]["content"] = f"{system_instruction}\n\n{content}"
                elif isinstance(content, list):
                    content.insert(0, {"type": "text", "text": system_instruction})
            else:
                # No user message found, add system as a user message at the start
                hf_messages.insert(0, {"role": "user", "content": system_instruction})

        # --- Pipeline Invocation following official MedGemma docs ---
        try:
            # Build generation kwargs - use max_new_tokens, avoid conflicts with generation_config
            gen_kwargs = {
                "max_new_tokens": 1024,
                "max_length": 8192,  # Set high max_length to override restrictive default (20)
                "return_full_text": False,  # Only return new tokens, not the input
            }
            
            # Only add temperature/sampling if temperature > 0
            if self.temperature is not None and self.temperature > 0:
                gen_kwargs["do_sample"] = True
                gen_kwargs["temperature"] = self.temperature
            else:
                # Use subtle sampling for "deterministic" output instead of greedy decoding
                # Greedy decoding (do_sample=False) can sometimes cause model loops/collapse
                gen_kwargs["do_sample"] = True
                gen_kwargs["temperature"] = 0.1
                gen_kwargs["top_p"] = 0.95

            # Log the input for debugging (truncated)
            input_debug = str(hf_messages)
            logger.info(f"HF Pipeline Input (full): {input_debug}") # Changed to INFO and full


            # Call pipeline - text-generation pipeline expects messages directly
            # Note: The official docs use pipe(text=messages, ...) but the actual
            # transformers text-generation pipeline expects it as first positional arg
            result = self.pipeline(hf_messages, **gen_kwargs)
            
            # Log raw result type/len
            logger.debug(f"HF Pipeline Result Type: {type(result)}")
            if isinstance(result, list) and len(result) > 0:
                logger.debug(f"HF Pipeline Result[0] keys: {result[0].keys() if isinstance(result[0], dict) else 'not a dict'}")

            # Extract generated text following official docs pattern
            # output[0]["generated_text"][-1]["content"]
            text_output = ""
            if isinstance(result, list) and len(result) > 0:
                item = result[0]
                if isinstance(item, dict) and "generated_text" in item:
                    gen_text = item["generated_text"]
                    if isinstance(gen_text, list) and len(gen_text) > 0:
                        # Get the last message (assistant's response)
                        # Skip input messages and get only the new generated content
                        last_msg = gen_text[-1]
                        if isinstance(last_msg, dict):
                            text_output = last_msg.get("content", str(last_msg))
                        else:
                            text_output = str(last_msg)
                    elif isinstance(gen_text, str):
                        # Fallback: generated_text is a string
                        text_output = gen_text
                    else:
                        text_output = str(gen_text)
                elif isinstance(item, dict) and "text" in item:
                    text_output = item.get("text", "")
                else:
                    text_output = str(item)
            else:
                text_output = str(result)

            # Ensure we have actual content
            if not text_output or text_output.strip() == "":
                logger.error(f"HF Pipeline returned empty output. Full result structure: {result}")
                raise ValueError("HuggingFace pipeline generated empty text. This may indicate a model configuration issue.")

            # Log the extracted output for debugging
            output_preview = text_output[:500] if len(text_output) > 500 else text_output
            logger.info(f"HF Pipeline Generated Text Preview: {output_preview}")

            message = AIMessage(content=text_output)
            return ChatResult(generations=[ChatGeneration(message=message)])
        except Exception as e:
            logger.error(f"Error in HuggingFacePipelineLLM._generate: {e}")
            logger.exception("Full traceback:")
            warnings.warn(f"HuggingFace pipeline error: {e}")
            return ChatResult(generations=[ChatGeneration(message=AIMessage(content=f"Error: {str(e)}"))])
        finally:
            # Aggressive memory cleanup after inference to prevent fragmentation
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    
    async def _agenerate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        """Async generation - offloads blocking pipeline to thread pool.
        
        This prevents the synchronous HuggingFace pipeline from blocking
        the async event loop when used in async contexts (FastAPI, Gradio, etc).
        """
        import asyncio
        import sys
        
        # Python 3.9+ has asyncio.to_thread, older versions need run_in_executor
        if sys.version_info >= (3, 9):
            result = await asyncio.to_thread(
                self._generate, messages, stop, run_manager, **kwargs
            )
        else:
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None,  # Use default ThreadPoolExecutor
                lambda: self._generate(messages, stop, run_manager, **kwargs)
            )
        return result
    
    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        """Get identifying parameters."""
        return {"model_name": self.model_name, "temperature": self.temperature}


# Lazy loading: Don't load configs or instantiate registry at import time
_CONFIG_PATH = Path(__file__).parent / "model_configs.yaml"
_DEFAULT_CONFIGS = None
# Cache for HuggingFace pipelines to prevent double loading (OOM)
_HF_PIPELINE_CACHE = {}
# GPU allocation tracking - maps model names to assigned GPU devices
_GPU_ASSIGNMENTS = {}


def _load_default_configs():
    """Lazy load default configs from YAML."""
    global _DEFAULT_CONFIGS
    if _DEFAULT_CONFIGS is None:
        _DEFAULT_CONFIGS = {}
        if _CONFIG_PATH.exists():
            with open(_CONFIG_PATH, 'r') as f:
                _DEFAULT_CONFIGS = yaml.safe_load(f) or {}
    return _DEFAULT_CONFIGS


def _get_device_assignment(model_name: str) -> Union[str, dict]:
    """Determine GPU device assignment for a model to distribute load across multiple GPUs.
    
    Strategy:
    - Multi-GPU (2+): Distribute 27B to cuda:1, 4B to cuda:0
    - Single GPU: Use 'auto' with CPU offloading for large models (27B)
    - No GPU: Use CPU
    
    Args:
        model_name: Name of the model being loaded
        
    Returns:
        Device map (string or dict) for transformers device_map parameter
    """
    global _GPU_ASSIGNMENTS
    
    # Check if already assigned
    if model_name in _GPU_ASSIGNMENTS:
        logger.info(f"Using cached GPU assignment for {model_name}: {_GPU_ASSIGNMENTS[model_name]}")
        return _GPU_ASSIGNMENTS[model_name]
    
    # Check GPU availability
    if not torch.cuda.is_available():
        logger.warning("CUDA not available, using CPU")
        _GPU_ASSIGNMENTS[model_name] = "cpu"
        return "cpu"
    
    num_gpus = torch.cuda.device_count()
    logger.info(f"Detected {num_gpus} CUDA device(s)")
    
    # Get available GPU memory
    try:
        gpu_mem_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        logger.info(f"GPU 0 total memory: {gpu_mem_gb:.2f} GB")
    except:
        gpu_mem_gb = 0
    
    # Single GPU: Use strict isolation on GPU 0
    if num_gpus < 2:
        logger.info("Single GPU detected, using strict isolation on GPU 0")
        device = {"": 0}  # Strict isolation - entire model on GPU 0
        _GPU_ASSIGNMENTS[model_name] = device
        return device
    
    # Strict GPU isolation strategy for 2x A10G setup
    # Force each model to a single GPU to prevent fragmentation from multi-GPU splits
    # 27B models: Force entirely onto GPU 0
    # 4B models: Force entirely onto GPU 1
    if "27b" in model_name.lower():
        device = {"":  0}  # Strict isolation - entire model on GPU 0
        logger.info(f"Assigning 27B model to GPU 0 with strict isolation (device_map={{\"\":  0}})")
    elif "4b" in model_name.lower() or "1.5" in model_name.lower():
        device = {"":  1}  # Strict isolation - entire model on GPU 1
        logger.info(f"Assigning 4B model to GPU 1 with strict isolation (device_map={{\"\":  1}})")
    else:
        # Unknown size, use GPU 0 with strict isolation
        device = {"":  0}
        logger.info(f"Unknown model size, assigning to GPU 0 with strict isolation")
    
    _GPU_ASSIGNMENTS[model_name] = device
    return device


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
        
        # Model selection: prefer explicit model argument, then env var, default to medgemma 27B text-only
        model_name = model or os.getenv("HUGGINGFACE_MODEL", "google/medgemma-27b-text-it")
        
        # MedGemma 27B is text-only, uses 'text-generation' task
        # TODO: Add multimodal model detection in the future if needed
        task = os.getenv("HUGGINGFACE_TASK", "text-generation")
        
        # Check cache explicitly
        cache_key = f"{model_name}:{task}"
        
        if cache_key in _HF_PIPELINE_CACHE:
            logger.info(f"Using cached HuggingFace pipeline for: {cache_key}")
            pipe = _HF_PIPELINE_CACHE[cache_key]
        else:
            # Create pipeline with device_map for automatic GPU support
            from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig, BitsAndBytesConfig
            
            # Detect if this is a quantized model (bnb-4bit suffix)
            is_quantized = "bnb-4bit" in model_name.lower()
            
            # Set dtype - use bfloat16 for 27B models, auto for others (unless quantized)
            if is_quantized:
                # Quantized models handle dtype internally via BitsAndBytesConfig
                model_dtype = None
            else:
                model_dtype = torch.bfloat16 if "medgemma-27b" in model_name.lower() else "auto"
            
            # Determine GPU assignment for multi-GPU load distribution
            device_map = _get_device_assignment(model_name)
            
            # Aggressive memory cleanup before loading model
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            logger.info("Performed garbage collection and CUDA cache cleanup before model load")
            
            try:
                logger.info(f"Loading HuggingFace model: {model_name} (quantized={is_quantized})")
                
                # Strict Manual GPU Assignment: 27B->GPU0, 4B->GPU1
                # Both models fit entirely on a single A10G (24GB)
                is_large_model = "27b" in model_name.lower()
                is_small_model = "4b" in model_name.lower() or "1.5" in model_name.lower()
                
                if is_large_model:
                    # 27B model: Assign entirely to GPU 0
                    logger.info("Assigning 27B model entirely to GPU 0 (strict manual assignment)")
                    model_kwargs = {
                        "device_map": {"":  0},  # Strict: entire model on GPU 0
                        "max_memory": {0: "22GiB", 1: "22GiB", "cpu": "60GiB"},  # Safety guardrails
                        "trust_remote_code": True,
                        "low_cpu_mem_usage": True,  # Works correctly with explicit device_map
                    }
                elif is_small_model:
                    # 4B model: Assign entirely to GPU 1
                    logger.info("Assigning 4B model entirely to GPU 1 (strict manual assignment)")
                    model_kwargs = {
                        "device_map": {"":  1},  # Strict: entire model on GPU 1
                        "trust_remote_code": True,
                        "low_cpu_mem_usage": True,  # Works correctly with explicit device_map
                    }
                else:
                    # Fallback for unknown models: assign to GPU 0
                    logger.warning(f"Unknown model size, assigning to GPU 0 with strict isolation")
                    model_kwargs = {
                        "device_map": {"":  0},
                        "trust_remote_code": True,
                        "low_cpu_mem_usage": True,
                    }
                
                # Add quantization config if using a bnb-4bit model
                if is_quantized:
                    logger.info("Using 4-bit quantization with bitsandbytes")
                    quantization_config = BitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_compute_dtype=torch.bfloat16,
                        bnb_4bit_use_double_quant=True,
                        bnb_4bit_quant_type="nf4"
                    )
                    model_kwargs["quantization_config"] = quantization_config
                else:
                    # Only set torch_dtype for non-quantized models
                    model_kwargs["torch_dtype"] = model_dtype
                
                # Load tokenizer and model separately to configure generation properly
                tokenizer = AutoTokenizer.from_pretrained(model_name)
                model_obj = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    **model_kwargs
                )
                
                # Don't set generation_config here - we'll pass all params via kwargs
                # This avoids conflicts between model config and runtime kwargs
                # Just ensure essential token IDs are set if not already
                if tokenizer.pad_token_id is None:
                    tokenizer.pad_token_id = tokenizer.eos_token_id
                
                # Create pipeline with configured model
                pipe = hf_pipeline(
                    task,
                    model=model_obj,
                    tokenizer=tokenizer,
                )
                
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


def preload_models(roles: Optional[List[str]] = None):
    """Preload models for specified roles or all configured roles.
    
    This is useful for initializing heavy models (like HuggingFace pipelines)
    at startup rather than on first request.
    
    Args:
        roles: List of role names to preload. If None, preloads all.
    """
    global _registry
    if _registry is None:
        _registry = LLMRegistry()
    
    if roles is None:
        # Load roles directly from default config schema (YAML source of truth)
        # This ensures we preload exactly what is defined in our configuration
        defaults = _load_default_configs()
        roles = list(defaults.keys())
        
    logger.info(f"🔥 WARMUP STARTING: Preloading models for roles: {roles}")
    print(f"🔥 WARMUP STARTING: Preloading models for roles: {roles}")  # Console output
    
    for role in roles:
        try:
            logger.info(f"Loading model for role: {role}")
            print(f"  ⏳ Loading model for role: {role}...")
            _registry.get_llm(role)
            logger.info(f"Successfully loaded model for role: {role}")
            print(f"  ✅ Successfully loaded model for role: {role}")
        except Exception as e:
            logger.error(f"Failed to preload model for role {role}: {e}")
            print(f"  ❌ Failed to preload model for role {role}: {e}")
            # Don't raise, try to continue loading other models
    
    logger.info("🎉 WARMUP COMPLETE: All models preloaded")
    print("🎉 WARMUP COMPLETE: All models preloaded")


