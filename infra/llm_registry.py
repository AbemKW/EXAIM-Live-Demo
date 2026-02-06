"""LLM Registry for EXAIM Infrastructure"""
# Fix CUDA memory fragmentation before any torch imports
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True,max_split_size_mb:128"
import gc
import yaml
import logging
import warnings
from enum import Enum
from pathlib import Path
from typing import Optional, Union, Any, List, Mapping
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

logger = logging.getLogger(__name__)

class LLMRole(str, Enum):
    MAS = "mas"
    SUMMARIZER = "summarizer"
    BUFFER_AGENT = "buffer_agent"

class HuggingFacePipelineLLM(BaseChatModel):
    """LangChain-compatible wrapper for Hugging Face pipelines."""
    
    pipeline: Any = None
    model_name: str = ""
    temperature: float = 0.0
    role: str = ""
    
    class Config:
        arbitrary_types_allowed = True
    
    def __init__(self, pipeline, model_name: str = "", temperature: float = 0.0, role: str = "", **kwargs):
        super().__init__(**kwargs)
        self.pipeline = pipeline
        self.model_name = model_name
        self.temperature = temperature
        self.role = role
    
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
        hf_messages = []

        if not messages:
            return ChatResult(generations=[ChatGeneration(message=AIMessage(content=""))])

        system_instruction = ""
        start_idx = 0
        if isinstance(messages[0], SystemMessage):
            system_instruction = messages[0].content
            start_idx = 1

        for msg in messages[start_idx:]:
            if isinstance(msg, AIMessage):
                role = "assistant"
            else:
                role = "user"
            content = msg.content
            if not isinstance(content, (str, list)):
                content = str(content)
            hf_messages.append({"role": role, "content": content})

        if system_instruction:
            first_user_idx = None
            for i, msg in enumerate(hf_messages):
                if msg["role"] == "user":
                    first_user_idx = i
                    break
            
            if first_user_idx is not None:
                content = hf_messages[first_user_idx]["content"]
                if isinstance(content, str):
                    hf_messages[first_user_idx]["content"] = f"{system_instruction}\n\n{content}"
                elif isinstance(content, list):
                    content.insert(0, {"type": "text", "text": system_instruction})
            else:
                hf_messages.insert(0, {"role": "user", "content": system_instruction})

        try:
            # --- GENERATION SETTINGS ---
            # 1. Chain-of-Thought needs token breathing room (1024)
            # 2. Greedy decoding (do_sample=False) avoids "thought loops" in 4B models
            
            gen_kwargs = {
                "max_new_tokens": 1024,
                "max_length": None, 
                "return_full_text": False,
            }
            
            # Greedy decoding for Buffer/Summarizer (temp=0) is faster and safer
            if self.temperature is None or self.temperature < 1e-5:
                gen_kwargs["do_sample"] = False
                gen_kwargs["temperature"] = None 
                gen_kwargs["top_p"] = None
            else:
                gen_kwargs["do_sample"] = True
                gen_kwargs["temperature"] = self.temperature
                gen_kwargs["top_p"] = 0.95

            logger.debug(f"HF Pipeline Input (Role: {self.role}): {str(hf_messages)[:200]}...")

            result = self.pipeline(hf_messages, **gen_kwargs)
            
            text_output = ""
            if isinstance(result, list) and len(result) > 0:
                item = result[0]
                if isinstance(item, dict) and "generated_text" in item:
                    gen_text = item["generated_text"]
                    if isinstance(gen_text, list) and len(gen_text) > 0:
                        last_msg = gen_text[-1]
                        if isinstance(last_msg, dict):
                            text_output = last_msg.get("content", str(last_msg))
                        else:
                            text_output = str(last_msg)
                    elif isinstance(gen_text, str):
                        text_output = gen_text
                    else:
                        text_output = str(gen_text)
                elif isinstance(item, dict) and "text" in item:
                    text_output = item.get("text", "")
                else:
                    text_output = str(item)
            else:
                text_output = str(result)

            if not text_output or text_output.strip() == "":
                logger.error(f"HF Pipeline returned empty output for {self.role}")
                text_output = "{}"

            logger.info(f"HF Output ({self.role}): {text_output[:100]}...")
            return ChatResult(generations=[ChatGeneration(message=AIMessage(content=text_output))])

        except Exception as e:
            logger.error(f"Error in HuggingFacePipelineLLM._generate: {e}")
            return ChatResult(generations=[ChatGeneration(message=AIMessage(content=f"Error: {str(e)}"))])
        finally:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    
    async def _agenerate(self, messages: List[BaseMessage], stop: Optional[List[str]] = None, run_manager: Optional[CallbackManagerForLLMRun] = None, **kwargs: Any) -> ChatResult:
        import asyncio
        import sys
        if sys.version_info >= (3, 9):
            return await asyncio.to_thread(self._generate, messages, stop, run_manager, **kwargs)
        else:
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(None, lambda: self._generate(messages, stop, run_manager, **kwargs))
    
    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        return {"model_name": self.model_name, "temperature": self.temperature, "role": self.role}

_CONFIG_PATH = Path(__file__).parent / "model_configs.yaml"
_DEFAULT_CONFIGS = None
_HF_PIPELINE_CACHE = {}
_GPU_ASSIGNMENTS = {}

def _load_default_configs():
    global _DEFAULT_CONFIGS
    if _DEFAULT_CONFIGS is None:
        _DEFAULT_CONFIGS = {}
        if _CONFIG_PATH.exists():
            with open(_CONFIG_PATH, 'r') as f:
                _DEFAULT_CONFIGS = yaml.safe_load(f) or {}
    return _DEFAULT_CONFIGS

def _get_device_assignment(model_name: str) -> Union[str, dict]:
    global _GPU_ASSIGNMENTS
    if model_name in _GPU_ASSIGNMENTS: return _GPU_ASSIGNMENTS[model_name]
    
    if not torch.cuda.is_available():
        return "cpu"
    
    # On HF Spaces, you typically only have GPU 0.
    # We assign everything to GPU 0. If you had 2 GPUs, you could split logic here.
    device = {"": 0}
         
    _GPU_ASSIGNMENTS[model_name] = device
    return device

def _create_llm_instance(provider: str, model: Optional[str] = None, streaming: bool = True, temperature: Optional[float] = None, role: str = ""):
    provider = provider.lower()
    if provider == "huggingface":
        try:
            from transformers import pipeline as hf_pipeline, AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
        except ImportError:
            raise ImportError("transformers required")

        model_name = model or os.getenv("HUGGINGFACE_MODEL", "google/medgemma-27b-text-it")
        task = os.getenv("HUGGINGFACE_TASK", "text-generation")
        cache_key = f"{model_name}:{task}"

        if cache_key in _HF_PIPELINE_CACHE:
            pipe = _HF_PIPELINE_CACHE[cache_key]
        else:
            is_quantized = "bnb-4bit" in model_name.lower()
            model_dtype = torch.bfloat16 if "medgemma" in model_name.lower() else "auto"
            device_map = _get_device_assignment(model_name)
            
            gc.collect()
            if torch.cuda.is_available(): torch.cuda.empty_cache()

            # --- CRITICAL FIX FOR LOADING ---
            model_kwargs = {
                "device_map": device_map,
                "trust_remote_code": True,
                "low_cpu_mem_usage": True, # Prevents meta tensor errors
            }
            
            # Try to use Flash Attention 2 for 2-3x speed boost on L4/A10G GPUs
            # Only enable if GPU supports it (compute capability >= 8.0) and flash-attn is available
            if torch.cuda.is_available():
                try:
                    compute_capability = torch.cuda.get_device_capability()[0]
                    if compute_capability >= 8:
                        model_kwargs["attn_implementation"] = "flash_attention_2"
                        logger.info(f"Enabling Flash Attention 2 for faster inference on GPU with compute {compute_capability}.x")
                except Exception as e:
                    logger.info(f"Flash Attention 2 not available, using standard attention: {e}")
            
            if is_quantized:
                model_kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.bfloat16,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4"
                )
            else:
                model_kwargs["torch_dtype"] = model_dtype

            logger.info(f"Loading HF Model: {model_name} on {device_map}")
            
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model_obj = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
            
            # Fix for missing pad token
            if tokenizer.pad_token_id is None: 
                tokenizer.pad_token_id = tokenizer.eos_token_id
            
            # Explicitly resize embeddings if needed (Fixes missing lm_head in some edge cases)
            model_obj.resize_token_embeddings(len(tokenizer))

            pipe = hf_pipeline(task, model=model_obj, tokenizer=tokenizer)
            _HF_PIPELINE_CACHE[cache_key] = pipe

        return HuggingFacePipelineLLM(
            pipeline=pipe,
            model_name=model_name,
            temperature=temperature if temperature is not None else 0.0,
            role=role
        )
    
    if provider == "google": return ChatGoogleGenerativeAI(model=model or "gemini-2.5-flash-lite", google_api_key=os.getenv("GOOGLE_API_KEY"))
    if provider == "groq": return ChatGroq(model=model or "llama-3.3-70b-versatile", api_key=os.getenv("GROQ_API_KEY"))
    if provider == "openai": return ChatOpenAI(model=model, api_key=os.getenv("OPENAI_API_KEY"))
    return None

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
        if role_str in self._instances: return self._instances[role_str]
        
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
    if _registry is None: _registry = LLMRegistry()
    return _registry.get_llm(role)

def preload_models(roles=None):
    global _registry
    if _registry is None: _registry = LLMRegistry()
    defaults = _load_default_configs()
    roles = roles or list(defaults.keys())
    for role in roles:
        _registry.get_llm(role)