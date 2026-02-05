#!/usr/bin/env python3
"""Test script to verify GPU distribution across multiple devices.

This script demonstrates that:
1. The 4B summarizer model loads on cuda:0 (first GPU)
2. The 27B buffer_agent model loads on cuda:1 (second GPU)
3. Memory is properly distributed across both GPUs
"""
import logging
import torch
from infra.llm_registry import get_llm, LLMRole

# Configure logging to see device assignments
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

def check_gpu_availability():
    """Check CUDA availability and device information."""
    logger.info("=" * 80)
    logger.info("GPU AVAILABILITY CHECK")
    logger.info("=" * 80)
    
    if not torch.cuda.is_available():
        logger.error("CUDA is not available! Models will load on CPU.")
        return False
    
    num_gpus = torch.cuda.device_count()
    logger.info(f"✓ CUDA is available")
    logger.info(f"✓ Number of GPUs: {num_gpus}")
    
    for i in range(num_gpus):
        props = torch.cuda.get_device_properties(i)
        total_mem = props.total_memory / (1024**3)  # Convert to GB
        logger.info(f"  GPU {i}: {props.name} - {total_mem:.1f} GB")
    
    if num_gpus < 2:
        logger.warning("⚠ Only 1 GPU detected. Multi-GPU distribution will not occur.")
    
    logger.info("")
    return num_gpus >= 2

def show_gpu_memory_usage():
    """Display current GPU memory usage."""
    if not torch.cuda.is_available():
        return
    
    logger.info("=" * 80)
    logger.info("GPU MEMORY USAGE")
    logger.info("=" * 80)
    
    for i in range(torch.cuda.device_count()):
        allocated = torch.cuda.memory_allocated(i) / (1024**3)  # GB
        reserved = torch.cuda.memory_reserved(i) / (1024**3)    # GB
        total = torch.cuda.get_device_properties(i).total_memory / (1024**3)
        
        logger.info(f"GPU {i}:")
        logger.info(f"  Allocated: {allocated:.2f} GB / {total:.1f} GB ({allocated/total*100:.1f}%)")
        logger.info(f"  Reserved:  {reserved:.2f} GB / {total:.1f} GB ({reserved/total*100:.1f}%)")
    
    logger.info("")

def test_model_loading():
    """Test loading both models and verify GPU distribution."""
    logger.info("=" * 80)
    logger.info("LOADING MODELS")
    logger.info("=" * 80)
    
    # Load summarizer (4B model - should go to cuda:0)
    logger.info("Loading SUMMARIZER model (4B)...")
    summarizer = get_llm(LLMRole.SUMMARIZER)
    logger.info("✓ Summarizer loaded successfully")
    logger.info("")
    
    # Show memory after first model
    show_gpu_memory_usage()
    
    # Load buffer_agent (27B model - should go to cuda:1)
    logger.info("Loading BUFFER_AGENT model (27B)...")
    buffer_agent = get_llm(LLMRole.BUFFER_AGENT)
    logger.info("✓ Buffer Agent loaded successfully")
    logger.info("")
    
    # Show memory after both models
    show_gpu_memory_usage()
    
    logger.info("=" * 80)
    logger.info("VERIFICATION COMPLETE")
    logger.info("=" * 80)
    logger.info("If you have 2 GPUs, you should see:")
    logger.info("  - GPU 0: ~3-4 GB allocated (summarizer 4B model)")
    logger.info("  - GPU 1: ~16-18 GB allocated (buffer_agent 27B model)")
    logger.info("=" * 80)

if __name__ == "__main__":
    # Check GPU availability
    multi_gpu = check_gpu_availability()
    
    # Show initial memory state
    show_gpu_memory_usage()
    
    # Load models and verify distribution
    test_model_loading()
