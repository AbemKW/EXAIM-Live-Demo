# GPU Distribution Fix - Multi-GPU Load Balancing

## Problem
All HuggingFace models were loading on `cuda:0` by default, causing memory bottlenecks:
- **27B buffer_agent model**: ~16-18 GB (quantized 4-bit)
- **4B summarizer model**: ~3 GB (quantized 4-bit)
- **Total on cuda:0**: ~20+ GB + overhead → **Exceeding 24GB limit**

With 2x 24GB GPUs (48GB total), the first GPU was maxed out while the second sat idle.

## Solution
Modified `infra/llm_registry.py` to intelligently distribute models across available GPUs:

### Key Changes

1. **Added GPU Assignment Function** (`_get_device_assignment`)
   - Detects number of available CUDA devices
   - Assigns models based on size:
     - **27B models** → `cuda:1` (second GPU)
     - **4B models** → `cuda:0` (first GPU)
   - Falls back to `"auto"` for single-GPU or CPU systems
   - Caches assignments to prevent redundant checks

2. **Modified Model Loading**
   - Changed from hardcoded `device_map="auto"` to dynamic `device_map=_get_device_assignment(model_name)`
   - Added logging to track which GPU each model is assigned to
   - Import `torch` at module level for GPU detection

3. **GPU Assignment Tracking**
   - Added `_GPU_ASSIGNMENTS` global cache
   - Prevents duplicate GPU assignments
   - Provides visibility into model placement

### Distribution Strategy

| Model | Size | GPU | Memory Usage |
|-------|------|-----|--------------|
| Summarizer (`medgemma-1.5-4b-it`) | 4B | cuda:0 | ~3-4 GB |
| Buffer Agent (`medgemma-27b-text-it`) | 27B | cuda:1 | ~16-18 GB |

**Result**: ~4GB on GPU 0, ~18GB on GPU 1 → Both GPUs under 24GB limit ✓

## Testing

Run the test script to verify GPU distribution:

```powershell
python test_gpu_distribution.py
```

Expected output:
```
GPU AVAILABILITY CHECK
✓ CUDA is available
✓ Number of GPUs: 2
  GPU 0: NVIDIA GeForce RTX 3090 - 24.0 GB
  GPU 1: NVIDIA GeForce RTX 3090 - 24.0 GB

LOADING MODELS
Loading SUMMARIZER model (4B)...
Assigning 4B model to cuda:0 (first GPU)
✓ Summarizer loaded successfully

GPU MEMORY USAGE
GPU 0:
  Allocated: 3.42 GB / 24.0 GB (14.3%)
  Reserved:  3.85 GB / 24.0 GB (16.0%)
GPU 1:
  Allocated: 0.00 GB / 24.0 GB (0.0%)
  Reserved:  0.00 GB / 24.0 GB (0.0%)

Loading BUFFER_AGENT model (27B)...
Assigning 27B model to cuda:1 (second GPU)
✓ Buffer Agent loaded successfully

GPU MEMORY USAGE
GPU 0:
  Allocated: 3.42 GB / 24.0 GB (14.3%)
  Reserved:  3.85 GB / 24.0 GB (16.0%)
GPU 1:
  Allocated: 17.21 GB / 24.0 GB (71.7%)
  Reserved:  17.84 GB / 24.0 GB (74.3%)
```

## Benefits

✅ **Eliminates memory bottleneck** - No more OOM errors on cuda:0  
✅ **Faster inference** - Both GPUs utilized, no memory swapping  
✅ **Automatic fallback** - Works on single-GPU and CPU systems  
✅ **Zero config changes** - Works with existing `model_configs.yaml`  
✅ **Cached assignments** - No overhead on subsequent loads  

## Environment Variables

No new environment variables required. The existing configuration works:

```yaml
# model_configs.yaml
summarizer:
  provider: huggingface
  model: unsloth/medgemma-1.5-4b-it-unsloth-bnb-4bit  # → cuda:0

buffer_agent:
  provider: huggingface
  model: unsloth/medgemma-27b-text-it-unsloth-bnb-4bit  # → cuda:1
```

## Fallback Behavior

- **2+ GPUs**: Models distributed as shown above
- **1 GPU**: Uses `device_map="auto"` (transformers decides placement)
- **0 GPUs (CPU)**: Uses `device_map="cpu"`

## Future Enhancements

If you add more models or need finer control, you can:
1. Add GPU affinity hints to `model_configs.yaml`
2. Implement round-robin assignment for 3+ models
3. Use `device_map` dictionaries for layer-wise splitting across GPUs
