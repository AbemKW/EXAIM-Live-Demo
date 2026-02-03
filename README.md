---
title: EXAIM - Clinical Decision Support Demo
emoji: 🏥
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: "5.9.1"
app_file: app.py
pinned: false
---

# EXAIM – MedGemma Impact Challenge Submission

## 🚀 Hugging Face Space Setup

This repository is configured to run as a Hugging Face Space. To deploy:

1. **Add Required Secrets** in your Space Settings:
   - `GOOGLE_API_KEY` - Your Google API key for Gemini models (required)
   - `MAS_LLM_PROVIDER` - Set to `google` (optional, defaults to google)
   - `MAS_LLM_MODEL` - Set to `gemini-2.0-flash-exp` (optional, defaults to gemini-2.0-flash-exp)

2. The Space will automatically use `app.py` as the entry point

3. For local development, copy `.env.example` to `.env` and add your API keys

---

EXAIM (Explainable AI Middleware) is an experimental system that captures timed multi-agent reasoning traces, buffers them, and produces concise structured summaries via role-based LLMs. This repository contains the EXAIM code and reproducible evaluation tooling for the MedGemma Impact Challenge submission.

## Quick Navigation

- 🎯 **Live Demo**: See EXAIM in action → [Interactive Demo Setup](#interactive-demo-local)
- 📊 **Performance Evaluation**: Run benchmarks → [Kaggle Notebook Evaluation](#kaggle--notebook-execution)
- 🔧 **Development**: Local Docker/Python setup → [evals/README.md](evals/README.md)

## Interactive Demo (Local)

EXAIM includes a **live interactive web demo** featuring a real-time Clinical Decision Support System (CDSS) interface. This demo showcases EXAIM's streaming summarization capabilities with a modern Next.js UI.

**⚠️ Note:** The interactive demo requires a local development environment with Node.js and cannot run in Kaggle notebooks. For Kaggle-based evaluation, see the [Kaggle & Notebook Execution](#kaggle--notebook-execution) section below.

### Running the Demo Locally

**Prerequisites:**
- Node.js 18+
- Python 3.10+

**Quick Start (Windows):**
```powershell
# Clone the repository with submodules
git clone --recurse-submodules https://github.com/AbemKW/EXAIM-MedGemma-Impact.git
cd EXAIM-MedGemma-Impact

# Install Python dependencies
pip install -r requirements.txt

# Install frontend dependencies
cd demos/frontend
npm install
cd ../..

# Start both servers and open browser automatically
.\start-dev.ps1
```

The script will:
- ✅ Start the FastAPI backend (port 8000)
- ✅ Start the Next.js frontend (port 3000)
- ✅ Automatically open your browser to http://localhost:3000

**Manual Start (Alternative):**

If you prefer to start servers manually or are on a non-Windows system:

```bash
# Terminal 1 - Backend
pip install -r requirements.txt
python -m demos.backend.server

# Terminal 2 - Frontend
cd demos/frontend
npm install
npm run dev
```

Then open [http://localhost:3000](http://localhost:3000) in your browser.

For detailed demo documentation, see [demos/frontend/README.md](demos/frontend/README.md).

## Kaggle & Notebook Execution

**For performance evaluation and benchmarking**, you can execute the full evaluation pipeline directly in a Kaggle Notebook (or Google Colab) using the provided free GPU resources (T4 x2). This approach allows you to run the evaluation without setting up Docker or requiring Vertex AI quotas.

**This section is for evaluation only** - not for the interactive demo. The interactive web demo requires a local environment (see [Interactive Demo](#interactive-demo-local) above).

### Setup for Kaggle (T4 x2 GPU)

Since Kaggle does not support Docker, use the following Python-native workflow to replicate the environment and serve the MedGemma model locally.

**1. Install Dependencies**
```python
# Install system libraries for nmslib
!apt-get update && apt-get install -y build-essential curl git

# Install Python dependencies (order matters for scispacy)
!pip install --no-binary :all: nmslib
!pip install -r evals/requirements-evals.txt
!pip install -r requirements.txt
!pip install --no-deps https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.4/en_core_sci_sm-0.5.4.tar.gz
!pip install vllm
```

**2. Download Knowledge Base**

The UMLS linker artifacts (~3GB) must be downloaded to a local directory:

```python
import os
linker_path = "/kaggle/working/scispacy_linkers/umls"
os.makedirs(linker_path, exist_ok=True)

# Download required artifacts
!curl -L -o {linker_path}/nmslib_index.bin https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/data/linkers/2020-10-09/umls/nmslib_index.bin
!curl -L -o {linker_path}/tfidf_vectorizer.joblib https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/data/linkers/2020-10-09/umls/tfidf_vectorizer.joblib
!curl -L -o {linker_path}/tfidf_vectors_sparse.npz https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/data/linkers/2020-10-09/umls/tfidf_vectors_sparse.npz
!curl -L -o {linker_path}/concept_aliases.json https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/data/linkers/2020-10-09/umls/concept_aliases.json
```

**3. Serve MedGemma Locally**

Instead of calling external APIs, we spin up a local vLLM server to host `google/medgemma-27b-it` using 4-bit quantization (fitting on Kaggle T4 GPUs).

```python
# Launch vLLM in background
# Ensure you have accepted the model terms on HF and added your HF_TOKEN as a secret
# Note: This process takes ~5-10 minutes to load weights
!vllm serve google/medgemma-27b-it --quantization bitsandbytes --dtype half --port 8000 &
```

**4. Run Evaluation**

Configure the `llm_registry` to point to the local server via the OpenAI compatibility layer:

```python
import os
import time
import requests

# Wait for server to be ready
print("Waiting for vLLM server...")
for i in range(20):
    try:
        requests.get("http://localhost:8000/v1/models")
        print("Server is ready!")
        break
    except:
        time.sleep(10)

# Configure environment to use local vLLM
os.environ["SUMMARIZER_LLM_PROVIDER"] = "openai"
os.environ["OPENAI_BASE_URL"] = "http://localhost:8000/v1"
os.environ["OPENAI_API_KEY"] = "EMPTY"

# Set paths for the evaluator
os.environ["PYTHONPATH"] = "/kaggle/working/ExAID:/kaggle/working/ExAID/third_party/mac"
os.environ["SCISPACY_LINKER_PATH"] = "/kaggle/working/scispacy_linkers"

# Run the variants
!python -m evals.cli.run_variants --traces data/traces/ --output data/runs/
```

**Results:** Official evaluation logs and results for the competition submission can be viewed directly in the output of our Public Kaggle Notebook (Version X).

## Docker & Local Evaluation

For Docker-based or local Python evaluation runs (development/testing), see the [evals/README.md](evals/README.md).

## Repository Structure

- `exaim_core/` — Core EXAIM implementation (BufferAgent, SummarizerAgent, TokenGate)
- `evals/` — Evaluation harness, pre-generated traces, and metrics computation
- `demos/` — Interactive web demo (CDSS interface)
- `infra/` — Model registry and configuration
- `third_party/mac/` — MAC trace generator (submodule, used for trace generation only)

## About the Submodule

This repository includes the [MAC (Multi-Agent Conversation)](https://github.com/AbemKW/mac-streaming-traces) framework as a git submodule in `third_party/mac`. MAC was used to generate the timed conversation traces stored in `evals/data/traces/`. 

**The traces are frozen and pre-generated** to ensure **deterministic replay across ablations** during evaluation. This allows fair comparison between different model variants (baseline vs. MedGemma) by replaying identical conversation traces with only the summarization model changed.

The submodule is **only needed if you want to regenerate traces from scratch** (not required for evaluation).

To initialize the submodule (optional):
```bash
git submodule update --init --recursive
```

**Note:** Our fork adds transparent delta-level timing instrumentation (`t_emitted_ms`) for realistic streaming replay. All MAC conversation logic remains unchanged.

---