# EXAIM Project Context & Goals

## Overview
**EXAIM (Explainable AI Middleware)** is a specialized middleware layer designed to bridge the gap between complex Multi-Agent Systems (MAS) and human-understandable clinical insights. It sits between Raw MAS outputs and the end-user to provide transparency, traceability, and interpretability in medical AI workflows.

## Deployment Target
* **Platform:** Hugging Face Spaces
* **Hardware:** Nvidia 1x L40S (48GB VRAM)
* **Performance Goal:** High-speed live demo execution for the Buffer and Summarizer agents.

## Technical Requirements
* **Models:** * **Buffer Agent:** Must utilize the 4-bit quantized **MedGemma-27b-IT** model: unsloth/medgemma-27b-text-it-unsloth-bnb-4bit.
    * **Summarizer Agent:** Must utilize the 4-bit quantized **MedGemma-4b** model: unsloth/medgemma-4b-it-unsloth-bnb-4bit.
* **Optimization:** Focus on low-latency interactions between the agents and the Raw MAS to ensure the live demo feels responsive.
* **Quantization:** All model implementations should prioritize 4-bit precision to maximize the L40S VRAM efficiency while maintaining clinical accuracy.

## Reference Material
* Full technical specifications and architecture details are located in: `/docs/Documentation.md`
* Current development focus: `enhance-summarizer` branch.

## AI Guidelines for Gemini CLI
1.  **Context Awareness:** Always refer to the code files themselves and also the `/docs/` folder for specific middleware logic before suggesting architectural changes.
2.  **Hardware Optimization:** When suggesting code for model loading, assume an NVIDIA L40S environment and utilize libraries like `bitsandbytes` for 4-bit quantization.