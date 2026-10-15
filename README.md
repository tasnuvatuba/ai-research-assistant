# AI Research Assistant

A hands-on project to learn Large Language Models (LLMs),
LoRA fine-tuning, Retrieval-Augmented Generation (RAG),
and backend development.

## Current Progress

### Day 1: Basic LLM Inference
- Loaded Qwen2.5-0.5B-Instruct using Hugging Face Transformers.
- Implemented a command-line question-answering interface.
- Explored tokenization, prompt formatting, and text generation.
- Used CPU inference with PyTorch.

## Technology Stack

- Python
- PyTorch
- Hugging Face Transformers
- Qwen2.5-0.5B-Instruct

## Setup

Create and activate a virtual environment, then install dependencies:

    python -m venv .venv
    source .venv/bin/activate
    pip install -r requirements.txt

Run the assistant:

    python main.py

The first run downloads the model from Hugging Face.
