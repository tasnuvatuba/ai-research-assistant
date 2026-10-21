# AI Research Assistant

An AI-powered research assistant for question answering over research papers, built using Large Language Models (LLMs) and Retrieval-Augmented Generation (RAG).

## Overview

This project explores the development of a research-paper question-answering system that combines local LLM inference with document retrieval to generate context-aware answers.

The project progresses from basic LLM inference to a complete conversational RAG system, with additional experiments in parameter-efficient fine-tuning using LoRA. The goal is to build an assistant that can retrieve relevant information from research papers, understand follow-up questions, maintain conversational context, and generate more relevant answers.

## Project Workflow

### 1. Local LLM Inference

- Load and run Qwen2.5-0.5B-Instruct locally using Hugging Face Transformers.
- Implement a command-line interface for question answering.
- Explore tokenization, prompt formatting, and text generation.
- Run inference on the CPU using PyTorch.

### 2. Document Processing and RAG Retrieval

- Load research papers from PDF files using PyMuPDF.
- Split document text into smaller chunks using LangChain.
- Generate semantic embeddings using Sentence Transformers.
- Store document embeddings in a FAISS vector store.
- Retrieve relevant document passages based on user questions.

### 3. RAG Pipeline with LLM

- Integrate document retrieval with local LLM inference.
- Pass retrieved passages and user questions to the language model as context.
- Generate answers grounded in information retrieved from research papers.
- Explore how retrieval quality and prompt formatting affect answer relevance.

### 4. Conversational Memory and Question Contextualization

- Extend the RAG pipeline to support multi-turn conversations.
- Maintain conversation history across successive questions.
- Contextualize follow-up questions using previous conversation turns.
- Use contextualized questions for document retrieval.
- Generate answers based on retrieved information and relevant conversational context.

### 5. LoRA Fine-Tuning and Local Adapter Integration

- Experiment with parameter-efficient fine-tuning using LoRA in Google Colab.
- Fine-tune a compatible language model using a task-specific dataset.
- Save the trained LoRA adapter.
- Load the trained adapter alongside its corresponding base model for local inference.
- Explore the effect of fine-tuning on answer quality and task performance.

## Technology Stack

- **Programming Language:** Python
- **Deep Learning:** PyTorch
- **LLM Inference:** Hugging Face Transformers
- **Base Language Model:** Qwen2.5-0.5B-Instruct
- **RAG Framework:** LangChain
- **Document Processing:** PyMuPDF
- **Text Embeddings:** Sentence Transformers (`all-MiniLM-L6-v2`)
- **Vector Database:** FAISS
- **Parameter-Efficient Fine-Tuning:** LoRA
- **Fine-Tuning Environment:** Google Colab

## Project Structure

```text
ai-research-assistant/
├── application/
├── src/
├── test/
├── scripts/
├── data/
├── vector_store/
├── main.py
├── requirements.txt
└── README.md
```

## Setup and Installation

### Prerequisites

- Python 3.10 or a compatible Python version
- Git
- Sufficient memory and storage to run the language model locally

### 1. Clone the repository

```bash
git clone https://github.com/tasnuvatuba/ai-research-assistant.git
cd ai-research-assistant
```

### 2. Create and activate a virtual environment

```bash
python -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Add research papers

Place the research-paper PDF files in the project's configured paper directory.

### 5. Run the assistant

Run the application using the appropriate entry point configured in the project.

The first run may download the base language model and embedding model from Hugging Face.

## Testing

Run the RAG pipeline test using:

```bash
PYTHONPATH=src python test/test_rag_pipeline.py
```
## Project Goals

This project serves as a practical exploration of LLM application development, covering local inference, semantic search, RAG pipeline construction, conversational question answering, and parameter-efficient fine-tuning.

The long-term goal is to develop a functional research assistant that can help users explore research papers through context-aware, document-grounded conversations.

## Repository

[AI Research Assistant on GitHub](https://github.com/tasnuvatuba/ai-research-assistant)
