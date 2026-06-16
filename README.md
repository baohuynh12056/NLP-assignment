# NLP Assignment: Chat-Based Code Assistant

A configurable, privacy-preserving Retrieval-Augmented Generation (RAG) chatbot designed specifically to assist developers with Python library functions. The application combines local document retrieval, optional cross-encoder reranking, local GGUF large language models (LLMs), dynamic prompt configuration, a FastAPI backend, and an interactive chat-style web UI.

---

### Course Information

| Item           | Information                           |
| -------------- | --------------------------------------|
| **Course**     | Natural Language Processing (CO3085)  |
| **Class**      | TN01                                  |
| **Instructor** | Nguyen Duc Dung, PhD.                 |

### Student Team

| Student ID | Full Name     |
| ---------- | ------------- |
| 2410233    | Huynh Gia Bao |
| 2413506    | Ngo Trung Tin |
| 2413631    | Lai Tran Tri  |

---

# 1. Introduction

## Project Overview

This project presents a **Chat-Based Code Assistant** built using a Retrieval-Augmented Generation (RAG) architecture. The system is designed to help developers efficiently search, understand, and utilize Python library functions through a conversational interface while maintaining privacy through fully local deployment.

The project was developed as part of the **Natural Language Processing (CO3085)** course under the supervision of **Nguyen Duc Dung, PhD.**

## The Problem

Modern software development requires developers to navigate a vast and constantly evolving ecosystem of libraries. Developers often know exactly *what* logic they want to implement but struggle to recall:

* Function signatures
* Expected data types
* Parameter behaviors
* Usage patterns

Constantly switching contexts to search through documentation disrupts productivity and increases cognitive load.

In addition, generic LLMs frequently suffer from **hallucinations**, generating non-existent APIs or incorrect parameters that can lead to broken code and wasted debugging time.

## The Solution

This system acts as a precise, interactive knowledge broker by combining retrieval, validation, and generation techniques.

### Neuro-Symbolic Guardrails

The system routes and filters user queries before generation, rejecting out-of-domain requests and minimizing hallucinations.

### Stateful Memory

Short-term conversational context allows developers to ask follow-up questions such as:

> "Give me an example."

or

> "Explain the `how='left'` parameter."

without repeating the original query.

### Configurable Trade-offs

Developers can dynamically switch between:

* **Fast Mode** — retrieval-focused, low latency
* **Full Mode** — LLM-generated explanations and code examples

## Target Audience

### Software Engineers & Data Scientists

Professionals looking for a fast, offline, and privacy-focused tool to quickly look up Python syntax without leaving their local environment.

### Students & Beginners

Learners transitioning to new Python frameworks who need runnable examples and clear explanations grounded in real documentation.

## Limitations & Scope

### Data-Bounded Knowledge Base

The current index contains approximately **2,978 parsed functions** across 10 Python libraries:

* Pandas
* NumPy
* Scikit-learn
* PyTorch
* TensorFlow
* Matplotlib
* SciPy
* Seaborn
* Requests
* FastAPI

The system cannot answer questions about libraries that have not been indexed.

### Hardware Constraints

Running the full local pipeline (Retriever, Reranker, Generator) may require consumer GPUs such as an RTX 4070. Full Mode responses will generally have higher latency than cloud-based services.

### Micro-Level Focus

The assistant is optimized for **function-level API lookup and explanation**, not large-scale repository architecture analysis.

---

# 2. Demo

https://github.com/user-attachments/assets/9833445b-b1b6-45f3-a41e-acc1d13d9724

# 3. Project Structure

```text
configs/
├── models.yaml          # Parser, generator, and reranker settings
├── prompts.yaml         # System prompts and answer templates
└── retriever.yaml       # Retrieval settings

docker/
└── docker-compose.yml   # PostgreSQL + pgvector service

models/
├── final_gguf/          # Local GGUF models (gitignored)
└── faiss/               # FAISS index files (gitignored)

src/
├── api/                 # FastAPI backend and chat UI
├── core/                # Base/factory abstractions
├── models/              # Query parser, retriever, reranker, generator
├── pipeline/            # RAG orchestration pipeline
├── tools/               # Indexing and utility scripts
└── utils/               # Config and logging helpers
```

---

# 4. System Overview

<img width="1536" height="1024" alt="core_architecture" src="https://github.com/user-attachments/assets/f4b0664b-b89b-4f35-8fb9-10137ed9f570" />

For short follow-up requests, the system can reuse the previous answer and source documents through a **Stateful Cache**, bypassing expensive retrieval and reranking steps.

---

## Fast Mode

1. Parse and normalize the query.
2. Retrieve relevant chunks from FAISS.
3. Skip reranking by default.
4. Return concise retrieval-based documentation snippets.

### Benefits

* Lowest latency
* Lower hardware usage
* Suitable for quick API lookups

---

## Full Mode

1. Parse and normalize the query.
2. Retrieve relevant chunks from FAISS.
3. Send context to the generator model.
4. Stream generated explanations and code examples.

### Benefits

* Rich explanations
* Better code examples
* Improved answer quality

Reranking can optionally be enabled for maximum retrieval precision.

---

# 5. Configuration

## Models

Modify `configs/models.yaml` to switch models or adjust generation parameters.

```yaml
parser:
  type: "llama_cpp"
  model_path: "models/final_gguf/qwen3-0.6b-instruct-q4_k_m.gguf"
  context_window: 512
  max_tokens: 150

generator:
  type: "llama_cpp"
  model_path: "models/final_gguf/qwen3-4b-instruct-q4_k_m.gguf"
  context_window: 4096
  max_tokens: 512

reranker:
  type: "cross_encoder"
  model_path: "BAAI/bge-reranker-base"
  top_k: 3
```

## Retriever

Modify `configs/retriever.yaml` to adjust retrieval behavior.

```yaml
type: "faiss_local"

retrieval_k: 8
context_k: 3

fast_mode_rerank: false
full_mode_rerank: false

cache_size: 128
```

### Tip

Enable reranking for improved accuracy:

```yaml
full_mode_rerank: true
```

This improves retrieval precision at the cost of slightly higher latency.

---

# Setup Instructions

## 1. Clone Repository

```bash
git clone https://github.com/baohuynh12056/NLP-assignment.git
cd NLP-assignment
```

---

## 2. Create Virtual Environment

### Windows (PowerShell)

```powershell
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

### Linux/macOS

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

---

## 3. Download and Configure Models

Download the required GGUF model files and place them in:

```text
models/final_gguf/
```

### Recommended Models

The project has been tested with the following GGUF models:

| Model                               | Size  | Download                                               |
| ----------------------------------- | ----- | ------------------------------------------------------ |
| `qwen2.5-0.5b-instruct-q4_k_m.gguf` | ~0.5B | https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF |
| `qwen2.5-1.5b-instruct-q4_k_m.gguf` | ~1.5B | https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct-GGUF |
| `qwen3-0.6b-instruct-q4_k_m.gguf`   | ~0.6B | https://huggingface.co/Qwen/Qwen3-0.6B-GGUF            |
| `qwen3-4b-instruct-q4_k_m.gguf`     | ~4B   | https://huggingface.co/Qwen/Qwen3-4B-GGUF              |

After downloading, place the `.gguf` files in:

```text
models/final_gguf/
```

Example:

```text
models/
└── final_gguf/
    ├── qwen3-0.6b-instruct-q4_k_m.gguf
    └── qwen3-4b-instruct-q4_k_m.gguf
```

### Important

The filenames **must exactly match** the paths specified in:

```text
configs/models.yaml
```

You can either:

* Rename the downloaded file to match the configuration.
* Update `model_path` in `configs/models.yaml` to match the downloaded filename.

For example:

```yaml
generator:
  model_path: models/final_gguf/qwen3-0.6b-instruct-q4_k_m.gguf
```

> **Note:** GGUF model files are intentionally excluded from Git because they are large local artifacts.

---

## 4. Start PostgreSQL + pgvector

```bash
cd docker
docker compose up -d
cd ..
```

Verify that Docker is running:

```bash
docker ps
```

---

## 5. Ingest Data into PostgreSQL

The preprocessed data already exists in:

```text
data/chunks/functions.jsonl
```

Load it into PostgreSQL:

### Windows

```powershell
.\.venv\Scripts\activate
$env:PYTHONPATH='src'
python src\tools\ingest.py
```

### Linux/macOS

```bash
source .venv/bin/activate
PYTHONPATH=src python src/tools/ingest.py
```

---

## 6. Build FAISS Index

Create the local FAISS index:

### Windows

```powershell
$env:PYTHONPATH='src'
python src\tools\build_faiss_index.py
```

### Linux/macOS

```bash
PYTHONPATH=src python src/tools/build_faiss_index.py
```

Generated files:

```text
models/faiss/
    functions.index
    functions_metadata.jsonl
```

---

# Running the Application

Start the FastAPI server:

### Windows

```powershell
$env:PYTHONPATH='src'
uvicorn api.main:app --host 127.0.0.1 --port 8000
```

### Linux/macOS

```bash
PYTHONPATH=src uvicorn api.main:app --host 127.0.0.1 --port 8000
```

---

## Access the Application

* **Web Chat:** `http://127.0.0.1:8000/`
* **Swagger API Docs:** `http://127.0.0.1:8000/docs`

---

# API Usage

## Standard Endpoint

```bash
curl -X POST http://127.0.0.1:8000/ask \
  -H "Content-Type: application/json" \
  -d '{
        "query":"How do I read a CSV file with pandas?",
        "mode":"fast"
      }'
```

### Supported Modes

| Mode   | Description                                   |
| ------ | --------------------------------------------- |
| `fast` | Retrieval-first response optimized for speed  |
| `full` | LLM-generated response with streaming support |

---

## Streaming Endpoint

Used by the web UI for Full mode:

```bash
curl -N -X POST http://127.0.0.1:8000/ask/stream \
  -H "Content-Type: application/json" \
  -d '{
        "query":"What is pandas.merge used for?",
        "mode":"full"
      }'
```

---

# Public Demo via Cloudflare Tunnel

Expose the local server publicly:

```bash
cloudflared tunnel --url http://127.0.0.1:8000
```

Cloudflare will generate a temporary public URL that can be shared for demonstrations.

---


# License

This project was developed for educational purposes as part of an NLP assignment.

