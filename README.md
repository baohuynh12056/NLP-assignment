# NLP Assignment: Code-Assistance RAG Chatbot

This project is a configurable RAG chatbot for answering questions about Python
library functions. Models, retrievers, rerankers, prompts, and demo-speed
settings are controlled from YAML config files instead of being hard-coded in
Python.

## What This App Does

The app answers questions about Python library functions. A user asks a question,
the pipeline parses the query, retrieves relevant function documentation,
optionally reranks the results, and returns an answer through a web chat UI or
the API.

Example questions:

- `How do I merge two pandas DataFrames on a common column?`
- `How do I read a CSV file with pandas?`
- `How do I split data into train and test sets with sklearn?`

## New Structure

Important folders:

- `configs/`: YAML configuration for models, prompts, and retrieval.
- `src/core/llm/`: base LLM interface, llama.cpp implementation, and LLM factory.
- `src/core/retriever/`: retriever interface, FAISS/PostgreSQL retrievers, and factory.
- `src/core/reranker/`: reranker interface, CrossEncoder reranker, and factory.
- `src/models/`: business-level agents such as query parser and answer generator.
- `src/pipeline/`: RAG orchestrator.
- `src/api/`: FastAPI app and web chat UI.

## Config Files

### Model Config

Model settings live in:

```text
configs/models.yaml
```

The app currently supports `llama_cpp` models through the LLM factory:

```yaml
parser:
  type: "llama_cpp"
  model_path: "models/final_gguf/qwen3-0.6b-instruct-q4_k_m.gguf"

generator:
  type: "llama_cpp"
  model_path: "models/final_gguf/qwen3-4b-instruct-q4_k_m.gguf"
```

To change model, update `model_path`, `context_window`, `temperature`, or
`max_tokens` in this file. The Python code does not need to change.

### Prompt Config

Prompt settings live in:

```text
configs/prompts.yaml
```

The generator prompt is split into:

- `system`: the model behavior.
- `context_template`: how each retrieved source is formatted.
- `user_template`: how context, query, and language hint are passed to the model.
- `max_chunk_chars`: max documentation length per source.
- `no_think`: appends `/no_think` for compatible models.

This makes prompt tuning easy without touching `AnswerGenerator`.

### Retriever Config

Retrieval settings live in:

```text
configs/retriever.yaml
```

It controls:

- embedding model path
- retrieval count and number of context chunks
- retriever type: `faiss_local` or `pg_hybrid`
- FAISS index/metadata paths
- Fast/Full reranking behavior
- response cache size
- PostgreSQL connection
- semantic/keyword fusion weights

Current default:

```yaml
type: "faiss_local"
retrieval_k: 8
context_k: 3
index_path: "models/faiss/functions.index"
metadata_path: "models/faiss/functions_metadata.jsonl"
fast_mode_rerank: false
full_mode_rerank: false
cache_size: 128
```

With this setting, both Fast mode and Full mode use FAISS and skip the
CrossEncoder reranker for lower demo latency. Full mode still uses the generator
LLM, but it receives fewer context chunks and can stream text to the UI while it
is generating.

To switch back to PostgreSQL hybrid search:

```yaml
type: "pg_hybrid"
fast_mode_rerank: true
full_mode_rerank: true
```

## Web Chat UI

The chat UI is served from:

```text
src/api/chat_ui.html
```

Features:

- Chat-only interface.
- Fixed top bar.
- Input composer at the bottom.
- Enter to send, Shift+Enter for newline.
- Quick prompt buttons.
- Fast/Full mode toggle in the settings drawer.
- Source snippets shown in the drawer.
- Code block rendering with a copy button.
- Full mode streaming, so the answer appears gradually instead of waiting for
  the whole generation to finish.

## Run Locally

Install dependencies:

```bash
pip install -r requirements.txt
```

Start PostgreSQL/pgvector:

```bash
cd docker
docker compose up -d
cd ..
```

Build the local FAISS retrieval index from PostgreSQL:

```bash
PYTHONPATH=src python src/tools/build_faiss_index.py
```

On Windows PowerShell:

```powershell
$env:PYTHONPATH='src'
python src\tools\build_faiss_index.py
```

This creates:

- `models/faiss/functions.index`
- `models/faiss/functions_metadata.jsonl`

These files are local artifacts and are ignored by Git.

Run the API:

```bash
PYTHONPATH=src uvicorn api.main:app --host 127.0.0.1 --port 8000
```

On Windows PowerShell:

```powershell
$env:PYTHONPATH='src'
uvicorn api.main:app --host 127.0.0.1 --port 8000
```

Open:

- Web chat: `http://127.0.0.1:8000/`
- API docs: `http://127.0.0.1:8000/docs`
- Health check: `http://127.0.0.1:8000/health`

## API

### Standard Answer

Request:

```bash
curl -X POST http://127.0.0.1:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"query":"How do I read a CSV file with pandas?","mode":"fast"}'
```

Body:

```json
{
  "query": "How do I read a CSV file with pandas?",
  "mode": "fast"
}
```

Modes:

- `fast`: retrieval-based answer with lightweight templates for common questions.
- `full`: uses the configured generator LLM and prompt from `configs/prompts.yaml`.

Response includes:

- `query`
- `optimized_query`
- `filters`
- `answer`
- `sources`

### Streaming Answer

The web UI uses `/ask/stream` for Full mode. It returns newline-delimited JSON
events:

```bash
curl -N -X POST http://127.0.0.1:8000/ask/stream \
  -H "Content-Type: application/json" \
  -d '{"query":"What is pandas.merge used for?","mode":"full"}'
```

Example event types:

```json
{"type":"token","text":"Use pandas.merge..."}
{"type":"done","response":{"answer":"...","sources":[]},"elapsed":16.05,"cached":false}
```

Repeated questions are served from an in-memory LRU cache. The cache key includes
the selected mode and normalized query text.

## Public Demo

To expose the local web app quickly:

```bash
cloudflared tunnel --url http://127.0.0.1:8000
```

Cloudflare will print a temporary public URL that can be shared for demos.

## Notes

- `.gguf` files are ignored and should not be pushed to GitHub.
- `models/faiss/` is ignored and should be rebuilt locally from PostgreSQL.
- The first request after server start is slower because models are loaded lazily.
- Warm repeated requests are much faster because they are served from the
  in-memory response cache.
- Fast mode is retrieval-first and avoids the generator model.
- Full mode can be made more accurate by enabling `full_mode_rerank: true`, but
  demo latency will increase.
- Edit `configs/models.yaml` to switch models.
- Edit `configs/prompts.yaml` to tune answer style and prompt format.
