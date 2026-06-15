# NLP Assignment: Code-Assistance RAG Chatbot

A configurable Retrieval-Augmented Generation (RAG) chatbot for answering
questions about Python library functions. The app combines local retrieval,
optional reranking, local GGUF language models, prompt configuration, a FastAPI
backend, and a chat-style web UI.

The default configuration is tuned for low-latency local demos:

- FAISS local retrieval for faster search.
- Fast mode for retrieval-first answers.
- Full mode for LLM-generated answers.
- Streaming response in Full mode.
- In-memory cache for repeated questions.
- Short conversation memory for follow-up requests such as `more detail`.
- YAML-based model, retriever, reranker, and prompt configuration.

## Demo Questions

Suggested prompts:

- `How do I merge two pandas DataFrames on a common column?`
- `What is pandas.merge used for?`
- `How do I read a CSV file with pandas?`
- `How can I split data into train and test sets with sklearn?`
- `How do I fill missing values in pandas?`

## Project Structure

```text
configs/
  models.yaml          # Parser, generator, and reranker settings
  prompts.yaml         # System prompt and answer-generation templates
  retriever.yaml       # FAISS/PostgreSQL retrieval settings

docker/
  docker-compose.yml   # PostgreSQL/pgvector service

models/
  final_gguf/          # Local GGUF models, ignored by Git
  faiss/               # Local FAISS index, ignored by Git

src/
  api/                 # FastAPI app and chat UI
  core/                # LLM, retriever, and reranker base/factory layers
  models/              # Query parser, retriever agent, reranker agent, answer generator
  pipeline/            # RAG orchestrator
  tools/               # Data/index build utilities
  utils/               # Config loader and logging
```

## Pipeline Flow

```text
User question
  -> QueryParser
  -> DocumentRetriever
  -> optional DocumentReranker
  -> AnswerGenerator or Fast template answer
  -> API response / streamed UI response
```

For short follow-up requests such as `more detail`, `explain more`, or `nói rõ
hơn`, the API reuses the previous answer and sources from the same
`conversation_id` instead of running a new unrelated retrieval query.

### Fast Mode

Fast mode focuses on low latency:

1. Parses and optimizes the user query.
2. Retrieves relevant chunks from the FAISS index.
3. Skips the CrossEncoder reranker by default.
4. Returns a concise retrieval-based answer.

### Full Mode

Full mode focuses on richer answers:

1. Parses and optimizes the user query.
2. Retrieves relevant chunks from the FAISS index.
3. Skips reranking by default for demo speed, but can enable it in config.
4. Sends context to the configured generator model.
5. Streams generated text to the UI while the model is still producing output.

## Configuration

### Models

Model settings are in:

```text
configs/models.yaml
```

Example:

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

To switch models, change the model path and generation parameters here. Python
code does not need to change.

### Prompts

Prompt settings are in:

```text
configs/prompts.yaml
```

This file controls:

- system behavior
- retrieved context formatting
- user prompt template
- language hint
- max documentation characters per chunk
- `/no_think` behavior for compatible models

### Retriever

Retrieval settings are in:

```text
configs/retriever.yaml
```

Current demo-oriented defaults:

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

Important options:

- `retrieval_k`: number of initial retrieved candidates.
- `context_k`: number of chunks passed into the final answer step.
- `fast_mode_rerank`: whether Fast mode uses the CrossEncoder reranker.
- `full_mode_rerank`: whether Full mode uses the CrossEncoder reranker.
- `cache_size`: maximum number of cached query responses.

For higher accuracy but slower responses:

```yaml
fast_mode_rerank: true
full_mode_rerank: true
```

## Setup

### 1. Clone Repository

```bash
git clone https://github.com/baohuynh12056/NLP-assignment.git
cd NLP-assignment
```

### 2. Create Environment

Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

Linux/macOS:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 3. Add Local Models

Place GGUF model files in:

```text
models/final_gguf/
```

Expected default paths:

```text
models/final_gguf/qwen3-0.6b-instruct-q4_k_m.gguf
models/final_gguf/qwen3-4b-instruct-q4_k_m.gguf
```

These files are ignored by Git because they are large local artifacts.

### 4. Start Database

```bash
cd docker
docker compose up -d
cd ..
```

### 5. Build FAISS Index

Windows PowerShell:

```powershell
$env:PYTHONPATH='src'
python src\tools\build_faiss_index.py
```

Linux/macOS:

```bash
PYTHONPATH=src python src/tools/build_faiss_index.py
```

This creates:

```text
models/faiss/functions.index
models/faiss/functions_metadata.jsonl
```

The FAISS files are ignored by Git and should be rebuilt locally when the source
data changes.

## Run Locally

Windows PowerShell:

```powershell
$env:PYTHONPATH='src'
uvicorn api.main:app --host 127.0.0.1 --port 8000
```

Linux/macOS:

```bash
PYTHONPATH=src uvicorn api.main:app --host 127.0.0.1 --port 8000
```

Open:

- Web chat: `http://127.0.0.1:8000/`
- Swagger docs: `http://127.0.0.1:8000/docs`
- Health check: `http://127.0.0.1:8000/health`

## API Usage

### Standard Endpoint

```bash
curl -X POST http://127.0.0.1:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"query":"How do I read a CSV file with pandas?","mode":"fast"}'
```

Request body:

```json
{
  "query": "How do I read a CSV file with pandas?",
  "mode": "fast",
  "conversation_id": "demo-session-1"
}
```

Supported modes:

- `fast`: retrieval-first answer, optimized for quick demo responses.
- `full`: generator-model answer, streamed in the UI.

`conversation_id` is optional but recommended for chat UIs. It lets the backend
understand follow-up requests like `more detail` by reusing the previous turn's
answer and source chunks.

Response fields:

- `query`
- `optimized_query`
- `filters`
- `answer`
- `sources`

### Streaming Endpoint

The web UI uses `/ask/stream` for Full mode.

```bash
curl -N -X POST http://127.0.0.1:8000/ask/stream \
  -H "Content-Type: application/json" \
  -d '{"query":"What is pandas.merge used for?","mode":"full"}'
```

The endpoint returns newline-delimited JSON events:

```json
{"type":"token","text":"The pandas.merge function..."}
{"type":"done","response":{"answer":"...","sources":[]},"elapsed":16.05,"cached":false}
```

Repeated questions are served from the in-memory cache. The cache key includes
the selected mode and normalized query text.

## Public Demo With Cloudflare Tunnel

Run the local API first, then expose it:

```bash
cloudflared tunnel --url http://127.0.0.1:8000
```

Cloudflare will print a temporary public URL. Share that URL for a quick demo.

For ngrok:

```bash
ngrok http 8000
```

## Performance Notes

- The first request after server start is slower because models load lazily.
- Repeated questions are much faster because the response cache is used.
- Lower `max_tokens` reduces Full mode generation time.
- Lower `retrieval_k` and `context_k` reduce retrieval and prompt size.
- Disabling reranker improves demo speed but can reduce retrieval precision.
- Enabling `full_mode_rerank: true` improves source selection but increases
  latency.

## Troubleshooting

### `Model path does not exist`

Check that the GGUF files exist at the paths configured in `configs/models.yaml`.
Also run the server from the repository root with `PYTHONPATH=src`.

### `FAISS index files are missing`

Build the index:

```powershell
$env:PYTHONPATH='src'
python src\tools\build_faiss_index.py
```

### `Not Found` in browser

Open the root web UI:

```text
http://127.0.0.1:8000/
```

Use API docs for endpoints:

```text
http://127.0.0.1:8000/docs
```

### Push rejected with `fetch first`

Remote has new commits. Sync first:

```bash
git pull --ff-only origin main
git push origin main
```

## Future Improvements

Recommended next improvements:

- Add persistent cache with SQLite or Redis so cache survives server restart.
- Add loading metrics in the UI for retrieval, generation, and cache hits.
- Add unit tests for query parsing, retrieval config, cache behavior, and stream
  response events.
- Add evaluation questions with expected source functions to measure retrieval
  accuracy.
- Add Docker Compose service for the API so setup is one command.
- Add authentication only if the app is exposed beyond a short-lived demo tunnel.
- Add a small admin/settings panel for changing `fast/full`, reranker, and token
  settings without editing YAML manually.

## Git Notes

Large local artifacts are intentionally not committed:

- GGUF model files under `models/final_gguf/`
- FAISS index files under `models/faiss/`

Commit source code, configs, prompts, scripts, and documentation only.
