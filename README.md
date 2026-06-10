# NLP Assignment: Configurable Code-Assistance RAG

Branch `newstructure` refactors the project into a cleaner, configurable RAG
application. Models, retrievers, rerankers, and prompts are selected from YAML
config files instead of being hard-coded in Python.

## What This App Does

The app answers questions about Python library functions. A user asks a question,
the pipeline retrieves relevant function documentation from PostgreSQL/pgvector,
reranks the results, and returns an answer through a web chat UI or the `/ask` API.

Example questions:

- `How do I merge two pandas DataFrames on a common column?`
- `How do I read a CSV file with pandas?`
- `How do I split data into train and test sets with sklearn?`

## New Structure

Important folders:

- `configs/`: YAML configuration for models, prompts, and retrieval.
- `src/core/llm/`: base LLM interface, llama.cpp implementation, and LLM factory.
- `src/core/retriever/`: retriever interface, PostgreSQL hybrid retriever, and factory.
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
- retrieval count
- PostgreSQL connection
- semantic/keyword fusion weights

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

## Public Demo

To expose the local web app quickly:

```bash
cloudflared tunnel --url http://127.0.0.1:8000
```

Cloudflare will print a temporary public URL that can be shared for demos.

## Notes

- `.gguf` files are ignored and should not be pushed to GitHub.
- The first request after server start is slower because models are loaded lazily.
- Edit `configs/models.yaml` to switch models.
- Edit `configs/prompts.yaml` to tune answer style and prompt format.
