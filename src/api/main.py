import argparse
import os
import sys
from functools import lru_cache
from pathlib import Path
from threading import Lock
from typing import Callable, List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from core.base_llm import BaseLLM
from core.schemas import Chunk, ParsedQuery, RAGResponse
from models.llm import QwenLocalLLM
from models.query_parser import MicroParserLLM
from models.reranker import BGEReranker
from models.retriever import PGHybridRetriever
from pipeline.rag_orchestrator import RAGPipeline


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

INDEX_HTML_PATH = Path(__file__).with_name("chat_ui.html")


INDEX_HTML = r"""
<!doctype html>
<html lang="vi">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Code Assistant</title>
  <style>
    :root {
      --bg: #f4f6f8;
      --panel: #ffffff;
      --ink: #111827;
      --muted: #667085;
      --line: #d9dee7;
      --user: #0f766e;
      --bot: #ffffff;
      --bot-line: #e4e7ec;
      --soft: #ecfdf5;
      --danger: #b42318;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      background: var(--bg);
      color: var(--ink);
      font: 15px/1.5 system-ui, -apple-system, Segoe UI, sans-serif;
    }
    .app {
      height: 100vh;
      height: 100dvh;
      overflow: hidden;
      display: grid;
      grid-template-rows: auto 1fr auto;
    }
    .topbar {
      height: 56px;
      display: flex;
      align-items: center;
      justify-content: center;
      position: sticky;
      top: 0;
      z-index: 5;
      background: rgba(255, 255, 255, 0.92);
      border-bottom: 1px solid var(--line);
      backdrop-filter: blur(10px);
    }
    .title {
      font-weight: 700;
      letter-spacing: 0;
    }
    .settings-btn {
      position: absolute;
      right: 16px;
      width: 36px;
      height: 36px;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: var(--panel);
      color: var(--ink);
      cursor: pointer;
      font-size: 18px;
    }
    .chat {
      overflow: auto;
      padding: 28px 16px 24px;
    }
    .messages {
      max-width: 820px;
      margin: 0 auto;
      display: grid;
      gap: 14px;
    }
    .message {
      display: grid;
      gap: 6px;
      max-width: min(720px, 92%);
    }
    .message.user {
      justify-self: end;
    }
    .message.bot {
      justify-self: start;
    }
    .bubble {
      padding: 12px 14px;
      border-radius: 16px;
      white-space: pre-wrap;
      word-break: break-word;
      box-shadow: 0 1px 2px rgba(16, 24, 40, 0.04);
    }
    .user .bubble {
      background: var(--user);
      color: white;
      border-bottom-right-radius: 6px;
    }
    .bot .bubble {
      background: var(--bot);
      border: 1px solid var(--bot-line);
      border-bottom-left-radius: 6px;
    }
    .label {
      color: var(--muted);
      font-size: 12px;
      padding: 0 4px;
    }
    .user .label {
      text-align: right;
    }
    .composer-wrap {
      padding: 14px 16px 18px;
      background: linear-gradient(to top, var(--bg) 78%, rgba(244, 246, 248, 0));
    }
    .composer {
      max-width: 820px;
      margin: 0 auto;
      display: grid;
      grid-template-columns: 1fr auto;
      gap: 10px;
      align-items: end;
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 14px;
      padding: 8px;
      box-shadow: 0 8px 24px rgba(16, 24, 40, 0.08);
    }
    textarea {
      width: 100%;
      min-height: 42px;
      max-height: 150px;
      resize: vertical;
      border: 0;
      padding: 10px 12px;
      color: var(--ink);
      font: inherit;
      outline: none;
      background: transparent;
    }
    .send {
      width: 42px;
      height: 42px;
      border: 0;
      border-radius: 10px;
      background: var(--user);
      color: white;
      cursor: pointer;
      font-size: 18px;
      font-weight: 800;
    }
    .send:disabled {
      opacity: 0.55;
      cursor: wait;
    }
    .meta {
      max-width: 820px;
      margin: 8px auto 0;
      color: var(--muted);
      font-size: 12px;
      min-height: 18px;
    }
    .drawer {
      position: fixed;
      top: 0;
      right: 0;
      height: 100vh;
      width: min(380px, 92vw);
      background: var(--panel);
      border-left: 1px solid var(--line);
      box-shadow: -20px 0 40px rgba(16, 24, 40, 0.12);
      transform: translateX(100%);
      transition: transform 160ms ease;
      z-index: 10;
      display: grid;
      grid-template-rows: auto auto 1fr;
    }
    .drawer.open {
      transform: translateX(0);
    }
    .drawer-head {
      display: flex;
      align-items: center;
      justify-content: space-between;
      padding: 16px;
      border-bottom: 1px solid var(--line);
    }
    .drawer h2 {
      margin: 0;
      font-size: 16px;
    }
    .close {
      border: 1px solid var(--line);
      border-radius: 8px;
      background: var(--panel);
      width: 34px;
      height: 34px;
      cursor: pointer;
    }
    .settings {
      padding: 14px 16px;
      border-bottom: 1px solid var(--line);
      display: grid;
      gap: 10px;
    }
    .mode {
      display: grid;
      grid-template-columns: 1fr 1fr;
      border: 1px solid var(--line);
      border-radius: 8px;
      overflow: hidden;
      height: 42px;
    }
    .mode label {
      display: grid;
      place-items: center;
      color: var(--muted);
      cursor: pointer;
      font-weight: 700;
    }
    .mode input {
      position: absolute;
      opacity: 0;
      pointer-events: none;
    }
    .mode label:has(input:checked) {
      background: var(--soft);
      color: var(--user);
    }
    .sources {
      overflow: auto;
      padding: 14px 16px 22px;
    }
    .source {
      border-top: 1px solid var(--line);
      padding: 12px 0;
    }
    .source:first-child {
      border-top: 0;
      padding-top: 0;
    }
    .source strong {
      display: block;
      font-size: 14px;
      margin-bottom: 4px;
    }
    .source small {
      color: var(--muted);
      display: block;
      margin-bottom: 6px;
    }
    .source p {
      margin: 0;
      color: #475467;
      font-size: 13px;
    }
    .error .bubble {
      color: var(--danger);
      border-color: #fda29b;
    }
    @media (max-width: 640px) {
      .topbar { height: 52px; }
      .chat { padding-top: 18px; }
      .message { max-width: 96%; }
      .composer-wrap { padding: 10px; }
    }
  </style>
</head>
<body>
  <div class="app">
    <header class="topbar">
      <div class="title">Code Assistant</div>
      <button class="settings-btn" id="settings-button" title="Settings and sources">⚙</button>
    </header>

    <main class="chat" id="chat">
      <div class="messages" id="messages">
        <div class="message bot">
          <div class="label">Assistant</div>
          <div class="bubble">Ask me about Python library functions. For example: How do I merge two pandas dataframes on a common column?</div>
        </div>
      </div>
    </main>

    <footer class="composer-wrap">
      <form class="composer" id="ask-form">
        <textarea id="query" placeholder="Ask a question..." required>Làm sao để kết hợp 2 dataframes dựa trên một cột chung trong pandas?</textarea>
        <button class="send" id="send" type="submit" title="Send">↑</button>
      </form>
      <div class="meta" id="meta"></div>
    </footer>
  </div>

  <aside class="drawer" id="drawer">
    <div class="drawer-head">
      <h2>Settings</h2>
      <button class="close" id="close-drawer">×</button>
    </div>
    <div class="settings">
      <div class="mode" aria-label="Answer mode">
        <label><input type="radio" name="mode" value="fast" checked />Fast</label>
        <label><input type="radio" name="mode" value="full" />Full</label>
      </div>
      <small style="color: var(--muted)">Fast is retrieval-based. Full uses the local Qwen model and is slower.</small>
      <a href="/docs" target="_blank" style="color: var(--user); font-weight: 700; text-decoration: none;">Open API docs</a>
    </div>
    <div class="sources" id="sources">No sources yet.</div>
  </aside>

  <script>
    const form = document.getElementById("ask-form");
    const queryEl = document.getElementById("query");
    const sendEl = document.getElementById("send");
    const messagesEl = document.getElementById("messages");
    const chatEl = document.getElementById("chat");
    const metaEl = document.getElementById("meta");
    const drawerEl = document.getElementById("drawer");
    const sourcesEl = document.getElementById("sources");

    document.getElementById("settings-button").addEventListener("click", () => {
      drawerEl.classList.add("open");
    });
    document.getElementById("close-drawer").addEventListener("click", () => {
      drawerEl.classList.remove("open");
    });

    function addMessage(role, text, className = "") {
      const message = document.createElement("div");
      message.className = `message ${role === "You" ? "user" : "bot"} ${className}`;
      const label = document.createElement("div");
      label.className = "label";
      label.textContent = role;
      const bubble = document.createElement("div");
      bubble.className = "bubble";
      bubble.textContent = text;
      message.append(label, bubble);
      messagesEl.appendChild(message);
      chatEl.scrollTop = chatEl.scrollHeight;
      return bubble;
    }

    function renderSources(sources) {
      sourcesEl.innerHTML = "";
      if (!sources.length) {
        sourcesEl.textContent = "No sources.";
        return;
      }
      for (const source of sources) {
        const item = document.createElement("div");
        item.className = "source";
        const score = typeof source.score === "number" ? source.score.toFixed(3) : "n/a";
        item.innerHTML = "<strong></strong><small></small><p></p>";
        item.querySelector("strong").textContent = source.function_name;
        item.querySelector("small").textContent = `${source.library_name} · score ${score}`;
        item.querySelector("p").textContent = source.snippet || "";
        sourcesEl.appendChild(item);
      }
    }

    form.addEventListener("submit", async (event) => {
      event.preventDefault();
      const query = queryEl.value.trim();
      const selectedMode = document.querySelector('input[name="mode"]:checked');
      const mode = selectedMode ? selectedMode.value : "fast";
      if (!query) return;

      addMessage("You", query);
      const botBubble = addMessage("Assistant", "Thinking...");
      sendEl.disabled = true;
      metaEl.textContent = "";
      const start = performance.now();

      try {
        const response = await fetch("/ask", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ query, mode }),
        });
        const data = await response.json();
        if (!response.ok) throw new Error(data.detail || "Request failed");

        botBubble.textContent = data.answer;
        renderSources(data.sources || []);
        const seconds = ((performance.now() - start) / 1000).toFixed(1);
        metaEl.textContent = `${mode.toUpperCase()} · ${seconds}s`;
      } catch (error) {
        botBubble.textContent = error.message;
        botBubble.parentElement.classList.add("error");
      } finally {
        sendEl.disabled = false;
      }
    });
  </script>
</body>
</html>
"""


app = FastAPI(
    title="Mini Code-Assistance RAG",
    version="0.1.0",
)
_pipeline_lock = Lock()


class LazyLLM(BaseLLM):
    def __init__(self, factory: Callable[[], BaseLLM]):
        self.factory = factory
        self.model: Optional[BaseLLM] = None

    def _get_model(self) -> BaseLLM:
        if self.model is None:
            self.model = self.factory()
        return self.model

    def parse_query(self, raw_query: str) -> ParsedQuery:
        return self._get_model().parse_query(raw_query)

    def generate_answer(self, query: str, context_chunks: List[Chunk]) -> str:
        return self._get_model().generate_answer(query, context_chunks)


class AskRequest(BaseModel):
    query: str = Field(..., min_length=1)
    mode: str = Field("fast", pattern="^(fast|full)$")


def get_db_config() -> dict:
    return {
        "dbname": os.getenv("POSTGRES_DB", "rag_database"),
        "user": os.getenv("POSTGRES_USER", "admin"),
        "password": os.getenv("POSTGRES_PASSWORD", "secretpassword"),
        "host": os.getenv("POSTGRES_HOST", "localhost"),
        "port": int(os.getenv("POSTGRES_PORT", "5432")),
    }


@lru_cache(maxsize=1)
def get_pipeline() -> RAGPipeline:
    print("Loading AI Models into memory...")
    parser_model_path = os.getenv(
        "PARSER_MODEL_PATH",
        "models/final_gguf/qwen3-0.6b-instruct-q4_k_m.gguf",
    )
    generator_model_path = os.getenv(
        "GENERATOR_MODEL_PATH",
        "models/final_gguf/qwen3-4b-instruct-q4_k_m.gguf",
    )

    return RAGPipeline(
        parser_llm=LazyLLM(lambda: MicroParserLLM(model_path=parser_model_path)),
        retriever=PGHybridRetriever(db_config=get_db_config()),
        reranker=BGEReranker(),
        generator_llm=LazyLLM(lambda: QwenLocalLLM(model_path=generator_model_path)),
        retrieval_k=int(os.getenv("RETRIEVAL_K", "12")),
        context_k=int(os.getenv("CONTEXT_K", "4")),
    )


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    if INDEX_HTML_PATH.exists():
        return INDEX_HTML_PATH.read_text(encoding="utf-8")
    return INDEX_HTML


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/ask", response_model=RAGResponse)
def ask(request: AskRequest) -> RAGResponse:
    try:
        # llama-cpp model instances are not safe to drive concurrently from
        # multiple request threads, so serialize generation for this demo app.
        with _pipeline_lock:
            return get_pipeline().run_with_details(
                request.query,
                generate_answer=request.mode == "full",
            )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


def run_demo() -> None:
    query = "Làm sao để kết hợp 2 dataframes dựa trên một cột chung trong thư viện pandas?"
    response = get_pipeline().run_with_details(query)
    print("ANSWER:")
    print(response.answer)
    print("\nSOURCES:")
    for source in response.sources:
        print(f"- {source.function_name} score={source.score:.4f}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--demo", action="store_true", help="Run one terminal demo query.")
    parser.add_argument("--host", default=os.getenv("HOST", "127.0.0.1"))
    parser.add_argument("--port", type=int, default=int(os.getenv("PORT", "8000")))
    args = parser.parse_args()

    if args.demo:
        run_demo()
        return

    import uvicorn

    uvicorn.run("api.main:app", host=args.host, port=args.port, reload=False)


if __name__ == "__main__":
    main()
