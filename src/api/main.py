import argparse
import sys
from functools import lru_cache
from pathlib import Path
from threading import Lock

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from core.llm.manager import LLMManager
from core.reranker.factory import RerankerFactory
from core.retriever.factory import RetrieverFactory
from core.schemas import RAGResponse
from models.answer_generator import AnswerGenerator
from models.query_parser import QueryParser
from models.reranker import DocumentReranker
from models.retriever import DocumentRetriever
from pipeline.rag_orchestrator import RAGOrchestrator
from utils.config_loader import GLOBAL_CONFIG
from utils.logger import get_logger


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")


logger = get_logger("api")
INDEX_HTML_PATH = Path(__file__).with_name("chat_ui.html")
_pipeline_lock = Lock()

app = FastAPI(
    title="Mini Code-Assistance RAG",
    version="0.2.0",
)


class AskRequest(BaseModel):
    query: str = Field(..., min_length=1)
    mode: str = Field("fast", pattern="^(fast|full)$")


def initialize_orchestrator() -> RAGOrchestrator:
    """Initializes models, retrieval components, business agents, and orchestrator."""
    logger.info("Initializing system components...")

    llm_manager = LLMManager()
    core_retriever = RetrieverFactory.create_retriever(
        GLOBAL_CONFIG.get("retriever", {})
    )
    core_reranker = RerankerFactory.create_reranker(
        GLOBAL_CONFIG.get("models", {}).get("reranker", {})
    )

    return RAGOrchestrator(
        query_parser=QueryParser(llm=llm_manager.parser_llm),
        retriever=DocumentRetriever(retriever_model=core_retriever),
        reranker=DocumentReranker(reranker_model=core_reranker),
        answer_generator=AnswerGenerator(llm=llm_manager.generator_llm),
    )


@lru_cache(maxsize=1)
def get_pipeline() -> RAGOrchestrator:
    return initialize_orchestrator()


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    if INDEX_HTML_PATH.exists():
        return INDEX_HTML_PATH.read_text(encoding="utf-8")
    return "<h1>Code Assistant</h1><p>Missing src/api/chat_ui.html</p>"


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/ask", response_model=RAGResponse)
def ask(request: AskRequest) -> RAGResponse:
    try:
        # llama.cpp model instances should not be driven concurrently in this demo.
        with _pipeline_lock:
            return get_pipeline().run_with_details(
                request.query,
                generate_answer=request.mode == "full",
            )
    except Exception as exc:
        logger.exception("Failed to answer request")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


def run_demo() -> None:
    query = "How do I merge two pandas DataFrames on a common column?"
    response = get_pipeline().run_with_details(query, generate_answer=False)
    print("ANSWER:")
    print(response.answer)
    print("\nSOURCES:")
    for source in response.sources:
        print(f"- {source.function_name} score={source.score}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--demo", action="store_true", help="Run one terminal demo query.")
    args = parser.parse_args()

    if args.demo:
        run_demo()
        return

    import uvicorn

    uvicorn.run("api.main:app", host="127.0.0.1", port=8000, reload=False)


if __name__ == "__main__":
    main()
