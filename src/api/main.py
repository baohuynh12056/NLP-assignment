# src/main.py
import sys
from pathlib import Path
from threading import Lock
from typing import Dict, Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

# --- Tầng Core (Hạ tầng) ---
from core.llm.manager import LLMManager
from core.retriever.factory import RetrieverFactory
from core.reranker.factory import RerankerFactory

# --- Tầng Models (Agents & Pipeline) ---
from models.query_parser import QueryParser
from models.retriever import DocumentRetriever
from models.reranker import DocumentReranker
from models.answer_generator import AnswerGenerator
from pipeline.rag_orchestrator import RAGOrchestrator

from utils.config_loader import GLOBAL_CONFIG
from utils.logger import get_logger

# Fix encoding cho console Windows/Docker
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

logger = get_logger(__name__)

# Initialize FastAPI App
app = FastAPI(title="Semantic Function Search API")

# Llama.cpp isn't completely thread-safe for concurrent generation without specific setups.
# This lock ensures we don't crash the GPU when multiple requests hit at the same time.
_pipeline_lock = Lock()
_pipeline_instance = None

INDEX_HTML_PATH = Path(__file__).parent / "chat_ui.html"

# --- Request Schemas ---
class AskRequest(BaseModel):
    query: str
    mode: str = "fast"


def get_pipeline() -> RAGOrchestrator:
    """Instantiates the entire system pipeline exactly once (Singleton pattern)."""
    global _pipeline_instance
    if _pipeline_instance is not None:
        return _pipeline_instance

    logger.info("Initializing System Dependencies...")
    
    # 1. Initialize Core Infrastructure from Configs
    models_cfg = GLOBAL_CONFIG.get("models", {})
    
    llm_manager = LLMManager()
    
    core_retriever = RetrieverFactory.create_retriever(
        models_cfg.get("retriever", {})
    )
    
    core_reranker = RerankerFactory.create_reranker(
        models_cfg.get("reranker", {})
    )

    # 2. Inject Core Models into Business Agents
    parser_agent = QueryParser(llm=llm_manager.parser_llm)
    retriever_agent = DocumentRetriever(retriever_model=core_retriever)
    reranker_agent = DocumentReranker(reranker_model=core_reranker)
    generator_agent = AnswerGenerator(llm=llm_manager.generator_llm)

    # 3. Assemble the RAG Orchestrator
    _pipeline_instance = RAGOrchestrator(
        query_parser=parser_agent,
        retriever=retriever_agent,
        reranker=reranker_agent,
        answer_generator=generator_agent
    )
    
    logger.info("System Pipeline successfully assembled!")
    return _pipeline_instance


# --- API Endpoints ---

@app.on_event("startup")
def on_startup():
    """Tải models vào RAM/VRAM ngay khi bật server thay vì đợi request đầu tiên."""
    get_pipeline()


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/", response_class=HTMLResponse)
def serve_ui():
    """Phục vụ giao diện Chat UI."""
    if not INDEX_HTML_PATH.exists():
        return HTMLResponse("<html><body><h1>chat_ui.html not found</h1></body></html>", status_code=404)
    return HTMLResponse(INDEX_HTML_PATH.read_text(encoding="utf-8"))


@app.post("/ask")
def ask(request: AskRequest) -> Dict[str, Any]:
    """Endpoint xử lý câu hỏi của người dùng."""
    try:
        with _pipeline_lock:
            pipeline = get_pipeline()
            # Pipeline orchestrator trả về dict: {"answer": str, "sources": list}
            return pipeline.run(user_query=request.query)
            
    except Exception as exc:
        logger.error(f"API Error during processing: {str(exc)}")
        raise HTTPException(status_code=500, detail=str(exc)) from exc