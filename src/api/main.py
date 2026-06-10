import argparse
import json
import sys
import time
from collections import OrderedDict
from functools import lru_cache
from pathlib import Path
from threading import Lock
from typing import Iterator

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, StreamingResponse
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
_cache_lock = Lock()
_response_cache: OrderedDict[str, RAGResponse] = OrderedDict()
CACHE_SIZE = int(GLOBAL_CONFIG.get("retriever", {}).get("cache_size", 128))

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


def _cache_key(mode: str, query: str) -> str:
    normalized_query = " ".join((query or "").strip().lower().split())
    return f"{mode}:{normalized_query}"


def _cache_get(mode: str, query: str) -> RAGResponse | None:
    key = _cache_key(mode, query)
    with _cache_lock:
        cached = _response_cache.get(key)
        if cached is None:
            return None
        _response_cache.move_to_end(key)
        return cached


def _cache_set(mode: str, query: str, response: RAGResponse) -> None:
    if CACHE_SIZE <= 0:
        return
    key = _cache_key(mode, query)
    with _cache_lock:
        _response_cache[key] = response
        _response_cache.move_to_end(key)
        while len(_response_cache) > CACHE_SIZE:
            _response_cache.popitem(last=False)


def _response_payload(response: RAGResponse) -> dict:
    if hasattr(response, "model_dump"):
        return response.model_dump()
    return response.dict()


def _json_line(payload: dict) -> str:
    return json.dumps(payload, ensure_ascii=False) + "\n"


def _stream_text(text: str, size: int = 28) -> Iterator[str]:
    for index in range(0, len(text), size):
        chunk = text[index : index + size]
        if chunk:
            yield chunk


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
        cached = _cache_get(request.mode, request.query)
        if cached is not None:
            return cached

        # llama.cpp model instances should not be driven concurrently in this demo.
        with _pipeline_lock:
            response = get_pipeline().run_with_details(
                request.query,
                generate_answer=request.mode == "full",
            )
            _cache_set(request.mode, request.query, response)
            return response
    except Exception as exc:
        logger.exception("Failed to answer request")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/ask/stream")
def ask_stream(request: AskRequest) -> StreamingResponse:
    def events() -> Iterator[str]:
        started_at = time.perf_counter()
        cached = _cache_get(request.mode, request.query)
        if cached is not None:
            yield _json_line({"type": "meta", "cached": True})
            for chunk in _stream_text(cached.answer):
                yield _json_line({"type": "token", "text": chunk})
            yield _json_line(
                {
                    "type": "done",
                    "response": _response_payload(cached),
                    "elapsed": round(time.perf_counter() - started_at, 2),
                    "cached": True,
                }
            )
            return

        try:
            with _pipeline_lock:
                pipeline = get_pipeline()
                parsed_query, chunks = pipeline.retrieve_context(
                    request.query,
                    generate_answer=request.mode == "full",
                )

                if not chunks:
                    answer = "I'm sorry, I couldn't find any relevant functions in the database to answer your query."
                    response = pipeline.build_response(request.query, parsed_query, chunks, answer)
                    _cache_set(request.mode, request.query, response)
                    yield _json_line({"type": "token", "text": answer})
                    yield _json_line(
                        {
                            "type": "done",
                            "response": _response_payload(response),
                            "elapsed": round(time.perf_counter() - started_at, 2),
                            "cached": False,
                        }
                    )
                    return

                if request.mode == "fast":
                    answer = pipeline._build_fast_answer(chunks)
                    response = pipeline.build_response(request.query, parsed_query, chunks, answer)
                    _cache_set(request.mode, request.query, response)
                    for chunk in _stream_text(answer):
                        yield _json_line({"type": "token", "text": chunk})
                else:
                    answer_parts = []
                    for chunk in pipeline.answer_generator.stream(request.query, chunks):
                        answer_parts.append(chunk)
                        yield _json_line({"type": "token", "text": chunk})
                    answer = pipeline.answer_generator._strip_thinking("".join(answer_parts))
                    response = pipeline.build_response(request.query, parsed_query, chunks, answer)
                    _cache_set(request.mode, request.query, response)

                yield _json_line(
                    {
                        "type": "done",
                        "response": _response_payload(response),
                        "elapsed": round(time.perf_counter() - started_at, 2),
                        "cached": False,
                    }
                )
        except Exception as exc:
            logger.exception("Failed to stream answer")
            yield _json_line({"type": "error", "detail": str(exc)})

    return StreamingResponse(events(), media_type="application/x-ndjson")


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
