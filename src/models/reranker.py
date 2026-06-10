# src/models/reranker.py
from typing import List
from core.reranker.base import BaseReranker
from core.schemas import Chunk
from utils.config_loader import GLOBAL_CONFIG
from utils.logger import get_logger

logger = get_logger(__name__)

class DocumentReranker:
    """Agent responsible for filtering and selecting the highest quality documents."""

    def __init__(self, reranker_model: BaseReranker):
        self.reranker_model = reranker_model
        # Load default top_k from config
        self.default_top_k = GLOBAL_CONFIG.get("reranker", {}).get("top_k", 5)

    def process(self, query: str, candidate_chunks: List[Chunk]) -> List[Chunk]:
        """Scores candidates and filters out low-relevance chunks."""
        logger.info(f"Starting reranking process for {len(candidate_chunks)} candidates...")

        if not candidate_chunks:
            logger.warning("No candidate chunks provided to rerank.")
            return []

        # Delegate scoring to the core model
        best_chunks = self.reranker_model.rerank(
            query=query, 
            chunks=candidate_chunks, 
            top_k=self.default_top_k
        )

        # Filter out chunks with scores below threshold
        best_chunks = [chunk for chunk in best_chunks if chunk.score is not None and chunk.score > 0.0]

        logger.info(f"Finished reranking. Kept top {len(best_chunks)} chunks.")
        return best_chunks