from typing import List
from core.reranker.base import BaseReranker
from core.schemas import Chunk
from utils.config_loader import GLOBAL_CONFIG
from utils.logger import get_logger

logger = get_logger(__name__)


class DocumentReranker:
    """Agent responsible for filtering and selecting the highest quality documents."""

    def __init__(self, reranker_model: BaseReranker):
        """
        Initializes the reranker agent with a core reranker model.
        """
        self.reranker_model = reranker_model

        # Dynamically load the target number of chunks (top_k) from the global config
        self.default_top_k = (
            GLOBAL_CONFIG.get("models", {}).get("reranker", {}).get("top_k", 5)
        )

    def process(self, query: str, candidate_chunks: List[Chunk]) -> List[Chunk]:
        """
        Executes the reranking pipeline and applies business-specific filtering logic.

        Args:
            query (str): The optimized semantic query.
            candidate_chunks (List[Chunk]): The initial raw chunks retrieved from the database.

        Returns:
            List[Chunk]: The refined, reranked, and sorted list of chunks.
        """
        logger.info(
            f"Agent starting reranking process for {len(candidate_chunks)} candidates..."
        )

        if not candidate_chunks:
            logger.warning("No candidate chunks provided to rerank.")
            return []

        # 1. Delegate the heavy mathematical scoring to the core model
        best_chunks = self.reranker_model.rerank(
            query=query, chunks=candidate_chunks, top_k=self.default_top_k
        )

        # 2. Filter chunk
        best_chunks = [chunk for chunk in best_chunks if chunk.score > 0.0]

        logger.info(f"Agent finished reranking. Kept top {len(best_chunks)} chunks.")
        return best_chunks
