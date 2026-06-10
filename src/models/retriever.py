from typing import List
from core.retriever.base import BaseRetriever
from core.schemas import Chunk, ParsedQuery
from utils.config_loader import GLOBAL_CONFIG
from utils.logger import get_logger

# Initialize module logger
logger = get_logger(__name__)


class DocumentRetriever:
    """Agent responsible for communicating with the database to fetch candidate chunks."""

    def __init__(self, retriever_model: BaseRetriever):
        """
        Initializes the retriever agent with a core DB implementation.
        """
        self.retriever_model = retriever_model
        retriever_cfg = GLOBAL_CONFIG.get("retriever", {})
        self.default_top_k = int(retriever_cfg.get("retrieval_k", 20))

    def process(self, parsed_query: ParsedQuery, top_k: int | None = None) -> List[Chunk]:
        """
        Executes the database retrieval using the parsed query and metadata filters.

        Args:
            parsed_query (ParsedQuery): The output from the QueryParser (contains optimized query and filters).
            top_k (int): Number of initial candidates to fetch (usually higher than reranker top_k).

        Returns:
            List[Chunk]: The raw list of candidate chunks from the database.
        """
        logger.info(
            f"Agent starting retrieval for optimized query: '{parsed_query.optimized_query}'"
        )

        # Execute the core retrieval operation
        candidates = self.retriever_model.retrieve(
            query=parsed_query.optimized_query,
            top_k=top_k or self.default_top_k,
            filters=parsed_query.filters,
        )

        if not candidates:
            logger.warning("Agent found 0 candidates in the database.")

        return candidates
