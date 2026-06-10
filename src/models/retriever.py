# src/models/retriever.py
from typing import List
from core.retriever.base import BaseRetriever
from core.schemas import Chunk, ParsedQuery
from utils.logger import get_logger

logger = get_logger(__name__)

class DocumentRetriever:
    """Agent responsible for communicating with the database to fetch candidate chunks."""

    def __init__(self, retriever_model: BaseRetriever):
        """Injects the core DB implementation."""
        self.retriever_model = retriever_model

    def process(self, parsed_query: ParsedQuery, top_k: int = 20) -> List[Chunk]:
        """Executes the database retrieval using the parsed query."""
        logger.info(f"Starting retrieval for optimized query: '{parsed_query.optimized_query}'")

        candidates = self.retriever_model.retrieve(
            query=parsed_query.optimized_query,
            top_k=top_k,
            filters=parsed_query.filters,
        )

        if not candidates:
            logger.warning("Found 0 candidates in the database.")
        else:
            logger.info(f"Retrieved {len(candidates)} raw candidates.")

        return candidates