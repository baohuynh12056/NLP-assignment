from abc import ABC, abstractmethod
from typing import List
from core.schemas import Chunk


class BaseReranker(ABC):
    """
    Standard interface for all cross-encoder reranking models.
    Ensures polymorphism across different reranker backends (e.g., BGE, Cohere).
    """

    @abstractmethod
    def rerank(self, query: str, chunks: List[Chunk], top_k: int = 5) -> List[Chunk]:
        """
        Re-evaluates and sorts chunks based on deep semantic relevance to the query.
        """
        pass
