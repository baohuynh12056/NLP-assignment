from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from core.schemas import Chunk


class BaseRetriever(ABC):
    """
    Standard interface for all retrieval engines (PostgreSQL, Elasticsearch, FAISS, etc.).
    Ensures that any retriever returns a standard list of Chunks.
    """

    @abstractmethod
    def retrieve(
        self, query: str, top_k: int = 10, filters: Optional[Dict[str, Any]] = None
    ) -> List[Chunk]:
        """
        Executes the search and returns a list of relevant document chunks.
        """
        pass
