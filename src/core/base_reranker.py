from abc import ABC, abstractmethod
from typing import List
from .schemas import Chunk

class BaseReranker(ABC):
    """
    Abstract base class for cross-encoder or reranking models.
    Used to refine and re-score the initial retrieval results.
    """
    
    @abstractmethod
    def rerank(self, query: str, chunks: List[Chunk], top_k: int = 5) -> List[Chunk]:
        """
        Re-evaluate and sort the retrieved chunks based on their deep contextual relevance.
        
        Args:
            query (str): The original or optimized user query.
            chunks (List[Chunk]): The initial list of chunks retrieved by the base retriever.
            top_k (int): The number of top-scoring chunks to keep after reranking.
            
        Returns:
            List[Chunk]: The reranked list of chunks, sorted by relevance score in descending order.
        """
        pass