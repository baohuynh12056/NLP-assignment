# src/core/reranker/base.py
from abc import ABC, abstractmethod
from typing import List
from core.schemas import Chunk

class BaseReranker(ABC):
    """Abstract interface for document reranking models."""
    
    @abstractmethod
    def rerank(self, query: str, chunks: List[Chunk], top_k: int = 5) -> List[Chunk]:
        pass