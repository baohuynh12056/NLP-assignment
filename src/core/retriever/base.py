# src/core/retriever/base.py
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from core.schemas import Chunk

class BaseRetriever(ABC):
    """Abstract interface for database retrieval implementations."""
    
    @abstractmethod
    def retrieve(self, query: str, top_k: int = 10, filters: Optional[Dict[str, Any]] = None) -> List[Chunk]:
        pass