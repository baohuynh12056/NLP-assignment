from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any
from .schemas import Chunk

class BaseRetriever(ABC):
    """
    Enforces a standard contract for retrieving documents.
    """
    
    @abstractmethod
    def retrieve(self, query: str, top_k: int = 10, filters: Optional[Dict[str, Any]] = None) -> List[Chunk]:
        """
        Perform the retrieval operation and return a list of relevant chunks.
        
        Args:
            query (str): The optimized search query.
            top_k (int): The maximum number of chunks to retrieve.
            filters (dict, optional): Metadata filters to apply during retrieval 
                                      (e.g., {"library_name": "pandas"}).
            
        Returns:
            List[Chunk]: A list of retrieved document chunks with their initial scores.
        """
        pass