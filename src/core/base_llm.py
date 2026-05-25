from abc import ABC, abstractmethod
from typing import List
from .schemas import ParsedQuery, Chunk

class BaseLLM(ABC):
    """
    Defines the two main LLM responsibilities in this RAG architecture.
    """
    
    @abstractmethod
    def parse_query(self, raw_query: str) -> ParsedQuery:
        """
        Analyze the raw user query to extract metadata filters and rewrite it for better retrieval.
        
        Args:
            raw_query (str): The initial query provided by the user.
            
        Returns:
            ParsedQuery: An object containing the rewritten query and extracted metadata filters.
        """
        pass

    @abstractmethod
    def generate_answer(self, query: str, context_chunks: List[Chunk]) -> str:
        """
        Generate the final answer based on the provided context chunks.
        
        Args:
            query (str): The user's specific question.
            context_chunks (List[Chunk]): The top-k reranked chunks containing relevant documentation.
            
        Returns:
            str: The final generated answer, complete with explanations and code snippets.
        """
        pass