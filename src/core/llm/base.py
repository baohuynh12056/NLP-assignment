# src/core/llm/base.py
from abc import ABC, abstractmethod
from typing import List, Dict

class BaseLLM(ABC):
    """Abstract interface for all Language Models."""
    
    @abstractmethod
    def chat_completion(self, messages: List[Dict[str, str]], **kwargs) -> str:
        """
        Takes a list of conversation messages (OpenAI format) and returns a string response.
        """
        pass