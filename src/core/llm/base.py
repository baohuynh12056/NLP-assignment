from abc import ABC, abstractmethod
from typing import List, Dict


class BaseLLM(ABC):
    """
    Standard interface for all LLM models.
    Ensures polymorphism across different LLM backends.
    """

    @abstractmethod
    def chat_completion(self, messages: List[Dict[str, str]], **kwargs) -> str:
        """Takes a conversation history and returns a generated text string."""
        pass
