from abc import ABC, abstractmethod
from typing import Dict, Iterator, List


class BaseLLM(ABC):
    """
    Standard interface for all LLM models.
    Ensures polymorphism across different LLM backends.
    """

    @abstractmethod
    def chat_completion(self, messages: List[Dict[str, str]], **kwargs) -> str:
        """Takes a conversation history and returns a generated text string."""
        pass

    def stream_chat_completion(
        self,
        messages: List[Dict[str, str]],
        **kwargs,
    ) -> Iterator[str]:
        """Streams generated text chunks. Backends can override this for true token streaming."""
        yield self.chat_completion(messages, **kwargs)
