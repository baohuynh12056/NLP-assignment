# src/models/answer_generator.py
import re
from typing import List
from core.llm.base import BaseLLM
from core.schemas import Chunk
from utils.config_loader import GLOBAL_CONFIG
from utils.logger import get_logger

logger = get_logger(__name__)

class AnswerGenerator:
    """Agent responsible for reading retrieved documents and generating user-friendly answers."""

    def __init__(self, llm: BaseLLM):
        self.llm = llm
        # Load the generator system prompt from global configuration
        self.system_prompt = GLOBAL_CONFIG.get("prompts", {}).get("generator", {}).get("system", "")

    def generate(self, query: str, context_chunks: List[Chunk]) -> str:
        """Builds context and queries the LLM for the final answer."""
        logger.info(f"Generating answer using {len(context_chunks)} retrieved chunks...")

        if not context_chunks:
            logger.warning("No context chunks provided. Generating fallback response.")
            return "I'm sorry, I cannot find any relevant documentation to answer your query."

        # Format the retrieved context chunks
        context_str = "\n\n---\n\n".join(
            [
                f"Function: {c.metadata.get('func_name', 'Unknown')}\n"
                f"Library: {c.metadata.get('library_name', 'Unknown')}\n"
                f"Parameters: {c.metadata.get('parameters', {})}\n"
                f"Docstring:\n{c.content}"
                for c in context_chunks
            ]
        )

        # Build user message with context and query
        user_message = (
            f"Retrieved documentation:\n{context_str}\n\n"
            f"User question: {query}\n\n"
            "/no_think"
        )

        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_message},
        ]

        # Call the LLM to generate the final response
        raw_answer = self.llm.chat_completion(messages)
        
        # Clean up <think> blocks if using specific reasoning models
        final_answer = self._strip_thinking(raw_answer)

        logger.info("Final answer generated successfully.")
        return final_answer

    @staticmethod
    def _strip_thinking(text: str) -> str:
        """Removes reasoning blocks from the model's output."""
        return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()