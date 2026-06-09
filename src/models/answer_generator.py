from core.llm.base import BaseLLM
from utils.config_loader import GLOBAL_CONFIG
from utils.logger import get_logger

# Initialize module logger
logger = get_logger(__name__)


class AnswerGenerator:
    """Agent responsible for reading retrieved documents and generating user-friendly answers."""

    def __init__(self, llm: BaseLLM):
        self.llm = llm
        # Load the generator system prompt from global configuration
        self.system_prompt = GLOBAL_CONFIG["prompts"]["generator"]["system"]

    def generate(self, query: str, context_chunks: list) -> str:
        """
        Generates a comprehensive answer based on the retrieved context.

        Args:
            query (str): The user's specific question.
            context_chunks (list): A list of dictionaries representing the retrieved functions.

        Returns:
            str: The final generated explanation and code examples.
        """
        logger.info(
            f"Generating answer using {len(context_chunks)} retrieved chunks..."
        )

        if not context_chunks:
            logger.warning("No context chunks provided. Generating fallback response.")
            return "I'm sorry, I cannot find any relevant documentation to answer your query."

        # Format the retrieved context chunks into a clean, readable string
        context_str = "\n\n---\n\n".join(
            [
                f"Function: {c.metadata.get('func_name', 'Unknown')}\n"
                f"Docstring: {c.content}"
                for c in context_chunks
            ]
        )

        user_message = f"Documents:\n{context_str}\n\nQuestion: {query}"

        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_message},
        ]

        # Call the LLM to generate the final response
        # Note: temperature and max_tokens are handled by the BaseLLM configuration
        final_answer = self.llm.chat_completion(messages)

        logger.info("Final answer generated successfully.")
        return final_answer
