import re
import unicodedata

from core.llm.base import BaseLLM
from utils.config_loader import GLOBAL_CONFIG
from utils.logger import get_logger

# Initialize module logger
logger = get_logger(__name__)


class AnswerGenerator:
    """Agent responsible for reading retrieved documents and generating user-friendly answers."""

    def __init__(self, llm: BaseLLM):
        self.llm = llm
        generator_cfg = GLOBAL_CONFIG["prompts"]["generator"]
        self.system_prompt = generator_cfg["system"]
        self.context_template = generator_cfg["context_template"]
        self.user_template = generator_cfg["user_template"]
        self.max_chunk_chars = int(generator_cfg.get("max_chunk_chars", 1400))
        self.no_think = bool(generator_cfg.get("no_think", True))

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

        messages = self._build_messages(query, context_chunks)

        # Call the LLM to generate the final response
        # Note: temperature and max_tokens are handled by the BaseLLM configuration
        final_answer = self._strip_thinking(self.llm.chat_completion(messages))

        logger.info("Final answer generated successfully.")
        return final_answer

    def stream(self, query: str, context_chunks: list):
        """Streams answer chunks from the configured LLM."""
        logger.info(
            f"Streaming answer using {len(context_chunks)} retrieved chunks..."
        )

        if not context_chunks:
            yield "I'm sorry, I cannot find any relevant documentation to answer your query."
            return

        messages = self._build_messages(query, context_chunks)
        for chunk in self.llm.stream_chat_completion(messages):
            yield chunk

    def generate_followup(self, query: str, previous_response) -> str:
        """Generates an answer for a follow-up request using the previous turn."""
        messages = self._build_followup_messages(query, previous_response)
        return self._strip_thinking(self.llm.chat_completion(messages))

    def stream_followup(self, query: str, previous_response):
        """Streams an answer for a follow-up request using the previous turn."""
        messages = self._build_followup_messages(query, previous_response)
        for chunk in self.llm.stream_chat_completion(messages):
            yield chunk

    def _build_messages(self, query: str, context_chunks: list) -> list[dict[str, str]]:
        context_str = self._build_context(context_chunks)
        user_message = self.user_template.format(
            context=context_str,
            query=query,
            language=self._detect_language(query),
        )
        if self.no_think:
            user_message += "\n/no_think"

        return [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_message},
        ]

    def _build_followup_messages(self, query: str, previous_response) -> list[dict[str, str]]:
        source_lines = []
        for index, source in enumerate(previous_response.sources, start=1):
            source_lines.append(
                (
                    f"[{index}] {source.library_name}.{source.function_name}\n"
                    f"Snippet: {source.snippet}"
                )
            )

        user_message = (
            "The user is continuing the previous conversation.\n"
            "Answer the follow-up using the previous question, previous answer, "
            "and previous sources below. Do not start an unrelated retrieval topic.\n\n"
            f"Previous question:\n{previous_response.query}\n\n"
            f"Previous answer:\n{previous_response.answer}\n\n"
            f"Previous sources:\n{chr(10).join(source_lines) or 'No sources'}\n\n"
            f"Follow-up request:\n{query}\n\n"
            "If the follow-up asks for more detail, expand the explanation with "
            "practical usage, parameters, caveats, and a short code example when useful. "
            f"Answer in {self._detect_language(query + ' ' + previous_response.query)}."
        )
        if self.no_think:
            user_message += "\n/no_think"

        return [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_message},
        ]

    def _build_context(self, context_chunks: list) -> str:
        blocks = []
        for index, chunk in enumerate(context_chunks, start=1):
            metadata = chunk.metadata if hasattr(chunk, "metadata") else chunk.get("metadata", {})
            content = chunk.content if hasattr(chunk, "content") else chunk.get("content", "")
            content = self._trim_doc(content)
            blocks.append(
                self.context_template.format(
                    index=index,
                    function_name=metadata.get("func_name", "Unknown"),
                    library_name=metadata.get("library_name", "Unknown"),
                    parameters=metadata.get("parameters", {}),
                    documentation=content,
                )
            )
        return "\n\n---\n\n".join(blocks)

    def _trim_doc(self, doc: str) -> str:
        doc = (doc or "").strip()
        if len(doc) <= self.max_chunk_chars:
            return doc
        return doc[: self.max_chunk_chars].rsplit("\n", 1)[0].strip()

    @staticmethod
    def _detect_language(text: str) -> str:
        normalized = unicodedata.normalize("NFKD", text or "")
        normalized = "".join(ch for ch in normalized if not unicodedata.combining(ch)).lower()
        vietnamese_terms = [
            "cach",
            "doc",
            "bang",
            "nhu the nao",
            "lam sao",
            "ket hop",
            "chia",
            "tach",
            "du lieu",
            "thu vien",
        ]
        if any(term in normalized for term in vietnamese_terms):
            return "Vietnamese"
        return "English"

    @staticmethod
    def _strip_thinking(text: str) -> str:
        return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
