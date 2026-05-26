import json
from core.llm.base import BaseLLM
from utils.config_loader import GLOBAL_CONFIG
from utils.logger import get_logger

# Initialize module logger
logger = get_logger(__name__)


class QueryParser:
    """Agent responsible for optimizing user queries and extracting metadata filters."""

    def __init__(self, llm: BaseLLM):
        self.llm = llm
        # Load the parser system prompt from global configuration
        self.system_prompt = GLOBAL_CONFIG["prompts"]["parser"]["system"]

    def parse(self, raw_query: str) -> dict:
        """
        Parses the raw query to extract filters and an optimized semantic query.

        Args:
            raw_query (str): The initial query provided by the user.

        Returns:
            dict: A dictionary containing 'optimized_query' and 'filters'.
        """
        logger.info(f"Parsing raw query: '{raw_query}'")

        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": raw_query},
        ]

        # Force temperature to 0.0 for deterministic JSON extraction
        raw_output = self.llm.chat_completion(messages, temperature=0.0)

        try:
            # Clean up Markdown JSON formatting if present
            if raw_output.startswith("```json"):
                raw_output = raw_output[7:-3].strip()

            parsed_data = json.loads(raw_output)
            logger.info(
                f"Successfully extracted filters: {parsed_data.get('filters', {})}"
            )
            return parsed_data

        except json.JSONDecodeError:
            # Safe fallback if the LLM hallucinated or failed to output valid JSON
            logger.warning(
                f"Failed to parse JSON. Fallback applied. Raw output: {raw_output}"
            )
            return {"optimized_query": raw_query, "filters": {}}
