import json
import unicodedata

from core.llm.base import BaseLLM
from core.schemas import ParsedQuery
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
        self.supported_libraries = [
            "pandas",
            "numpy",
            "sklearn",
            "torch",
            "tensorflow",
            "matplotlib",
            "scipy",
            "seaborn",
            "requests",
            "fastapi",
        ]

    def parse(self, raw_query: str) -> ParsedQuery:
        """
        Parses the raw query to extract filters and an optimized semantic query.

        Args:
            raw_query (str): The initial query provided by the user.

        Returns:
            dict: A dictionary containing 'optimized_query' and 'filters'.
        """
        logger.info(f"Parsing raw query: '{raw_query}'")

        parsed_data = self._rule_based_parse(raw_query)
        if parsed_data is None:
            messages = [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": raw_query},
            ]

            # Force temperature to 0.0 for deterministic JSON extraction
            raw_output = self.llm.chat_completion(messages, temperature=0.0)
            parsed_data = self._parse_json_output(raw_output, raw_query)

        parsed_data.optimized_query = self._fallback_rewrite(
            raw_query,
            parsed_data.optimized_query,
        )
        parsed_data.filters = self._validate_filters(raw_query, parsed_data.filters)
        logger.info(f"Final optimized query: '{parsed_data.optimized_query}'")
        logger.info(f"Validated filters: {parsed_data.filters}")
        return parsed_data

    def _parse_json_output(self, raw_output: str, raw_query: str) -> ParsedQuery:
        try:
            raw_output = raw_output.strip()
            if raw_output.startswith("```json"):
                raw_output = raw_output[7:-3].strip()

            parsed_data = json.loads(raw_output)
            return ParsedQuery(
                optimized_query=parsed_data.get("optimized_query", raw_query),
                filters=parsed_data.get("filters") or {},
            )
        except json.JSONDecodeError:
            logger.warning(
                f"Failed to parse JSON. Fallback applied. Raw output: {raw_output}"
            )
            return ParsedQuery(optimized_query=raw_query, filters={})

    def _rule_based_parse(self, raw_query: str) -> ParsedQuery | None:
        inferred_library = self._infer_library(raw_query)
        optimized_query = self._fallback_rewrite(raw_query, raw_query)
        if optimized_query != raw_query:
            return ParsedQuery(
                optimized_query=optimized_query,
                filters={"library_name": inferred_library} if inferred_library else {},
            )

        normalized = self._normalize_text(raw_query)
        if inferred_library and any(
            term in normalized for term in ["merge", "join", "concat", "dataframe"]
        ):
            return ParsedQuery(
                optimized_query=raw_query,
                filters={"library_name": inferred_library},
            )
        return None

    def _validate_filters(self, raw_query: str, filters: dict) -> dict:
        validated = {}
        if not isinstance(filters, dict):
            filters = {}

        if "library_name" not in filters:
            inferred_library = self._infer_library(raw_query)
            if inferred_library:
                filters["library_name"] = inferred_library

        library_name = filters.get("library_name")
        if library_name:
            library_name = str(library_name).strip().lower()
            if library_name in self.supported_libraries:
                validated["library_name"] = library_name
        return validated

    def _infer_library(self, raw_query: str) -> str | None:
        normalized = self._normalize_text(raw_query)
        for library_name in self.supported_libraries:
            if library_name in normalized:
                return library_name
        if "scikit" in normalized:
            return "sklearn"
        return None

    def _fallback_rewrite(self, raw_query: str, optimized_query: str) -> str:
        normalized = self._normalize_text(raw_query)
        if (
            "dataframe" in normalized
            and ("cot chung" in normalized or "common column" in normalized)
            and ("ket hop" in normalized or "join" in normalized or "merge" in normalized)
            and "pandas" in normalized
        ):
            return "merge two pandas DataFrames on a common column using pandas.merge"
        if (
            "pandas" in normalized
            and "csv" in normalized
            and any(term in normalized for term in ["read", "load", "doc"])
        ):
            return "read a CSV file using pandas.read_csv"
        if (
            ("sklearn" in normalized or "scikit" in normalized)
            and "train" in normalized
            and "test" in normalized
            and any(term in normalized for term in ["split", "chia", "tach"])
        ):
            return "split data into train and test sets using sklearn.model_selection.train_test_split"
        return optimized_query

    @staticmethod
    def _normalize_text(text: str) -> str:
        text = unicodedata.normalize("NFKD", text or "")
        text = "".join(ch for ch in text if not unicodedata.combining(ch))
        return text.lower()
