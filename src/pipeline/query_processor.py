import logging
from typing import List, Optional

from core.base_llm import BaseLLM
from core.schemas import ParsedQuery

# Configure basic logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class QueryProcessor:
    """
    Handles the preprocessing of user queries before they hit the retrieval systems.
    Acts as a safety and validation layer on top of the LLM's extraction logic 
    to prevent SQL/Database errors caused by LLM hallucinations.
    """

    def __init__(self, llm: BaseLLM, supported_libraries: Optional[List[str]] = None):
        """
        Initializes the QueryProcessor.

        Args:
            llm (BaseLLM): The language model used to parse the query.
            supported_libraries (List[str], optional): A list of valid libraries in the database.
                                                       Used to validate LLM extractions.
        """
        self.llm = llm
        # Default fallback list (you can load this dynamically from PostgreSQL later)
        self.supported_libraries = supported_libraries or [
            "pandas", "numpy", "sklearn", "torch", "tensorflow", 
            "matplotlib", "scipy", "seaborn", "requests", "fastapi"
        ]

    def process(self, raw_query: str) -> ParsedQuery:
        """
        Executes the query parsing pipeline: 
        1. Calls the LLM to extract intent and filters.
        2. Validates and sanitizes the extracted metadata.
        
        Args:
            raw_query (str): The raw text inputted by the user.
            
        Returns:
            ParsedQuery: The validated and safe parsed query object.
        """
        logger.info(f"Processing raw query: '{raw_query}'")

        # Step 1: Let the Mini-LLM do the heavy lifting (Extraction & Rewriting)
        parsed_result: ParsedQuery = self.llm.parse_query(raw_query)
        
        # Step 2: Validate the extracted metadata (Safety Layer)
        validated_filters = {}
        
        if "library_name" in parsed_result.filters:
            extracted_lib = str(parsed_result.filters["library_name"]).strip().lower()
            
            # Check if the extracted library actually exists in our database
            if extracted_lib in self.supported_libraries:
                validated_filters["library_name"] = extracted_lib
                logger.info(f"Validated filter applied: library_name = {extracted_lib}")
            else:
                logger.warning(
                    f"LLM extracted an unsupported/hallucinated library: '{extracted_lib}'. "
                    f"Dropping this filter to prevent database lookup errors."
                )
                # Optionally: You could append the hallucinated library back to the query text here
                parsed_result.optimized_query += f" {extracted_lib}"

        # Update the object with the safe filters
        parsed_result.filters = validated_filters
        
        # Step 3: Final sanitization of the semantic query
        # Ensure it's not empty; if the LLM completely failed, fallback to the raw query
        if not parsed_result.optimized_query.strip():
            logger.warning("LLM returned an empty optimized query. Falling back to raw query.")
            parsed_result.optimized_query = raw_query

        logger.info(f"Final Optimized Query: '{parsed_result.optimized_query}'")
        return parsed_result
