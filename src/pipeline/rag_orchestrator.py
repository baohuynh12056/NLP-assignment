from models.query_parser import QueryParser
from models.retriever import DocumentRetriever
from models.reranker import DocumentReranker
from models.answer_generator import AnswerGenerator
from utils.logger import get_logger

# Initialize module logger
logger = get_logger(__name__)


class RAGOrchestrator:
    """
    The central orchestrator that wires together the entire RAG pipeline.
    Manages data flow from query parsing to retrieval, reranking, and final answer generation.
    """

    def __init__(
        self,
        query_parser: QueryParser,
        retriever: DocumentRetriever,
        reranker: DocumentReranker,
        answer_generator: AnswerGenerator,
    ):
        """Injects the business agents required for the pipeline."""
        self.query_parser = query_parser
        self.retriever = retriever
        self.reranker = reranker
        self.answer_generator = answer_generator

    def run(self, user_query: str) -> str:
        """
        Executes the end-to-end RAG workflow.

        Args:
            user_query (str): The raw input question from the user.

        Returns:
            str: The final generated answer.
        """
        logger.info("========== STARTING RAG PIPELINE ==========")
        logger.info(f"User Query: '{user_query}'")

        # Step 1: Parse and optimize the query
        logger.info("[Step 1/4] Parsing Query...")
        parsed_query = self.query_parser.parse(user_query)

        # Step 2: Retrieve raw candidate chunks from the database
        logger.info("[Step 2/4] Retrieving Candidates...")
        raw_chunks = self.retriever.process(parsed_query)

        if not raw_chunks:
            logger.warning(
                "Pipeline terminating early: No chunks retrieved from the database."
            )
            return "I'm sorry, I couldn't find any relevant functions in the database to answer your query."

        # Step 3: Rerank the candidates to get the most relevant top-K chunks
        logger.info("[Step 3/4] Reranking Candidates...")
        # Use the optimized query for semantic reranking
        refined_query = parsed_query.get("optimized_query", user_query)
        refined_chunks = self.reranker.process(
            query=refined_query, candidate_chunks=raw_chunks
        )

        if not refined_chunks:
            logger.warning(
                "Pipeline terminating early: All chunks were filtered out during reranking."
            )
            return "I'm sorry, no retrieved functions met the relevance threshold."

        # Step 4: Generate the final answer using the top-K chunks
        logger.info("[Step 4/4] Generating Final Answer...")
        final_answer = self.answer_generator.generate(
            query=user_query, context_chunks=refined_chunks
        )

        logger.info("========== RAG PIPELINE COMPLETED ==========")
        return final_answer
