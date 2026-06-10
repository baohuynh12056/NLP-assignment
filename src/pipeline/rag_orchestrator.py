from models.query_parser import QueryParser
from models.retriever import DocumentRetriever
from models.reranker import DocumentReranker
from models.answer_generator import AnswerGenerator
from core.schemas import Chunk, ParsedQuery, RAGResponse, SourceChunk
from utils.config_loader import GLOBAL_CONFIG
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
        retriever_cfg = GLOBAL_CONFIG.get("retriever", {})
        self.fast_mode_rerank = bool(retriever_cfg.get("fast_mode_rerank", True))
        self.full_mode_rerank = bool(retriever_cfg.get("full_mode_rerank", True))
        self.context_k = int(retriever_cfg.get("context_k", self.reranker.default_top_k))

    def run(self, user_query: str) -> str:
        return self.run_with_details(user_query, generate_answer=True).answer

    def run_with_details(self, user_query: str, generate_answer: bool = True) -> RAGResponse:
        """
        Executes the end-to-end RAG workflow.

        Args:
            user_query (str): The raw input question from the user.

        Returns:
            RAGResponse: The generated answer and source chunks.
        """
        logger.info("========== STARTING RAG PIPELINE ==========")
        logger.info(f"User Query: '{user_query}'")

        parsed_query, refined_chunks = self.retrieve_context(
            user_query=user_query,
            generate_answer=generate_answer,
        )

        if not refined_chunks:
            logger.warning("Pipeline terminating early: No relevant chunks found.")
            return RAGResponse(
                query=user_query,
                optimized_query=parsed_query.optimized_query,
                filters=parsed_query.filters,
                answer="I'm sorry, I couldn't find any relevant functions in the database to answer your query.",
                sources=[],
            )

        # Step 4: Generate the final answer using the top-K chunks
        if generate_answer:
            logger.info("[Step 4/4] Generating Final Answer...")
            final_answer = self.answer_generator.generate(
                query=user_query, context_chunks=refined_chunks
            )
        else:
            logger.info("[Step 4/4] Building Fast Answer...")
            final_answer = self._build_fast_answer(refined_chunks)

        logger.info("========== RAG PIPELINE COMPLETED ==========")
        return self.build_response(user_query, parsed_query, refined_chunks, final_answer)

    def retrieve_context(
        self,
        user_query: str,
        generate_answer: bool = True,
    ) -> tuple[ParsedQuery, list[Chunk]]:
        """Parse, retrieve, and optionally rerank chunks for a user query."""
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
            return parsed_query, []

        refined_query = parsed_query.optimized_query or user_query
        should_rerank = (
            (generate_answer and self.full_mode_rerank)
            or (not generate_answer and self.fast_mode_rerank)
        )
        if should_rerank:
            # Step 3: Rerank the candidates to get the most relevant top-K chunks
            logger.info("[Step 3/4] Reranking Candidates...")
            refined_chunks = self.reranker.process(
                query=refined_query, candidate_chunks=raw_chunks
            )
        else:
            mode_name = "Full" if generate_answer else "Fast"
            logger.info(f"[Step 3/4] Skipping reranker for {mode_name} mode...")
            refined_chunks = raw_chunks[: self.context_k]

        refined_chunks = self._ensure_exact_function(refined_query, raw_chunks, refined_chunks)

        if not refined_chunks:
            logger.warning(
                "Pipeline terminating early: All chunks were filtered out during reranking."
            )
            return parsed_query, []

        return parsed_query, refined_chunks

    def build_response(
        self,
        user_query: str,
        parsed_query: ParsedQuery,
        chunks: list[Chunk],
        answer: str,
    ) -> RAGResponse:
        return RAGResponse(
            query=user_query,
            optimized_query=parsed_query.optimized_query,
            filters=parsed_query.filters,
            answer=answer,
            sources=[self._chunk_to_source(chunk) for chunk in chunks],
        )

    def _ensure_exact_function(
        self,
        query: str,
        candidates: list[Chunk],
        chunks: list[Chunk],
    ) -> list[Chunk]:
        exact_matches = self._prioritize_exact_function(query, candidates)
        if not exact_matches:
            return chunks

        merged = exact_matches + chunks
        seen = set()
        deduped = []
        for chunk in merged:
            key = chunk.id or chunk.metadata.get("full_name") or chunk.metadata.get("func_name")
            if key in seen:
                continue
            seen.add(key)
            deduped.append(chunk)
        return deduped[: self.context_k]

    @staticmethod
    def _prioritize_exact_function(query: str, chunks: list[Chunk]) -> list[Chunk]:
        query_lower = (query or "").lower()
        exact = []
        for chunk in chunks:
            names = [
                str(chunk.metadata.get("func_name", "")).lower(),
                str(chunk.metadata.get("full_name", "")).lower(),
            ]
            if any(name and (name in query_lower or name.split(".")[-1] in query_lower) for name in names):
                exact.append(chunk)
        return sorted(exact, key=lambda chunk: -(chunk.score or 0.0))

    @staticmethod
    def _chunk_to_source(chunk: Chunk) -> SourceChunk:
        snippet = " ".join((chunk.content or "").strip().split())
        if len(snippet) > 500:
            snippet = snippet[:497].rstrip() + "..."

        return SourceChunk(
            id=chunk.id,
            function_name=chunk.metadata.get("func_name", "Unknown"),
            library_name=chunk.metadata.get("library_name", "Unknown"),
            score=chunk.score,
            snippet=snippet,
            parameters=chunk.metadata.get("parameters", {}),
        )

    @staticmethod
    def _build_fast_answer(chunks: list[Chunk]) -> str:
        if not chunks:
            return "I could not find a matching function in the documentation."

        top = chunks[0]
        function_name = top.metadata.get("func_name", "the matched function")
        function_key = str(function_name).split(".")[-1]
        snippet = " ".join((top.content or "").strip().split())
        if len(snippet) > 900:
            snippet = snippet[:897].rstrip() + "..."

        if function_name == "pandas.merge" or function_key == "merge":
            return (
                "Use `pandas.merge(left, right, on='common_column', how='inner')` "
                "to combine two DataFrames on a shared column.\n\n"
                "Example:\n"
                "```python\n"
                "result = pandas.merge(df1, df2, on='common_column', how='inner')\n"
                "```\n\n"
                "Change `how` to `left`, `right`, or `outer` depending on the join behavior you need."
            )

        if function_name == "pandas.read_csv" or function_key == "read_csv":
            return (
                "Use `pandas.read_csv(path)` to load a CSV file into a DataFrame.\n\n"
                "Example:\n"
                "```python\n"
                "df = pandas.read_csv('data.csv')\n"
                "```\n\n"
                "Common options include `sep=','`, `encoding='utf-8'`, and `usecols=[...]`."
            )

        if (
            function_name == "sklearn.model_selection.train_test_split"
            or function_key == "train_test_split"
        ):
            return (
                "Use `train_test_split` from `sklearn.model_selection` to split arrays or DataFrames "
                "into training and testing subsets.\n\n"
                "Example:\n"
                "```python\n"
                "from sklearn.model_selection import train_test_split\n\n"
                "X_train, X_test, y_train, y_test = train_test_split(\n"
                "    X, y, test_size=0.2, random_state=42\n"
                ")\n"
                "```\n\n"
                "`test_size` controls the test proportion, and `random_state` makes the split reproducible."
            )

        return (
            f"Most relevant function: `{function_name}`.\n\n"
            f"Documentation excerpt:\n{snippet}"
        )
