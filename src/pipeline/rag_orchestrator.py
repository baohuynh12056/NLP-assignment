import logging
from typing import List

from core.base_llm import BaseLLM
from core.base_retriever import BaseRetriever
from core.base_reranker import BaseReranker
from core.schemas import Chunk, ParsedQuery, RAGResponse, SourceChunk
from pipeline.query_processor import QueryProcessor


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RAGPipeline:
    """
    Orchestrates the full RAG lifecycle:
    Query parsing -> hybrid retrieval -> cross-encoder reranking -> LLM generation.
    """

    def __init__(
        self,
        parser_llm: BaseLLM,
        retriever: BaseRetriever,
        reranker: BaseReranker,
        generator_llm: BaseLLM,
        retrieval_k: int = 12,
        context_k: int = 4,
    ):
        self.processor = QueryProcessor(llm=parser_llm)
        self.retriever = retriever
        self.reranker = reranker
        self.generator_llm = generator_llm
        self.retrieval_k = retrieval_k
        self.context_k = context_k

    def run(self, raw_query: str) -> str:
        return self.run_with_details(raw_query, generate_answer=True).answer

    def run_with_details(self, raw_query: str, generate_answer: bool = True) -> RAGResponse:
        print("\n" + "=" * 60)
        print("INITIATING RAG PIPELINE")
        print(f"USER QUERY: '{raw_query}'")
        print("=" * 60)

        print("\n[Step 1/4] Parsing Query & Extracting Filters...")
        parsed_data: ParsedQuery = self.processor.process(raw_query)

        print("\n[Step 2/4] Retrieving from Database...")
        initial_chunks: List[Chunk] = self.retriever.retrieve(
            query=parsed_data.optimized_query,
            top_k=self.retrieval_k,
            filters=parsed_data.filters,
        )

        if not initial_chunks:
            logger.warning("No relevant documents found in the database.")
            return RAGResponse(
                query=raw_query,
                optimized_query=parsed_data.optimized_query,
                filters=parsed_data.filters,
                answer="I'm sorry, but I couldn't find any relevant functions in the documentation to answer your question.",
                sources=[],
            )

        print(f"\n[Step 3/4] Reranking {len(initial_chunks)} candidates...")
        best_chunks: List[Chunk] = self.reranker.rerank(
            query=parsed_data.optimized_query,
            chunks=initial_chunks,
            top_k=self.context_k,
        )
        best_chunks = self._ensure_exact_function(
            parsed_data.optimized_query,
            initial_chunks,
            best_chunks,
        )

        if generate_answer:
            print("\n[Step 4/4] Generating Final Answer...")
            final_answer = self.generator_llm.generate_answer(
                query=raw_query,
                context_chunks=best_chunks,
            )
        else:
            print("\n[Step 4/4] Building Fast Answer...")
            final_answer = self._build_fast_answer(raw_query, best_chunks)

        print("\n" + "=" * 60)
        print("PIPELINE COMPLETE")
        print("=" * 60 + "\n")

        return RAGResponse(
            query=raw_query,
            optimized_query=parsed_data.optimized_query,
            filters=parsed_data.filters,
            answer=final_answer,
            sources=[self._chunk_to_source(chunk) for chunk in best_chunks],
        )

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
            semantic_score=chunk.metadata.get("semantic_score"),
            keyword_score=chunk.metadata.get("keyword_score"),
            snippet=snippet,
            parameters=chunk.metadata.get("parameters", {}),
        )

    def _ensure_exact_function(
        self,
        query: str,
        candidates: List[Chunk],
        chunks: List[Chunk],
    ) -> List[Chunk]:
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
    def _prioritize_exact_function(query: str, chunks: List[Chunk]) -> List[Chunk]:
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
    def _build_fast_answer(query: str, chunks: List[Chunk]) -> str:
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

        if function_name == "sklearn.model_selection.train_test_split" or function_key == "train_test_split":
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
