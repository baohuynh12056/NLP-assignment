from utils.config_loader import GLOBAL_CONFIG
from utils.logger import get_logger

# Import Core Managers & Factories
from core.llm.manager import LLMManager
from core.retriever.factory import RetrieverFactory
from core.reranker.factory import RerankerFactory

# Import Business Agents
from models.query_parser import QueryParser
from models.retriever import DocumentRetriever
from models.reranker import DocumentReranker
from models.answer_generator import AnswerGenerator

# Import Pipeline Orchestrator
from pipeline.rag_orchestrator import RAGOrchestrator

logger = get_logger("main")


def initialize_orchestrator() -> RAGOrchestrator:
    """Initializes all models, agents, and the orchestrator."""
    logger.info("Initializing System Components...")

    # 1. Initialize Core Models (Hardware/AI layer)
    llm_manager = LLMManager()
    core_retriever = RetrieverFactory.create_retriever(
        GLOBAL_CONFIG.get("retriever", {})
    )
    core_reranker = RerankerFactory.create_reranker(
        GLOBAL_CONFIG.get("models", {}).get("reranker", {})
    )

    # 2. Initialize Business Agents (Logic layer)
    query_parser = QueryParser(llm=llm_manager.parser_llm)
    answer_generator = AnswerGenerator(llm=llm_manager.generator_llm)
    retriever_agent = DocumentRetriever(retriever_model=core_retriever)
    reranker_agent = DocumentReranker(reranker_model=core_reranker)

    # 3. Inject Agents into Orchestrator
    orchestrator = RAGOrchestrator(
        query_parser=query_parser,
        retriever=retriever_agent,
        reranker=reranker_agent,
        answer_generator=answer_generator,
    )

    logger.info("System Initialization Complete.")
    return orchestrator


def main():
    # Setup and get the fully configured pipeline
    pipeline = initialize_orchestrator()

    # Test the system
    test_query = "Làm sao để gom nhóm dữ liệu theo một cột và tính giá trị trung bình trong thư viện pandas?"

    # Run the pipeline
    result = pipeline.run(test_query)

    # Output the result
    print("\n\n🤖 AI ASSISTANT ANSWER:\n")
    print(result)
    print("\n" + "=" * 50)


if __name__ == "__main__":
    main()
