from core.retriever.base import BaseRetriever
from core.retriever.faiss_retriever import FAISSRetrieverModel
from core.retriever.pg_hybrid_retriever import PGHybridRetrieverModel
from typing import Dict, Any


class RetrieverFactory:
    """Creates Retriever instances dynamically based on configuration (Factory Pattern)."""

    @staticmethod
    def create_retriever(config: Dict[str, Any]) -> BaseRetriever:
        ret_type = config.get("type", "pg_hybrid").lower()

        if ret_type == "pg_hybrid":
            return PGHybridRetrieverModel(config)
        if ret_type == "faiss_local":
            return FAISSRetrieverModel(config)
        # elif ret_type == "elasticsearch":
        #     return ElasticsearchRetrieverModel(config)
        else:
            raise ValueError(f"Unsupported Retriever type: {ret_type}")
