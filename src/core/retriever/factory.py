from core.retriever.base import BaseRetriever
from core.retriever.pg_hybrid_retriever import PGHybridRetrieverModel
from typing import Dict, Any


class RetrieverFactory:
    """Creates Retriever instances dynamically based on configuration (Factory Pattern)."""

    @staticmethod
    def create_retriever(config: Dict[str, Any]) -> BaseRetriever:
        ret_type = config.get("type", "pg_hybrid").lower()

        if ret_type == "pg_hybrid":
            return PGHybridRetrieverModel(config)
        # elif ret_type == "elasticsearch":
        #     return ElasticsearchRetrieverModel(config)
        else:
            raise ValueError(f"Unsupported Retriever type: {ret_type}")
