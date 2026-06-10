# src/core/retriever/factory.py
from typing import Dict, Any
from core.retriever.base import BaseRetriever
from core.retriever.pg_hybrid_retriever import PGHybridRetrieverModel

class RetrieverFactory:
    """Creates Retriever instances dynamically based on configuration."""

    @staticmethod
    def create_retriever(config: Dict[str, Any]) -> BaseRetriever:
        ret_type = config.get("type", "pg_hybrid").lower()

        if ret_type == "pg_hybrid":
            return PGHybridRetrieverModel(config)
        else:
            raise ValueError(f"Unsupported Retriever type: {ret_type}")