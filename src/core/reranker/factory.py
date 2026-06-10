# src/core/reranker/factory.py
from typing import Dict, Any
from core.reranker.base import BaseReranker
from core.reranker.cross_encoder_reranker import CrossEncoderRerankerModel

class RerankerFactory:
    """Creates Reranker instances dynamically based on configuration (Factory Pattern)."""

    @staticmethod
    def create_reranker(config: Dict[str, Any]) -> BaseReranker:
        reranker_type = config.get("type", "cross_encoder").lower()

        if reranker_type == "cross_encoder":
            return CrossEncoderRerankerModel(config)
        else:
            raise ValueError(f"Unsupported Reranker type: {reranker_type}")