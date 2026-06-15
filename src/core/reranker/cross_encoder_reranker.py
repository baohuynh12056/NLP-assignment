import torch
from typing import List, Dict, Any
from sentence_transformers import CrossEncoder

from core.reranker.base import BaseReranker
from core.schemas import Chunk
from utils.logger import get_logger

logger = get_logger(__name__)


class CrossEncoderRerankerModel(BaseReranker):
    """Generic implementation for any Cross-Encoder model (BGE, Jina, MiniLM, etc.)."""

    def __init__(self, config: Dict[str, Any]):
        model_path = config.get("model_path")
        if not model_path:
            raise ValueError("model_path is required for CrossEncoderRerankerModel")

        max_length = config.get("max_length", 512)

        logger.info(f"Loading Cross-Encoder Reranker from: {model_path}")

        # Auto-detect hardware device
        device = config.get("device")
        if not device:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        # Optimize VRAM usage with float16 if running on GPU
        model_kwargs = {"torch_dtype": torch.float16} if device == "cuda" else {}

        self.model = CrossEncoder(
            model_path, max_length=max_length, device=device, model_kwargs=model_kwargs
        )
        logger.info(f"Cross-Encoder loaded successfully on {device.upper()}.")

    def rerank(self, query: str, chunks: List[Chunk], top_k: int = 5) -> List[Chunk]:
        if not chunks:
            return []

        # Format inputs for the CrossEncoder
        sentence_pairs = [[query, chunk.content] for chunk in chunks]

        # Predict relevance scores
        scores = self.model.predict(sentence_pairs, show_progress_bar=False)

        # Update chunks with new scores
        for chunk, score in zip(chunks, scores):
            chunk.score = float(score)

        # Sort chunks in descending order
        chunks.sort(key=lambda x: x.score, reverse=True)
        return chunks[:top_k]
