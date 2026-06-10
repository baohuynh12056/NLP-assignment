import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from core.retriever.base import BaseRetriever
from core.schemas import Chunk
from utils.logger import get_logger


logger = get_logger(__name__)


class FAISSRetrieverModel(BaseRetriever):
    """Local FAISS retriever for fast in-memory vector search."""

    def __init__(self, config: Dict[str, Any]):
        embed_model_path = config.get("embed_model_path", "BAAI/bge-small-en-v1.5")
        self.index_path = Path(config.get("index_path", "models/faiss/functions.index"))
        self.metadata_path = Path(
            config.get("metadata_path", "models/faiss/functions_metadata.jsonl")
        )
        self.query_prefix = config.get("query_prefix", "")
        self.score_threshold = float(config.get("score_threshold", -1.0))
        self.fetch_multiplier = int(config.get("fetch_multiplier", 4))

        logger.info(f"Loading embedding model from: {embed_model_path}")
        self.embed_model = SentenceTransformer(embed_model_path)

        if not self.index_path.exists() or not self.metadata_path.exists():
            raise FileNotFoundError(
                "FAISS index files are missing. Run: "
                "PYTHONPATH=src python src/tools/build_faiss_index.py"
            )

        logger.info(f"Loading FAISS index from: {self.index_path}")
        self.index = faiss.read_index(str(self.index_path))
        self.records = self._load_records(self.metadata_path)
        logger.info(
            f"Loaded FAISS retriever with {self.index.ntotal} vectors and {len(self.records)} records."
        )

        if self.index.ntotal != len(self.records):
            raise ValueError(
                f"FAISS index has {self.index.ntotal} vectors but metadata has {len(self.records)} records."
            )

    def retrieve(
        self, query: str, top_k: int = 10, filters: Optional[Dict[str, Any]] = None
    ) -> List[Chunk]:
        query_text = f"{self.query_prefix}{query}" if self.query_prefix else query
        query_vector = self.embed_model.encode(
            query_text,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        ).astype("float32")
        query_vector = np.expand_dims(query_vector, axis=0)

        fetch_k = min(max(top_k * self.fetch_multiplier, top_k), self.index.ntotal)
        scores, indices = self.index.search(query_vector, fetch_k)

        chunks: List[Chunk] = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:
                continue
            score = float(score)
            if score < self.score_threshold:
                continue

            record = self.records[int(idx)]
            if not self._matches_filters(record, filters):
                continue

            chunks.append(
                Chunk(
                    id=str(record.get("id", idx)),
                    content=record.get("content", ""),
                    metadata=record.get("metadata", {}),
                    score=score,
                )
            )
            if len(chunks) >= top_k:
                break

        logger.info(f"FAISS retrieved {len(chunks)} chunks for query: '{query}'")
        return chunks

    @staticmethod
    def _load_records(path: Path) -> List[Dict[str, Any]]:
        records = []
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    records.append(json.loads(line))
        return records

    @staticmethod
    def _matches_filters(record: Dict[str, Any], filters: Optional[Dict[str, Any]]) -> bool:
        if not filters:
            return True
        metadata = record.get("metadata", {})
        for key, value in filters.items():
            if value is None:
                continue
            if str(metadata.get(key, "")).lower() != str(value).lower():
                return False
        return True
