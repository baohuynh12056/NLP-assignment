from typing import List, Optional, Dict, Any
from sentence_transformers import SentenceTransformer

from core.base_retriever import BaseRetriever
from core.schemas import Chunk
from database.hybrid_search import PostgreSQLHybridSearch

class PGHybridRetriever(BaseRetriever):
    """
    Implementation of BaseRetriever using PostgreSQL.
    Combines pgvector (Semantic Search) and Full-Text Search (Keyword Search).
    Uses BAAI/bge-small-en-v1.5 by default to match the 384-dimension pgvector schema.
    """

    def __init__(self, db_config: Dict[str, str], embed_model: str = "BAAI/bge-small-en-v1.5"):
        """
        Initializes the database connection and the local embedding model.
        
        Args:
            db_config (Dict): Database connection parameters (dbname, user, password, host, port).
            embed_model (str): The lightweight embedding model to use for the semantic route.
        """
        print(f"[PGHybridRetriever] Loading embedding model: {embed_model}...")
        self.model = SentenceTransformer(embed_model)
        
        self.db_config = db_config
        self.search_backend = PostgreSQLHybridSearch(db_config)
        print("[PGHybridRetriever] Ready.")

    def retrieve(self, query: str, top_k: int = 10, filters: Optional[Dict[str, Any]] = None) -> List[Chunk]:
        """
        Executes a Hybrid Search (Vector + FTS) in PostgreSQL.
        """
        query_vector = self.model.encode(query, normalize_embeddings=True).tolist()

        rows = self.search_backend.search(
            query=query,
            query_vector=query_vector,
            top_k=top_k,
            filters=filters,
        )

        chunks = []
        for row in rows:
            metadata = {
                "library_name": row["library_name"],
                "func_name": row["full_name"],
                "parameters": self.search_backend.decode_parameters(row.get("parameters")),
                "semantic_score": float(row.get("semantic_norm") or 0.0),
                "keyword_score": float(row.get("keyword_norm") or 0.0),
            }

            chunks.append(Chunk(
                id=str(row["id"]),
                content=row["docstring"],
                metadata=metadata,
                score=float(row["hybrid_score"])
            ))
                
        print(f"[PGHybridRetriever] Retrieved {len(chunks)} hybrid chunks from DB.")
        return chunks

    def __del__(self):
        """Ensure the database connection is closed when the object is destroyed."""
        if hasattr(self, "search_backend") and self.search_backend:
            self.search_backend.close()
