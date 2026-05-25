import json
import psycopg2
from psycopg2.extras import RealDictCursor
from typing import List, Optional, Dict, Any
from sentence_transformers import SentenceTransformer

from core.base_retriever import BaseRetriever
from core.schemas import Chunk

class PGHybridRetriever(BaseRetriever):
    """
    Implementation of BaseRetriever using PostgreSQL.
    Combines pgvector (Semantic Search) and Full-Text Search (Keyword Search).
    Uses BAAI/bge-base-en-v1.5 for lightning-fast query embedding.
    """

    def __init__(self, db_config: Dict[str, str], embed_model: str = "BAAI/bge-base-en-v1.5"):
        """
        Initializes the database connection and the local embedding model.
        
        Args:
            db_config (Dict): Database connection parameters (dbname, user, password, host, port).
            embed_model (str): The lightweight embedding model to use for the semantic route.
        """
        print(f"[PGHybridRetriever] Loading embedding model: {embed_model}...")
        self.model = SentenceTransformer(embed_model)
        
        print("[PGHybridRetriever] Connecting to PostgreSQL...")
        self.db_config = db_config
        # We establish a connection pool or a persistent connection. 
        # For production, consider using SQLAlchemy or asyncpg.
        self.conn = psycopg2.connect(**self.db_config)
        print("[PGHybridRetriever] Ready.")

    def retrieve(self, query: str, top_k: int = 10, filters: Optional[Dict[str, Any]] = None) -> List[Chunk]:
        """
        Executes a Hybrid Search (Vector + FTS) in PostgreSQL.
        """
        # 1. Encode the optimized query into a vector
        query_vector = self.model.encode(query, normalize_embeddings=True).tolist()
        
        # 2. Build the Metadata Filter string safely
        filter_clause = ""
        sql_params = [query_vector, query]  # params for: 1. vector, 2. FTS keyword
        
        if filters and "library_name" in filters:
            filter_clause = "AND library_name = %s"
            sql_params.append(filters["library_name"])
            
        sql_params.append(top_k) # param for LIMIT
        
        # 3. The Hybrid SQL Query
        # We use a CTE (Common Table Expression) to calculate both Vector Distance and FTS Rank.
        # Note: In a real advanced setup, you would normalize and combine these scores (RRF).
        # Here we order primarily by semantic similarity, with FTS as a secondary booster.
        sql_query = f"""
            WITH search_results AS (
                SELECT 
                    id,
                    full_name,
                    library_name,
                    docstring,
                    parameters,
                    1 - (embedding <=> %s::vector) AS semantic_score,
                    ts_rank_cd(to_tsvector('english', search_text), plainto_tsquery('english', %s)) AS keyword_score
                FROM functions
                WHERE 1=1 {filter_clause}
            )
            SELECT 
                *,
                (semantic_score * 0.7) + (keyword_score * 0.3) AS hybrid_score
            FROM search_results
            ORDER BY hybrid_score DESC
            LIMIT %s;
        """

        # 4. Execute Query
        chunks = []
        with self.conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(sql_query, sql_params)
            rows = cur.fetchall()
            
            # 5. Map SQL rows to Standard Chunk Schema
            for row in rows:
                metadata = {
                    "library_name": row["library_name"],
                    "func_name": row["full_name"],
                    "parameters": row["parameters"] if isinstance(row["parameters"], dict) else json.loads(row["parameters"] or "{}")
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
        if hasattr(self, 'conn') and self.conn:
            self.conn.close()