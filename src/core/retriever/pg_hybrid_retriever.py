import json
import psycopg2
from psycopg2.extras import RealDictCursor
from typing import List, Dict, Any, Optional
from sentence_transformers import SentenceTransformer

from core.retriever.base import BaseRetriever
from core.schemas import Chunk
from utils.logger import get_logger

logger = get_logger(__name__)


class PGHybridRetrieverModel(BaseRetriever):
    """
    Implementation of BaseRetriever using PostgreSQL.
    Performs Hybrid Search (pgvector + Full-Text Search) and Result Fusion inside the DB.
    """

    def __init__(self, config: Dict[str, Any]):
        # Load embedding model
        embed_model_path = config.get("embed_model_path", "BAAI/bge-small-en-v1.5")
        logger.info(f"Loading local Embedding model from: {embed_model_path}")
        self.embed_model = SentenceTransformer(embed_model_path)

        # Load fusion weights
        weights = config.get("weights", {})
        self.semantic_weight = float(weights.get("semantic", 0.7))
        self.keyword_weight = float(weights.get("keyword", 0.3))

        # Establish Database connection
        db_config = config.get("database", {})
        logger.info(
            f"Connecting to PostgreSQL Database at {db_config.get('host')}:{db_config.get('port')}..."
        )
        self.conn = psycopg2.connect(**db_config)
        logger.info("Database connection established successfully.")

    def retrieve(
        self, query: str, top_k: int = 10, filters: Optional[Dict[str, Any]] = None
    ) -> List[Chunk]:
        logger.info(f"Executing hybrid search for query: '{query}'")

        # 1. Embed the query into a vector
        query_vector = self.embed_model.encode(query, normalize_embeddings=True).tolist()

        # 2. Build dynamic metadata filters securely
        filter_clause = ""
        sql_params = [query_vector, query]  # Base parameters for vector and FTS

        if filters and "library_name" in filters:
            filter_clause = "AND library_name = %s"
            sql_params.append(filters["library_name"])
            logger.debug(
                f"Applied database filter: library_name = {filters['library_name']}"
            )

        # Append parameters for fusion calculation and LIMIT
        sql_params.extend([self.semantic_weight, self.keyword_weight, top_k])

        # 3. Hybrid SQL Query using Common Table Expressions (CTE)
        sql_query = f"""
            WITH search_results AS (
                SELECT
                    id,
                    func_name,
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
                (semantic_score * %s) + (keyword_score * %s) AS hybrid_score
            FROM search_results
            ORDER BY hybrid_score DESC
            LIMIT %s;
        """

        # 4. Execute query and map to Chunk schema
        chunks = []
        try:
            with self.conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute("SET LOCAL ivfflat.probes = 100;")
                cur.execute(sql_query, sql_params)
                rows = cur.fetchall()

                for row in rows:
                    metadata = {
                        "library_name": row["library_name"],
                        "func_name": row["func_name"],
                        "parameters": row["parameters"]
                        if isinstance(row["parameters"], dict)
                        else json.loads(row["parameters"] or "{}"),
                    }

                    chunks.append(
                        Chunk(
                            id=str(row["id"]),
                            content=row["docstring"],
                            metadata=metadata,
                            score=float(row["hybrid_score"]),
                        )
                    )
            logger.info(f"Retrieved {len(chunks)} chunks successfully from Database.")
        except Exception as e:
            logger.error(f"Database retrieval failed: {str(e)}")

        return chunks

    def __del__(self):
        """Safely close the database connection upon object destruction."""
        if hasattr(self, "conn") and self.conn:
            self.conn.close()
            logger.debug("Database connection closed.")
