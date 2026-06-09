from psycopg2.extras import RealDictCursor
from typing import List, Dict, Any, Optional
from database.db_manager import DatabaseManager
from utils.logger import get_logger
import json

logger = get_logger(__name__)

class PostgresHybridSearch:
    """Handles raw SQL execution for Hybrid Search (Vector + Full Text Search)."""
    
    def __init__(self):
        self.db = DatabaseManager()

    # def execute_search(self, query_vector: List[float], query_text: str, top_k: int, semantic_weight: float = 0.7, filters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
    #     """
    #     Executes the hybrid search fusion directly inside PostgreSQL.
    #     """
    #     keyword_weight = 1.0 - semantic_weight
    #     filter_clause = ""
    #     sql_params = [query_vector, query_text]
        
    #     # Apply dynamic metadata filters safely
    #     if filters and "library_name" in filters:
    #         filter_clause = "AND library_name = %s"
    #         sql_params.append(filters["library_name"])
            
    #     sql_params.extend([semantic_weight, keyword_weight, top_k])
        
    #     sql_query = f"""
    #         WITH search_results AS (
    #             SELECT 
    #                 id, func_name, library_name, docstring, parameters,
    #                 1 - (embedding <=> %s::vector) AS semantic_score,
    #                 ts_rank_cd(to_tsvector('english', search_text), plainto_tsquery('english', %s)) AS keyword_score
    #             FROM functions
    #             WHERE 1=1 {filter_clause}
    #         )
    #         SELECT 
    #             *, (semantic_score * %s) + (keyword_score * %s) AS hybrid_score
    #         FROM search_results
    #         ORDER BY hybrid_score DESC
    #         LIMIT %s;
    #     """

    #     results = []
    #     with self.db.get_connection() as conn:
    #         with conn.cursor(cursor_factory=RealDictCursor) as cur:
    #             try:
    #                 cur.execute(sql_query, sql_params)
    #                 rows = cur.fetchall()
    #                 for row in rows:
    #                     # Convert DB row format to expected Dictionary format
    #                     row_dict = dict(row)
    #                     if isinstance(row_dict.get("parameters"), str):
    #                         row_dict["parameters"] = json.loads(row_dict["parameters"])
    #                     results.append(row_dict)
    #             except Exception as e:
    #                 logger.error(f"SQL execution failed during hybrid search: {e}")
    #     return results
    def execute_search(
    self,
    query_vector: List[float],
    query_text: str,
    top_k: int,
    semantic_weight: float = 0.7,
    filters: Optional[Dict[str, Any]] = None
) -> List[Dict[str, Any]]:

        keyword_weight = 1.0 - semantic_weight

        filter_value = None
        if filters and "library_name" in filters:
            filter_value = filters["library_name"]

        sql_query = """
            WITH search_results AS (
                SELECT 
                    id,
                    func_name,
                    library_name,
                    docstring,
                    parameters,

                    -- VECTOR SCORE (luôn chạy)
                    1 - (embedding <=> %s::vector) AS semantic_score,

                    -- KEYWORD SCORE (luôn chạy)
                    ts_rank_cd(
                        to_tsvector('english', search_text),
                        plainto_tsquery('english', %s)
                    ) AS keyword_score,

                    -- FILTER BOOST (KHÔNG CHẶN DATA)
                    CASE
                        WHEN %s IS NULL THEN 0
                        WHEN library_name = %s THEN 1
                        ELSE 0
                    END AS filter_score

                FROM functions
            )
            SELECT 
                *,
                (
                    semantic_score * %s +
                    keyword_score * %s +
                    filter_score * 0.2
                ) AS hybrid_score
            FROM search_results
            ORDER BY hybrid_score DESC
            LIMIT %s;
        """

        sql_params = [
            query_vector,   # %s vector
            query_text,     # %s text
            filter_value,   # %s filter check
            filter_value,   # %s filter compare
            semantic_weight,
            keyword_weight,
            top_k
        ]

        results = []

        with self.db.get_connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                try:
                    cur.execute(sql_query, sql_params)
                    rows = cur.fetchall()

                    for row in rows:
                        row_dict = dict(row)

                        if isinstance(row_dict.get("parameters"), str):
                            row_dict["parameters"] = json.loads(row_dict["parameters"])

                        results.append(row_dict)

                except Exception as e:
                    logger.error(f"SQL execution failed during hybrid search: {e}")

        return results  