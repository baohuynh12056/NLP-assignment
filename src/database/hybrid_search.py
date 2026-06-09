import json
from typing import Any, Dict, List, Optional

import psycopg2
from psycopg2.extras import RealDictCursor


class PostgreSQLHybridSearch:
    """
    Runs keyword and semantic search as separate routes, then fuses normalized scores.
    """

    def __init__(self, db_config: Dict[str, Any]):
        self.db_config = db_config
        self.conn = psycopg2.connect(**self.db_config)

    def search(
        self,
        query: str,
        query_vector: List[float],
        top_k: int = 10,
        filters: Optional[Dict[str, Any]] = None,
        route_k: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        route_k = route_k or max(top_k * 4, 20)
        semantic_rows = self.semantic_search(query_vector, route_k, filters)
        keyword_rows = self.keyword_search(query, route_k, filters)
        return self.fuse_results(semantic_rows, keyword_rows, top_k=top_k)

    def semantic_search(
        self,
        query_vector: List[float],
        top_k: int,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        filter_clause, params = self._filter_clause(filters)
        vector_literal = "[" + ",".join(str(float(x)) for x in query_vector) + "]"
        sql = f"""
            SELECT
                id, full_name, library_name, docstring, parameters,
                1 - (embedding <=> %s::vector) AS semantic_score
            FROM functions
            WHERE embedding IS NOT NULL {filter_clause}
            ORDER BY embedding <=> %s::vector
            LIMIT %s;
        """
        return self._fetch(sql, [vector_literal, *params, vector_literal, top_k])

    def keyword_search(
        self,
        query: str,
        top_k: int,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        filter_clause, params = self._filter_clause(filters)
        sql = f"""
            SELECT
                id, full_name, library_name, docstring, parameters,
                ts_rank_cd(search_tsv, websearch_to_tsquery('english', %s)) AS keyword_score
            FROM functions
            WHERE search_tsv @@ websearch_to_tsquery('english', %s) {filter_clause}
            ORDER BY keyword_score DESC
            LIMIT %s;
        """
        return self._fetch(sql, [query, query, *params, top_k])

    def fuse_results(
        self,
        semantic_rows: List[Dict[str, Any]],
        keyword_rows: List[Dict[str, Any]],
        top_k: int,
        semantic_weight: float = 0.65,
        keyword_weight: float = 0.35,
    ) -> List[Dict[str, Any]]:
        merged: Dict[int, Dict[str, Any]] = {}
        semantic_scores = self._normalize([float(row.get("semantic_score") or 0.0) for row in semantic_rows])
        keyword_scores = self._normalize([float(row.get("keyword_score") or 0.0) for row in keyword_rows])

        for row, score in zip(semantic_rows, semantic_scores):
            item = merged.setdefault(row["id"], dict(row))
            item["semantic_norm"] = max(item.get("semantic_norm", 0.0), score)

        for row, score in zip(keyword_rows, keyword_scores):
            item = merged.setdefault(row["id"], dict(row))
            item["keyword_norm"] = max(item.get("keyword_norm", 0.0), score)

        fused = []
        for item in merged.values():
            item.setdefault("semantic_norm", 0.0)
            item.setdefault("keyword_norm", 0.0)
            item["hybrid_score"] = (
                semantic_weight * item["semantic_norm"] +
                keyword_weight * item["keyword_norm"]
            )
            fused.append(item)

        fused.sort(key=lambda row: row["hybrid_score"], reverse=True)
        return fused[:top_k]

    def _fetch(self, sql: str, params: List[Any]) -> List[Dict[str, Any]]:
        with self.conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(sql, params)
            return [dict(row) for row in cur.fetchall()]

    @staticmethod
    def _normalize(scores: List[float]) -> List[float]:
        if not scores:
            return []
        min_score = min(scores)
        max_score = max(scores)
        if max_score == min_score:
            return [1.0 for _ in scores]
        return [(score - min_score) / (max_score - min_score) for score in scores]

    @staticmethod
    def _filter_clause(filters: Optional[Dict[str, Any]]) -> tuple[str, List[Any]]:
        if filters and filters.get("library_name"):
            return "AND library_name = %s", [filters["library_name"]]
        return "", []

    @staticmethod
    def decode_parameters(raw: Any) -> Dict[str, Any]:
        if isinstance(raw, dict):
            return raw
        return json.loads(raw or "{}")

    def close(self) -> None:
        if self.conn:
            self.conn.close()

    def __del__(self):
        self.close()
