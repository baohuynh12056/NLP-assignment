import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import psycopg2
from psycopg2.extras import execute_values


DEFAULT_DB_CONFIG = {
    "dbname": "rag_database",
    "user": "admin",
    "password": "secretpassword",
    "host": "localhost",
    "port": 5432,
}


class DatabaseManager:
    """Small PostgreSQL helper for schema setup and function-doc ingestion."""

    def __init__(self, db_config: Optional[Dict[str, Any]] = None):
        self.db_config = db_config or DEFAULT_DB_CONFIG

    def connect(self):
        return psycopg2.connect(**self.db_config)

    def init_schema(self, schema_path: str = "src/database/schema.sql") -> None:
        sql = Path(schema_path).read_text(encoding="utf-8")
        with self.connect() as conn:
            with conn.cursor() as cur:
                cur.execute(sql)

    def upsert_functions(self, records: Iterable[Dict[str, Any]], batch_size: int = 500) -> int:
        rows = []
        total = 0

        for record in records:
            rows.append(self._record_to_row(record))
            if len(rows) >= batch_size:
                total += self._insert_rows(rows)
                rows = []

        if rows:
            total += self._insert_rows(rows)
        return total

    def insert_dataset_examples(self, records: Iterable[Dict[str, Any]], batch_size: int = 500) -> int:
        rows = []
        total = 0
        for record in records:
            rows.append(self._dataset_record_to_row(record))
            if len(rows) >= batch_size:
                total += self._insert_dataset_rows(rows)
                rows = []
        if rows:
            total += self._insert_dataset_rows(rows)
        return total

    def _insert_rows(self, rows: List[tuple]) -> int:
        sql = """
            INSERT INTO functions (
                library_name, module_name, func_name, full_name, signature,
                docstring, parameters, returns, examples, source_url, version,
                chunk_id, embedding
            )
            VALUES %s
            ON CONFLICT (chunk_id) DO UPDATE SET
                library_name = EXCLUDED.library_name,
                module_name = EXCLUDED.module_name,
                func_name = EXCLUDED.func_name,
                full_name = EXCLUDED.full_name,
                signature = EXCLUDED.signature,
                docstring = EXCLUDED.docstring,
                parameters = EXCLUDED.parameters,
                returns = EXCLUDED.returns,
                examples = EXCLUDED.examples,
                source_url = EXCLUDED.source_url,
                version = EXCLUDED.version,
                embedding = EXCLUDED.embedding,
                updated_at = now();
        """
        template = """
            (%s, %s, %s, %s, %s, %s, %s::jsonb, %s, %s, %s, %s, %s, %s::vector)
        """
        with self.connect() as conn:
            with conn.cursor() as cur:
                execute_values(cur, sql, rows, template=template)
        return len(rows)

    def _insert_dataset_rows(self, rows: List[tuple]) -> int:
        sql = """
            INSERT INTO dataset_examples (
                split, task_type, library_name, query, positive_chunk_id,
                positive_function, answer, hard_negatives, metadata
            )
            VALUES %s;
        """
        template = "(%s, %s, %s, %s, %s, %s, %s, %s::jsonb, %s::jsonb)"
        with self.connect() as conn:
            with conn.cursor() as cur:
                execute_values(cur, sql, rows, template=template)
        return len(rows)

    @staticmethod
    def _record_to_row(record: Dict[str, Any]) -> tuple:
        parameters = record.get("parameters") or {}
        if not isinstance(parameters, str):
            parameters = json.dumps(parameters)

        embedding = record.get("embedding")
        embedding_value = None
        if embedding is not None:
            embedding_value = "[" + ",".join(str(float(x)) for x in embedding) + "]"

        return (
            record["library_name"],
            record.get("module_name"),
            record.get("func_name") or record["full_name"].split(".")[-1],
            record["full_name"],
            record.get("signature"),
            record["docstring"],
            parameters,
            record.get("returns"),
            record.get("examples"),
            record.get("source_url"),
            record.get("version"),
            record.get("chunk_id") or record["full_name"],
            embedding_value,
        )

    @staticmethod
    def _dataset_record_to_row(record: Dict[str, Any]) -> tuple:
        return (
            record["split"],
            record.get("task_type", "retrieval"),
            record["library_name"],
            record["query"],
            record.get("positive_chunk_id"),
            record["positive_function"],
            record.get("answer"),
            json.dumps(record.get("hard_negatives") or []),
            json.dumps(record.get("metadata") or {}),
        )
