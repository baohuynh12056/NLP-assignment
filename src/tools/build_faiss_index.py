import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import faiss
import numpy as np
import psycopg2
from psycopg2.extras import RealDictCursor
from sentence_transformers import SentenceTransformer

from utils.config_loader import GLOBAL_CONFIG
from utils.logger import get_logger


logger = get_logger(__name__)


def load_function_rows(db_config: Dict[str, Any]) -> List[Dict[str, Any]]:
    sql = """
        SELECT id, func_name, library_name, docstring, parameters
        FROM functions
        WHERE docstring IS NOT NULL AND length(trim(docstring)) > 0
        ORDER BY id;
    """
    with psycopg2.connect(**db_config) as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(sql)
            return list(cur.fetchall())


def build_search_text(row: Dict[str, Any], doc_limit: int) -> str:
    docstring = (row.get("docstring") or "").strip()
    summary = next((line.strip() for line in docstring.splitlines() if line.strip()), "")
    parameters = row.get("parameters") or {}
    if isinstance(parameters, str):
        try:
            parameters = json.loads(parameters)
        except json.JSONDecodeError:
            parameters = {}
    param_names = " ".join(parameters.keys()) if isinstance(parameters, dict) else ""
    return " ".join(
        [
            str(row.get("library_name", "")),
            str(row.get("func_name", "")),
            summary,
            param_names,
            docstring[:doc_limit],
        ]
    )


def row_to_record(row: Dict[str, Any]) -> Dict[str, Any]:
    parameters = row.get("parameters") or {}
    if isinstance(parameters, str):
        try:
            parameters = json.loads(parameters)
        except json.JSONDecodeError:
            parameters = {}

    function_name = str(row.get("func_name", ""))
    library_name = str(row.get("library_name", ""))
    full_name = function_name
    if library_name and not function_name.startswith(f"{library_name}."):
        full_name = f"{library_name}.{function_name}"

    return {
        "id": str(row.get("id")),
        "content": row.get("docstring") or "",
        "metadata": {
            "func_name": function_name,
            "full_name": full_name,
            "library_name": library_name,
            "parameters": parameters if isinstance(parameters, dict) else {},
        },
    }


def build_index(
    embed_model_path: str,
    db_config: Dict[str, Any],
    index_path: Path,
    metadata_path: Path,
    batch_size: int,
    doc_limit: int,
) -> None:
    rows = load_function_rows(db_config)
    if not rows:
        raise RuntimeError("No function rows found in PostgreSQL. Ingest data before building FAISS.")

    logger.info(f"Loaded {len(rows)} function rows from PostgreSQL.")
    texts = [build_search_text(row, doc_limit) for row in rows]
    records = [row_to_record(row) for row in rows]

    logger.info(f"Loading embedding model: {embed_model_path}")
    model = SentenceTransformer(embed_model_path)
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=True,
    ).astype("float32")
    embeddings = np.ascontiguousarray(embeddings)

    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)

    index_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(index_path))
    with metadata_path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    size_mb = (index_path.stat().st_size + metadata_path.stat().st_size) / 1_000_000
    logger.info(
        f"Saved FAISS index to {index_path} and metadata to {metadata_path} ({size_mb:.1f} MB)."
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a local FAISS index from PostgreSQL functions.")
    parser.add_argument("--index-path", default=None)
    parser.add_argument("--metadata-path", default=None)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--doc-limit", type=int, default=1400)
    args = parser.parse_args()

    retriever_cfg = GLOBAL_CONFIG.get("retriever", {})
    db_config = retriever_cfg.get("database", {})
    build_index(
        embed_model_path=retriever_cfg.get("embed_model_path", "BAAI/bge-small-en-v1.5"),
        db_config=db_config,
        index_path=Path(args.index_path or retriever_cfg.get("index_path", "models/faiss/functions.index")),
        metadata_path=Path(
            args.metadata_path
            or retriever_cfg.get("metadata_path", "models/faiss/functions_metadata.jsonl")
        ),
        batch_size=args.batch_size,
        doc_limit=args.doc_limit,
    )


if __name__ == "__main__":
    main()
