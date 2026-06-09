import hashlib
from typing import Any, Dict, Iterable, List


DEFAULT_TARGET_LIBRARIES = [
    "pandas",
    "numpy",
    "sklearn",
    "torch",
    "tensorflow",
    "matplotlib",
    "scipy",
    "seaborn",
    "requests",
    "fastapi",
]


class FunctionDocChunker:
    """
    Converts parsed function documentation into retriever-ready records.
    One function usually becomes one chunk because API docs are already compact.
    """

    def __init__(self, min_doc_chars: int = 80):
        self.min_doc_chars = min_doc_chars

    def chunk_records(self, records: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
        chunks = []
        for record in records:
            docstring = (record.get("docstring") or "").strip()
            if len(docstring) < self.min_doc_chars:
                continue

            full_name = record["full_name"]
            library_name = record["library_name"].lower()
            chunk = {
                "library_name": library_name,
                "module_name": record.get("module_name"),
                "func_name": record.get("func_name") or full_name.split(".")[-1],
                "full_name": full_name,
                "signature": record.get("signature"),
                "docstring": docstring,
                "parameters": record.get("parameters") or {},
                "returns": record.get("returns"),
                "examples": record.get("examples"),
                "source_url": record.get("source_url"),
                "version": record.get("version"),
                "chunk_id": record.get("chunk_id") or self.make_chunk_id(library_name, full_name),
            }
            chunks.append(chunk)
        return chunks

    @staticmethod
    def make_chunk_id(library_name: str, full_name: str) -> str:
        raw = f"{library_name}:{full_name}".encode("utf-8")
        return hashlib.sha1(raw).hexdigest()


def attach_embeddings(
    records: List[Dict[str, Any]],
    model_name: str = "BAAI/bge-small-en-v1.5",
    batch_size: int = 64,
) -> List[Dict[str, Any]]:
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_name)
    texts = [format_embedding_text(record) for record in records]
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        normalize_embeddings=True,
        show_progress_bar=True,
    )
    for record, embedding in zip(records, embeddings):
        record["embedding"] = embedding.tolist()
    return records


def format_embedding_text(record: Dict[str, Any]) -> str:
    parts = [
        record.get("full_name", ""),
        record.get("signature", ""),
        record.get("docstring", ""),
        record.get("examples", ""),
    ]
    return "\n".join(part for part in parts if part)


def make_retriever_training_text(record: Dict[str, Any]) -> str:
    return (
        f"Function: {record.get('full_name', '')}\n"
        f"Signature: {record.get('signature', '')}\n"
        f"Parameters: {record.get('parameters', {})}\n"
        f"Documentation:\n{record.get('docstring', '')}"
    )
