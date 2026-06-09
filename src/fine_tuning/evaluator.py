import json
import argparse
from pathlib import Path
from typing import Any, Dict, Iterable, List


def load_benchmark(path: str = "data/benchmark/seed_queries.jsonl") -> List[Dict[str, Any]]:
    records = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def evaluate_retriever(
    retriever,
    benchmark: Iterable[Dict[str, Any]],
    k_values: tuple[int, ...] = (1, 3, 5, 10),
) -> Dict[str, float]:
    rows = list(benchmark)
    hits = {k: 0 for k in k_values}
    reciprocal_rank_total = 0.0

    for row in rows:
        question = row.get("question") or row["query"]
        expected = (row.get("expected_function") or row["positive_function"]).lower()
        filters = {"library_name": row["library_name"]} if row.get("library_name") else None
        chunks = retriever.retrieve(question, top_k=max(k_values), filters=filters)
        ranked_names = [chunk.metadata.get("func_name", "").lower() for chunk in chunks]

        rank = None
        for idx, name in enumerate(ranked_names, start=1):
            if expected == name or expected.endswith(name) or name.endswith(expected):
                rank = idx
                break

        if rank is not None:
            reciprocal_rank_total += 1.0 / rank
            for k in k_values:
                if rank <= k:
                    hits[k] += 1

    total = max(len(rows), 1)
    metrics = {f"hit@{k}": hits[k] / total for k in k_values}
    metrics["mrr"] = reciprocal_rank_total / total
    metrics["total"] = float(len(rows))
    return metrics


def print_metrics(metrics: Dict[str, float]) -> None:
    for key, value in metrics.items():
        if key == "total":
            print(f"{key}: {int(value)}")
        else:
            print(f"{key}: {value:.4f}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", default="data/benchmark/benchmark_queries.jsonl")
    parser.add_argument("--embedding-model", default="BAAI/bge-small-en-v1.5")
    args = parser.parse_args()

    from database.db_manager import DEFAULT_DB_CONFIG
    from models.retriever import PGHybridRetriever

    benchmark = load_benchmark(args.benchmark)
    retriever = PGHybridRetriever(DEFAULT_DB_CONFIG, embed_model=args.embedding_model)
    metrics = evaluate_retriever(retriever, benchmark)
    print_metrics(metrics)


if __name__ == "__main__":
    main()
