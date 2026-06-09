import argparse
import json
import random
from pathlib import Path
from typing import Any, Dict, Iterable, List

from data_pipeline.chunker import (
    DEFAULT_TARGET_LIBRARIES,
    FunctionDocChunker,
    attach_embeddings,
    make_retriever_training_text,
)
from data_pipeline.parsers.parsers import (
    introspect_libraries,
    parse_docs_directory,
    write_jsonl,
)


BENCHMARK_SEED: List[Dict[str, Any]] = [
    {"library_name": "pandas", "question": "How do I join two dataframes on a shared column?", "expected_function": "pandas.merge", "tags": ["dataframe", "join"]},
    {"library_name": "pandas", "question": "How can I fill missing dataframe values?", "expected_function": "pandas.DataFrame.fillna", "tags": ["missing-values"]},
    {"library_name": "numpy", "question": "How do I create evenly spaced numbers over an interval?", "expected_function": "numpy.linspace", "tags": ["array"]},
    {"library_name": "numpy", "question": "How do I stack arrays vertically?", "expected_function": "numpy.vstack", "tags": ["array", "stack"]},
    {"library_name": "sklearn", "question": "How do I split arrays into train and test subsets?", "expected_function": "sklearn.model_selection.train_test_split", "tags": ["ml", "split"]},
    {"library_name": "sklearn", "question": "How do I standardize features by removing the mean?", "expected_function": "sklearn.preprocessing.StandardScaler", "tags": ["preprocessing"]},
    {"library_name": "torch", "question": "How do I create a tensor from Python data?", "expected_function": "torch.tensor", "tags": ["tensor"]},
    {"library_name": "torch", "question": "How do I disable gradient calculation during inference?", "expected_function": "torch.no_grad", "tags": ["inference"]},
    {"library_name": "tensorflow", "question": "How do I build a sequential neural network model?", "expected_function": "tensorflow.keras.Sequential", "tags": ["deep-learning"]},
    {"library_name": "matplotlib", "question": "How do I draw a simple line chart?", "expected_function": "matplotlib.pyplot.plot", "tags": ["visualization"]},
    {"library_name": "scipy", "question": "How do I optimize a scalar function?", "expected_function": "scipy.optimize.minimize", "tags": ["optimization"]},
    {"library_name": "seaborn", "question": "How do I draw a histogram with a density curve?", "expected_function": "seaborn.histplot", "tags": ["visualization"]},
    {"library_name": "requests", "question": "How do I send an HTTP GET request?", "expected_function": "requests.get", "tags": ["http"]},
    {"library_name": "fastapi", "question": "How do I declare an API application?", "expected_function": "fastapi.FastAPI", "tags": ["api"]},
]


def build_seed_benchmark(output_path: str = "data/benchmark/seed_queries.jsonl") -> None:
    write_jsonl(BENCHMARK_SEED, output_path)


def build_introspection_dataset(
    output_path: str = "data/parsed/function_docs.jsonl",
    libraries: List[str] | None = None,
    max_members_per_library: int = 1200,
) -> List[Dict[str, Any]]:
    libraries = libraries or DEFAULT_TARGET_LIBRARIES
    raw_records = introspect_libraries(libraries, max_members_per_library)
    chunks = FunctionDocChunker().chunk_records(raw_records)
    write_jsonl(chunks, output_path)
    return chunks


def build_docs_dataset(
    docs_root: str = "data/raw",
    output_path: str = "data/chunks/functions.jsonl",
    libraries: List[str] | None = None,
) -> List[Dict[str, Any]]:
    libraries = libraries or DEFAULT_TARGET_LIBRARIES
    raw_records = []
    for library_name in libraries:
        library_docs_dir = Path(docs_root) / library_name
        if library_docs_dir.exists():
            raw_records.extend(parse_docs_directory(str(library_docs_dir), library_name))
    chunks = FunctionDocChunker().chunk_records(raw_records)
    write_jsonl(chunks, output_path)
    return chunks


def build_retriever_datasets(
    function_records: Iterable[Dict[str, Any]],
    output_dir: str = "data/training",
    test_ratio: float = 0.2,
    seed: int = 42,
    examples_per_function: int = 3,
) -> Dict[str, List[Dict[str, Any]]]:
    records = list(function_records)
    random.Random(seed).shuffle(records)
    split_idx = int(len(records) * (1 - test_ratio))
    train_records = records[:split_idx]
    test_records = records[split_idx:]

    train = [
        example
        for record in train_records
        for example in function_to_training_examples(record, "train", examples_per_function)
    ]
    test = [
        example
        for record in test_records
        for example in function_to_training_examples(record, "test", examples_per_function)
    ]
    benchmark = [seed_to_dataset_example(row) for row in BENCHMARK_SEED]

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    write_jsonl(train, str(output / "retriever_train.jsonl"))
    write_jsonl(test, str(output / "retriever_test.jsonl"))
    write_jsonl(benchmark, "data/benchmark/benchmark_queries.jsonl")

    return {"train": train, "test": test, "benchmark": benchmark}


def function_to_training_examples(
    record: Dict[str, Any],
    split: str,
    examples_per_function: int,
) -> List[Dict[str, Any]]:
    examples = []
    for query in build_synthetic_queries(record)[:examples_per_function]:
        examples.append({
            "split": split,
            "task_type": "retrieval",
            "library_name": record["library_name"],
            "query": query,
            "positive": make_retriever_training_text(record),
            "positive_chunk_id": record.get("chunk_id"),
            "positive_function": record["full_name"],
            "answer": None,
            "hard_negatives": [],
            "metadata": {"generator": "library_docstring_templates"},
        })
    return examples


def seed_to_dataset_example(row: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "split": "benchmark",
        "task_type": "retrieval",
        "library_name": row["library_name"],
        "query": row["question"],
        "positive_chunk_id": None,
        "positive_function": row["expected_function"],
        "answer": row.get("expected_answer"),
        "hard_negatives": [],
        "metadata": {"tags": row.get("tags", []), "generator": "seed"},
    }


def build_synthetic_queries(record: Dict[str, Any]) -> List[str]:
    full_name = record["full_name"]
    func_name = record["func_name"]
    library_name = record["library_name"]
    signature = f" with signature {record['signature']}" if record.get("signature") else ""
    first_sentence = (record.get("docstring") or "").strip().split(".")[0]
    intent = first_sentence[:140].strip()
    queries = [
        f"How do I use {full_name}{signature}?",
        f"What does {full_name} do in {library_name}?",
        f"When should I use {func_name}?",
    ]
    if intent:
        queries.append(f"Which {library_name} function should I use to {intent.lower()}?")
    return queries


def ingest_introspection_dataset(
    db_config: Dict[str, Any] | None = None,
    embedding_model: str = "BAAI/bge-small-en-v1.5",
) -> int:
    records = build_introspection_dataset()
    records = attach_embeddings(records, model_name=embedding_model)
    from database.db_manager import DatabaseManager

    db = DatabaseManager(db_config)
    db.init_schema()
    return db.upsert_functions(records)


def load_jsonl(path: str) -> List[Dict[str, Any]]:
    rows = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def ingest_functions_file(
    functions_path: str = "data/chunks/functions.jsonl",
    db_config: Dict[str, Any] | None = None,
    embedding_model: str = "BAAI/bge-small-en-v1.5",
) -> int:
    records = load_jsonl(functions_path)
    records = attach_embeddings(records, model_name=embedding_model)
    from database.db_manager import DatabaseManager

    db = DatabaseManager(db_config)
    db.init_schema()
    return db.upsert_functions(records)


def ingest_dataset_examples(
    paths: List[str],
    db_config: Dict[str, Any] | None = None,
) -> int:
    records = []
    for path in paths:
        records.extend(load_jsonl(path))
    from database.db_manager import DatabaseManager

    db = DatabaseManager(db_config)
    db.init_schema()
    return db.insert_dataset_examples(records)


def build_all(args: argparse.Namespace) -> None:
    docs_chunks = []
    if args.use_raw_docs:
        docs_chunks.extend(build_docs_dataset(args.docs_root, args.functions_output, args.libraries))
    if args.include_introspection:
        docs_chunks.extend(build_introspection_dataset(libraries=args.libraries))
        write_jsonl(docs_chunks, args.functions_output)
    build_seed_benchmark()
    build_retriever_datasets(docs_chunks, examples_per_function=args.examples_per_function)
    if args.ingest:
        count = ingest_functions_file(args.functions_output, embedding_model=args.embedding_model)
        print(f"Inserted/updated {count} function chunks.")
    if args.ingest_datasets:
        count = ingest_dataset_examples(args.dataset_files)
        print(f"Inserted {count} dataset examples.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--docs-root", default="data/raw")
    parser.add_argument("--functions-output", default="data/chunks/functions.jsonl")
    parser.add_argument("--embedding-model", default="BAAI/bge-small-en-v1.5")
    parser.add_argument("--libraries", nargs="+", default=DEFAULT_TARGET_LIBRARIES)
    parser.add_argument("--include-introspection", action="store_true", default=True)
    parser.add_argument("--use-raw-docs", action="store_true")
    parser.add_argument("--examples-per-function", type=int, default=3)
    parser.add_argument("--ingest", action="store_true")
    parser.add_argument("--ingest-datasets", action="store_true")
    parser.add_argument(
        "--dataset-files",
        nargs="+",
        default=[
            "data/training/retriever_train.jsonl",
            "data/training/retriever_test.jsonl",
            "data/benchmark/benchmark_queries.jsonl",
        ],
    )
    build_all(parser.parse_args())
