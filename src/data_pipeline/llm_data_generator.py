import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List

from openai import OpenAI

from data_pipeline.chunker import make_retriever_training_text
from data_pipeline.parsers.parsers import write_jsonl


DATASET_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "examples": {
            "type": "array",
            "minItems": 1,
            "maxItems": 5,
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "query": {"type": "string"},
                    "answer": {"type": "string"},
                    "difficulty": {"type": "string", "enum": ["basic", "intermediate", "advanced"]},
                    "tags": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["query", "answer", "difficulty", "tags"],
            },
        }
    },
    "required": ["examples"],
}


def load_jsonl(path: str) -> List[Dict[str, Any]]:
    rows = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def generate_examples_for_functions(
    functions: Iterable[Dict[str, Any]],
    model: str = "gpt-4o",
    examples_per_function: int = 3,
) -> List[Dict[str, Any]]:
    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    output = []
    for function_record in functions:
        generated = generate_for_function(client, function_record, model, examples_per_function)
        for item in generated:
            output.append({
                "split": "train",
                "task_type": "retrieval_and_answer",
                "library_name": function_record["library_name"],
                "query": item["query"],
                "positive": make_retriever_training_text(function_record),
                "positive_chunk_id": function_record.get("chunk_id"),
                "positive_function": function_record["full_name"],
                "answer": item["answer"],
                "hard_negatives": [],
                "metadata": {
                    "generator": model,
                    "difficulty": item["difficulty"],
                    "tags": item["tags"],
                },
            })
    return output


def generate_for_function(
    client: OpenAI,
    function_record: Dict[str, Any],
    model: str,
    examples_per_function: int,
) -> List[Dict[str, Any]]:
    prompt = build_generation_prompt(function_record, examples_per_function)
    response = client.responses.create(
        model=model,
        input=[
            {
                "role": "system",
                "content": (
                    "You generate high-quality code-assistance retrieval datasets. "
                    "Return only examples that can be answered from the provided function documentation."
                ),
            },
            {"role": "user", "content": prompt},
        ],
        text={
            "format": {
                "type": "json_schema",
                "name": "code_assistance_dataset_examples",
                "strict": True,
                "schema": DATASET_SCHEMA,
            }
        },
    )
    data = json.loads(response.output_text)
    return data["examples"]


def build_generation_prompt(function_record: Dict[str, Any], examples_per_function: int) -> str:
    return f"""
Create {examples_per_function} diverse user questions and short grounded answers.

Function metadata:
- library_name: {function_record.get("library_name")}
- full_name: {function_record.get("full_name")}
- signature: {function_record.get("signature")}
- parameters: {json.dumps(function_record.get("parameters") or {}, ensure_ascii=False)}

Documentation:
{function_record.get("docstring", "")[:5000]}

Rules:
- Questions should look like real developer questions.
- Include exact function intent, common parameter usage, and one edge-case style question when possible.
- Answers must not invent behavior not present in the documentation.
- Prefer English questions for retrieval training.
""".strip()


def split_generated_examples(
    examples: List[Dict[str, Any]],
    test_ratio: float = 0.2,
    benchmark_ratio: float = 0.1,
) -> Dict[str, List[Dict[str, Any]]]:
    train = []
    test = []
    benchmark = []
    for idx, example in enumerate(examples):
        if idx % int(1 / benchmark_ratio) == 0:
            item = dict(example)
            item["split"] = "benchmark"
            benchmark.append(item)
        elif idx % int(1 / test_ratio) == 0:
            item = dict(example)
            item["split"] = "test"
            test.append(item)
        else:
            item = dict(example)
            item["split"] = "train"
            train.append(item)
    return {"train": train, "test": test, "benchmark": benchmark}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--functions", default="data/chunks/functions.jsonl")
    parser.add_argument("--output-dir", default="data/training")
    parser.add_argument("--model", default="gpt-4o")
    parser.add_argument("--limit", type=int, default=200)
    parser.add_argument("--examples-per-function", type=int, default=3)
    args = parser.parse_args()

    functions = load_jsonl(args.functions)[: args.limit]
    generated = generate_examples_for_functions(functions, args.model, args.examples_per_function)
    splits = split_generated_examples(generated)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(splits["train"], str(output_dir / "llm_retriever_train.jsonl"))
    write_jsonl(splits["test"], str(output_dir / "llm_retriever_test.jsonl"))
    write_jsonl(splits["benchmark"], "data/benchmark/llm_benchmark.jsonl")
    print(
        "Generated "
        f"{len(splits['train'])} train, {len(splits['test'])} test, "
        f"{len(splits['benchmark'])} benchmark examples."
    )


if __name__ == "__main__":
    main()
