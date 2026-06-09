import json
import argparse
from pathlib import Path
from typing import List

from sentence_transformers import InputExample, SentenceTransformer, losses
from torch.utils.data import DataLoader


def load_training_examples(path: str = "data/training/retriever_train.jsonl") -> List[InputExample]:
    examples = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            query = row["query"]
            positive = row["positive"]
            examples.append(InputExample(texts=[query, positive]))
    return examples


def fine_tune_retriever(
    train_path: str = "data/training/retriever_train.jsonl",
    base_model: str = "BAAI/bge-small-en-v1.5",
    output_path: str = "models/checkpoints/bge-small-code-assistant",
    batch_size: int = 16,
    epochs: int = 1,
) -> None:
    """
    Fine-tunes a small bi-encoder with MultipleNegativesRankingLoss.
    Training JSONL format: {"query": "...", "positive": "function doc text ..."}
    """

    model = SentenceTransformer(base_model)
    examples = load_training_examples(train_path)
    dataloader = DataLoader(examples, shuffle=True, batch_size=batch_size)
    train_loss = losses.MultipleNegativesRankingLoss(model)

    model.fit(
        train_objectives=[(dataloader, train_loss)],
        epochs=epochs,
        warmup_steps=max(len(dataloader) // 10, 1),
        output_path=output_path,
        show_progress_bar=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-path", default="data/training/retriever_train.jsonl")
    parser.add_argument("--base-model", default="BAAI/bge-small-en-v1.5")
    parser.add_argument("--output-path", default="models/checkpoints/bge-small-code-assistant")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=1)
    args = parser.parse_args()
    fine_tune_retriever(
        train_path=args.train_path,
        base_model=args.base_model,
        output_path=args.output_path,
        batch_size=args.batch_size,
        epochs=args.epochs,
    )
