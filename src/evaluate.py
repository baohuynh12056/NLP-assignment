import time

from tqdm import tqdm
from transformers import AutoTokenizer

from config import Config
from data_loader import RDDataset
from inference import RDEngine


class IREvaluator:
    """
    Evaluator for Information Retrieval metrics (Recall@K, MRR, Latency).
    """

    def __init__(
        self,
        engine: RDEngine,
        dataset: RDDataset,
        top_k_list: list[int] = [1, 5, 10, 100],
    ):
        self.engine = engine
        self.dataset = dataset
        self.top_k_list = top_k_list
        self.max_k = max(top_k_list)

    def evaluate(self, use_cross_encoder: bool = True, num_samples: int = None) -> dict:
        """Runs end-to-end evaluation on the dataset."""
        data_to_eval = self.dataset.data
        if num_samples:
            data_to_eval = data_to_eval[:num_samples]

        print(f"\n[INFO] Starting evaluation on {len(data_to_eval)} samples...")
        print(
            f"[INFO] Config -> MRL Dim: {self.engine.mrl_dim} | Cross-Encoder: {use_cross_encoder}"
        )

        metrics = {f"R@{k}": 0.0 for k in self.top_k_list}
        metrics["MRR"] = 0.0

        start_time = time.time()

        for item in tqdm(data_to_eval, desc="Evaluating"):
            target_word = item["word"].lower()
            query_text = item["definition"]

            if not use_cross_encoder:
                # Bypass cross-encoder for bi-encoder baseline
                query_emb, _ = self.engine._build_query_embedding(query_text)
                results, _ = self.engine._bi_encoder_retrieve(query_emb, k=self.max_k)
            else:
                # Full pipeline: bi-encoder + cross-encoder reranking
                results_with_scores = self.engine.search(query_text, top_k=self.max_k)
                results = [w for w, _ in results_with_scores]

            # Compute metrics
            rank = None
            for idx, pred_word in enumerate(results):
                if pred_word == target_word:
                    rank = idx + 1
                    break

            if rank is not None:
                metrics["MRR"] += 1.0 / rank
                for k in self.top_k_list:
                    if rank <= k:
                        metrics[f"R@{k}"] += 1.0

        total_time = time.time() - start_time
        num_queries = len(data_to_eval)

        # Aggregate results
        for k in self.top_k_list:
            metrics[f"R@{k}"] = (metrics[f"R@{k}"] / num_queries) * 100
        metrics["MRR"] = metrics["MRR"] / num_queries
        metrics["QPS"] = num_queries / total_time
        metrics["Latency_ms"] = (total_time / num_queries) * 1000

        return metrics


def print_metrics(title: str, metrics: dict) -> None:
    print(f"\n{'-' * 50}")
    print(f" {title}")
    print(f"{'-' * 50}")
    print(f" MRR        : {metrics['MRR']:.4f}")
    for k in [1, 5, 10]:
        if f"R@{k}" in metrics:
            print(f" Recall@{k:<2} : {metrics[f'R@{k}']:.2f}%")
    print(
        f" Latency    : {metrics['Latency_ms']:.1f} ms/query (QPS: {metrics['QPS']:.1f})"
    )
    print(f"{'-' * 50}\n")


def run_experiments() -> None:
    tokenizer = AutoTokenizer.from_pretrained(Config.MODEL_NAME)

    # Load validation set (no augmentation for clean ground truth evaluation)
    val_dataset = RDDataset(
        tokenizer, Config.MAX_LEN_DEF, Config.MAX_LEN_WORD, split="val", augment=False
    )

    # Subsample for rapid testing; set to None for full dataset evaluation
    NUM_SAMPLES = 500

    # --- Experiment 1: MRL Dimension Trade-off ---
    # Evaluates speed vs. accuracy degradation across truncated dimensions
    print("\n=== EXPERIMENT 1: MRL Dimension Trade-off (Bi-encoder baseline) ===")
    for dim in Config.MRL_DIMS:
        engine = RDEngine(mrl_dim=dim)
        evaluator = IREvaluator(engine, val_dataset)
        metrics = evaluator.evaluate(use_cross_encoder=False, num_samples=NUM_SAMPLES)
        print_metrics(f"Bi-encoder (Dim: {dim})", metrics)

    # --- Experiment 2: Cross-Encoder Re-ranking Impact ---
    # Measures the performance boost provided by the phase 2 re-ranker
    print("\n=== EXPERIMENT 2: Cross-Encoder Re-ranking ===")
    best_dim = max(Config.MRL_DIMS)
    engine = RDEngine(mrl_dim=best_dim)
    evaluator = IREvaluator(engine, val_dataset)

    metrics_full = evaluator.evaluate(use_cross_encoder=True, num_samples=NUM_SAMPLES)
    print_metrics(f"Full Pipeline (Bi-enc dim={best_dim} + Cross-enc)", metrics_full)


if __name__ == "__main__":
    run_experiments()
