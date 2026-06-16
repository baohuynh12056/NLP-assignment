import json
import random
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Dict

from sentence_transformers import SentenceTransformer, losses
from sentence_transformers.evaluation import InformationRetrievalEvaluator
from sentence_transformers.training_args import SentenceTransformerTrainingArguments
from sentence_transformers.trainer import SentenceTransformerTrainer
from datasets import Dataset
from .lora_utils import apply_selective_lora
from sentence_transformers import InputExample
import shutil
from safetensors.torch import load_file

def to_input_examples(data: List[Dict]) -> List[InputExample]:
    """Convert raw dicts to SentenceTransformer InputExamples (Supports Hard Negatives)"""
    examples = []
    for row in data:
        query = row["query"]
        # Handle both list (BGE format) and string formats
        positive = row["pos"][0] if "pos" in row else (row["positive"][0] if isinstance(row.get("positive"), list) else row["positive"])
        texts = [query, positive]
        
        # If hard negatives exist, append the first one for MultipleNegativesRankingLoss
        negatives = row.get("neg", []) or row.get("hard_negatives", [])
        if negatives:
            texts.append(negatives[0])
            
        examples.append(InputExample(texts=texts))
    return examples

def load_raw_data(path: str) -> List[Dict]:
    """Safely load JSONL dataset"""
    data = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                data.append(json.loads(line))
    return data

def prepare_ir_evaluator(data: List[Dict], name: str = 'val') -> InformationRetrievalEvaluator:
    """Build Evaluator and FILTER OUT unnecessary metrics to keep logs clean."""
    queries, corpus, relevant_docs = {}, {}, {}
    
    for i, row in enumerate(data):
        q_id, c_id = f"q_{name}_{i}", f"c_{name}_{i}"
        
        queries[q_id] = row["query"]
        positive = row["pos"][0] if "pos" in row else (row["positive"][0] if isinstance(row.get("positive"), list) else row["positive"])
        corpus[c_id] = positive
        relevant_docs[q_id] = {c_id}
        
        negatives = row.get("neg", []) or row.get("hard_negatives", [])
        for j, neg in enumerate(negatives):
            corpus[f"c_{name}_{i}_neg_{j}"] = neg

    # LIMIT METRICS: Only calculate Cosine Similarity for selected K values
    return InformationRetrievalEvaluator(
        queries, corpus, relevant_docs, name=name,
        main_score_function="cosine", # Ignore Euclidean, Manhattan, Dot Product
        accuracy_at_k=[1, 3],          # Only Val Cosine Accuracy@1, 3
        precision_recall_at_k=[1, 3],  # Only Val Cosine Precision/Recall@1, 3
        ndcg_at_k=[10],                # Only Val Cosine Ndcg@10
        mrr_at_k=[10],                 # Only Val Cosine Mrr@10
        map_at_k=[100],                # Only Val Cosine Map@100
    )

def plot_training_history(log_history: List[Dict], output_path: str):
    """Parse HuggingFace Trainer logs to plot Loss and Metrics curves."""
    # Extract Training Loss
    train_steps = [log["step"] for log in log_history if "loss" in log]
    train_loss = [log["loss"] for log in log_history if "loss" in log]
    
    # Extract Evaluation Metrics
    eval_steps = [log["step"] for log in log_history if "eval_val_cosine_mrr@10" in log]
    val_mrr = [log["eval_val_cosine_mrr@10"] for log in log_history if "eval_val_cosine_mrr@10" in log]
    val_ndcg = [log["eval_val_cosine_ndcg@10"] for log in log_history if "eval_val_cosine_ndcg@10" in log]
    
    if not train_steps:
        print("Warning: No training logs found to plot.")
        return

    plt.figure(figsize=(14, 5))
    
    # Subplot 1: Training Loss
    plt.subplot(1, 2, 1)
    plt.plot(train_steps, train_loss, color="tomato", label="Training Loss")
    plt.title("Training Loss Curve")
    plt.xlabel("Steps")
    plt.ylabel("Loss")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend()
    
    # Subplot 2: Validation Metrics
    if eval_steps:
        plt.subplot(1, 2, 2)
        plt.plot(eval_steps, val_mrr, marker="o", color="dodgerblue", label="Val MRR@10")
        plt.plot(eval_steps, val_ndcg, marker="s", color="mediumseagreen", label="Val NDCG@10")
        plt.title("Validation Metrics Curve")
        plt.xlabel("Steps")
        plt.ylabel("Score")
        plt.grid(True, linestyle="--", alpha=0.6)
        plt.legend()
        
    plt.tight_layout()
    plot_file = Path(output_path) / "training_metrics_plot.png"
    plt.savefig(plot_file)
    print(f"Training plot successfully saved to: {plot_file}")
    plt.show()
def train_lora_retriever(
    model: SentenceTransformer,
    train_path: str,
    test_path: str = None,
    output_path: str = "models/checkpoints/bge-lora",
    batch_size: int = 16,
    epochs: int = 30,
    lr: float = 2e-4,
    val_ratio: float = 0.3
) -> None:
    print(f"Loading raw dataset from {train_path}...")
    raw_data = load_raw_data(train_path)
    
    # 1. SPLIT DATA
    random.shuffle(raw_data)
    val_size = max(int(len(raw_data) * val_ratio), 1)
    val_data = raw_data[:val_size]
    train_data = raw_data[val_size:]
    
    # 2. FORMAT DATA FOR HUGGINGFACE TRAINER
    def format_to_hf_dataset(data):
        formatted = []
        for row in data:
            item = {
                "anchor": row["query"],
                "positive": row["pos"][0] if "pos" in row else (row["positive"][0] if isinstance(row.get("positive"), list) else row["positive"])
            }
            negatives = row.get("neg", []) or row.get("hard_negatives", [])
            if negatives:
                item["negative"] = negatives[0]
            formatted.append(item)
        return Dataset.from_list(formatted)

    train_dataset = format_to_hf_dataset(train_data)
    val_dataset = format_to_hf_dataset(val_data)
    
    # 3. SETUP EVALUATOR
    val_evaluator = prepare_ir_evaluator(val_data, name='val')
    evaluation_steps = max(len(train_dataset) // batch_size // 2, 1)

    # 4. CONFIGURE TRAINING ARGUMENTS
    args = SentenceTransformerTrainingArguments(
        output_dir=output_path,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=lr,
        warmup_steps=max(len(train_dataset) // batch_size // 10, 1),
        logging_steps=1,             
        eval_strategy="steps",
        eval_steps=evaluation_steps,
        save_strategy="steps",
        save_steps=evaluation_steps,
        save_total_limit=2,          
        
        # ❌ QUAN TRỌNG: Tắt để tránh Bug làm mất Wrapper PEFT của SentenceTransformer
        load_best_model_at_end=False, 
        
        metric_for_best_model="eval_val_cosine_mrr@10",
        greater_is_better=True,
        remove_unused_columns=False
    )

    train_loss = losses.MultipleNegativesRankingLoss(model)

    # 5. INITIALIZE TRAINER
    trainer = SentenceTransformerTrainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        loss=train_loss,
        evaluator=val_evaluator,
    )

    print("\n--- STARTING LORA FINE-TUNING ---")
    trainer.train()
    
    best_ckpt = trainer.state.best_model_checkpoint
    dest_adapter = Path(output_path) / "lora_adapter"
    
    if best_ckpt:
        print(f"\nTraining completed. Extracting best adapter from checkpoint: {best_ckpt}")
        # Lõi SentenceTransformer lưu PEFT adapter tại thư mục '0_Transformer'
        src_adapter = Path(best_ckpt) / "0_Transformer"
        
        if dest_adapter.exists():
            shutil.rmtree(dest_adapter)
            
        shutil.copytree(src_adapter, dest_adapter)
        print(f"LoRA adapter được lưu tại: {dest_adapter}")
        
        # Nạp lại Tạ (Weights) tốt nhất vào mô hình hiện tại để chạy Test Set
        try:
            best_adapter_weights = dest_adapter / "adapter_model.safetensors"
            state_dict = load_file(best_adapter_weights)
            model[0].auto_model.load_state_dict(state_dict, strict=False)
            print("Đã nạp thành công trọng số tốt nhất vào RAM để chấm điểm Test.")
        except Exception as e:
            print(f"Lưu ý khi nạp tạ: {e}")
    else:
        # Fallback nếu epoch quá ngắn không sinh ra checkpoint
        peft_model = model[0].auto_model
        peft_model.save_pretrained(dest_adapter)
        print(f"LoRA adapter saved to: {dest_adapter}")

    # 7. PLOT METRICS
    plot_training_history(trainer.state.log_history, output_path)
    
    # 8. FINAL TEST EVALUATION
    if test_path and Path(test_path).exists():
        print(f"\n--- RUNNING FINAL EVALUATION ON TEST SET: {test_path} ---")
        test_data = load_raw_data(test_path)
        test_evaluator = prepare_ir_evaluator(test_data, name='test')
        test_metrics = test_evaluator(model, output_path=output_path)
        
        print("\n--- Final Test Set Metrics ---")
        for metric, value in test_metrics.items():
            print(f" - {metric}: {value:.4f}")