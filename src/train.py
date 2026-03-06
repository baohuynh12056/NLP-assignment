import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributed.nn.functional import all_gather
from tqdm import tqdm
from transformers import get_linear_schedule_with_warmup
from accelerate import Accelerator

from config import Config
from data_loader import get_data_loaders
from model import RDModel


# -----------------------------------------------------------------------
# Loss functions
# -----------------------------------------------------------------------

def info_nce_with_hard_negatives(
    def_emb: torch.Tensor,
    word_emb: torch.Tensor,
    accelerator: Accelerator,
    temperature: float = Config.TEMPERATURE,
    num_hard_negatives: int = Config.NUM_HARD_NEGATIVES,
) -> torch.Tensor:
    """
    InfoNCE loss augmented with hard negative weighting.
    Multi-GPU Optimized: Gathers embeddings across all GPUs to maximize 
    in-batch negatives, which is crucial for contrastive learning performance.
    """
    # Gather embeddings from all processes to construct a global batch
    if getattr(accelerator.state, "num_processes", 1) > 1:
        def_emb = torch.cat(all_gather(def_emb), dim=0)
        word_emb = torch.cat(all_gather(word_emb), dim=0)

    B = def_emb.size(0)
    labels = torch.arange(B, device=def_emb.device)

    # Cosine similarity
    sim = torch.matmul(def_emb, word_emb.T) / temperature  # (B, B)

    with torch.no_grad():
        diag_mask = torch.eye(B, dtype=torch.bool, device=def_emb.device)
        neg_sim = sim.clone().masked_fill(diag_mask, float("-inf"))
        _, hard_idx = neg_sim.topk(min(num_hard_negatives, B - 1), dim=-1)

        hard_weight = torch.zeros_like(sim)
        hard_weight.scatter_(1, hard_idx, 0.5)   
        hard_weight.masked_fill_(diag_mask, 0.0) 

    sim = sim + hard_weight

    loss_d2w = nn.CrossEntropyLoss()(sim, labels)
    loss_w2d = nn.CrossEntropyLoss()(sim.T, labels)
    return (loss_d2w + loss_w2d) / 2


def matryoshka_loss(
    model: nn.Module,
    def_ids: torch.Tensor,
    def_mask: torch.Tensor,
    word_ids: torch.Tensor,
    word_mask: torch.Tensor,
    accelerator: Accelerator,
) -> torch.Tensor:
    """Matryoshka Representation Learning (MRL) loss."""
    
    # Unwrap model to safely access custom methods like 'encode_all_mrl_dims'
    # without breaking distributed execution hooks.
    unwrapped_model = accelerator.unwrap_model(model)
    def_embs = unwrapped_model.encode_all_mrl_dims(def_ids, def_mask)
    word_embs = unwrapped_model.encode_all_mrl_dims(word_ids, word_mask)

    total_loss = torch.tensor(0.0, device=def_ids.device)
    for dim, weight in zip(Config.MRL_DIMS, Config.MRL_WEIGHTS):
        dim_loss = info_nce_with_hard_negatives(
            def_embs[dim], word_embs[dim], accelerator
        )
        total_loss = total_loss + weight * dim_loss

    return total_loss


# -----------------------------------------------------------------------
# Training / evaluation loops
# -----------------------------------------------------------------------

def train_epoch(
    model: nn.Module,
    dataloader,
    optimizer: optim.Optimizer,
    scheduler,
    accelerator: Accelerator,
) -> float:
    model.train()
    total_loss = 0.0

    # Disable tqdm on non-main processes to avoid console clutter
    for batch in tqdm(dataloader, desc="Training", leave=False, disable=not accelerator.is_local_main_process):
        # Automatic device placement handled by Accelerator
        def_ids  = batch["def_input_ids"]
        def_mask = batch["def_attention_mask"]
        word_ids  = batch["word_input_ids"]
        word_mask = batch["word_attention_mask"]

        optimizer.zero_grad()

        # Autocast context is managed automatically by Accelerator
        loss = matryoshka_loss(model, def_ids, def_mask, word_ids, word_mask, accelerator)

        # Accelerator handles AMP scaling and backward pass
        accelerator.backward(loss)
        accelerator.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()
        scheduler.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)


@torch.no_grad()
def eval_epoch(model: nn.Module, dataloader, accelerator: Accelerator) -> float:
    model.eval()
    total_loss = 0.0

    for batch in tqdm(dataloader, desc="Evaluating", leave=False, disable=not accelerator.is_local_main_process):
        def_ids  = batch["def_input_ids"]
        def_mask = batch["def_attention_mask"]
        word_ids  = batch["word_input_ids"]
        word_mask = batch["word_attention_mask"]

        loss = matryoshka_loss(model, def_ids, def_mask, word_ids, word_mask, accelerator)
        total_loss += loss.item()

    return total_loss / len(dataloader)


# -----------------------------------------------------------------------
# Entry point
# -----------------------------------------------------------------------

def main() -> None:
    # Initialize Accelerator with Mixed Precision for T4 GPUs
    accelerator = Accelerator(mixed_precision="fp16")
    
    accelerator.print(f"[INFO] Starting training on {accelerator.num_processes} GPUs...")
    torch.manual_seed(Config.SEED)

    with accelerator.main_process_first():
        train_loader, val_loader, _ = get_data_loaders()

    model = RDModel(Config.MODEL_NAME)

    optimizer = optim.AdamW(model.parameters(), lr=Config.LEARNING_RATE, weight_decay=1e-2)

    total_steps = len(train_loader) * Config.EPOCHS
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.1 * total_steps),
        num_training_steps=total_steps,
    )

    # Prepare all objects for distributed processing
    model, optimizer, train_loader, val_loader, scheduler = accelerator.prepare(
        model, optimizer, train_loader, val_loader, scheduler
    )

    best_val_loss = float("inf")

    for epoch in range(Config.EPOCHS):
        accelerator.print(f"\n[INFO] Epoch {epoch + 1}/{Config.EPOCHS}")
        
        train_loss = train_epoch(model, train_loader, optimizer, scheduler, accelerator)
        val_loss   = eval_epoch(model, val_loader, accelerator)
        
        accelerator.print(f"[INFO] Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

        # Ensure only the main process saves the checkpoint to prevent data corruption
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            accelerator.wait_for_everyone()  # Sync all GPUs before saving
            
            if accelerator.is_main_process:
                unwrapped_model = accelerator.unwrap_model(model)
                torch.save(unwrapped_model.state_dict(), Config.MODEL_SAVE_PATH)
                accelerator.print(f"[INFO] Checkpoint saved → {Config.MODEL_SAVE_PATH}")

    accelerator.print("[INFO] Training complete.")


if __name__ == "__main__":
    main()