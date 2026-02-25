import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast  # For Mixed Precision Training
from tqdm import tqdm
from transformers import get_linear_schedule_with_warmup

from .config import Config
from .data_loader import get_data_loaders
from .model import RDModel


def compute_info_nce_loss(def_embeddings, word_embeddings, temperature=0.07):
    """
    Calculates Contrastive Loss.
    """
    # Similarity Matrix (Cosine Similarity if vectors are normalized)
    # Shape: [Batch_Size, Batch_Size]
    logits = torch.matmul(def_embeddings, word_embeddings.T) / temperature

    # Labels: The diagonal elements are the positive pairs (i, i)
    labels = torch.arange(logits.shape[0], device=Config.DEVICE)

    # Cross Entropy calculates the loss maximizing the diagonal
    return nn.CrossEntropyLoss()(logits, labels)


def train_epoch(model, dataloader, optimizer, scheduler, scaler):
    """
    Runs one training epoch.
    """
    model.train()
    total_loss = 0

    loop = tqdm(dataloader, desc="Training", leave=False)

    for batch in loop:
        # Move inputs to device
        def_ids = batch["def_input_ids"].to(Config.DEVICE)
        def_mask = batch["def_attention_mask"].to(Config.DEVICE)
        word_ids = batch["word_input_ids"].to(Config.DEVICE)
        word_mask = batch["word_attention_mask"].to(Config.DEVICE)

        optimizer.zero_grad()

        # Mixed Precision Forward Pass (Save memory & Speed up)
        with autocast():
            def_emb = model.encode_text(def_ids, def_mask)
            word_emb = model.encode_word(word_ids, word_mask)
            loss = compute_info_nce_loss(def_emb, word_emb, Config.TEMPERATURE)

        # Backward Pass with Scaler
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()  # Update learning rate

        total_loss += loss.item()
        loop.set_postfix(loss=loss.item())

    return total_loss / len(dataloader)


def eval_epoch(model, dataloader):
    """
    Runs validation epoch (No gradient calculation).
    """
    model.eval()
    total_loss = 0

    loop = tqdm(dataloader, desc="Evaluating", leave=False)

    with torch.no_grad():
        for batch in loop:
            def_ids = batch["def_input_ids"].to(Config.DEVICE)
            def_mask = batch["def_attention_mask"].to(Config.DEVICE)
            word_ids = batch["word_input_ids"].to(Config.DEVICE)
            word_mask = batch["word_attention_mask"].to(Config.DEVICE)

            def_emb = model.encode_text(def_ids, def_mask)
            word_emb = model.encode_word(word_ids, word_mask)

            loss = compute_info_nce_loss(def_emb, word_emb, Config.TEMPERATURE)
            total_loss += loss.item()

    return total_loss / len(dataloader)


def main():
    print(f"[INFO] Starting training on {Config.DEVICE}...")

    # Prepare Data
    train_loader, val_loader, tokenizer = get_data_loaders()

    # Initialize Model
    model = RDModel(Config.MODEL_NAME).to(Config.DEVICE)

    # Optimization Setup
    optimizer = optim.AdamW(model.parameters(), lr=Config.LEARNING_RATE)
    scaler = GradScaler()  # FP16 Scaler

    # Scheduler: Warmup for 10% of steps, then linear decay
    total_steps = len(train_loader) * Config.EPOCHS
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.1 * total_steps),
        num_training_steps=total_steps,
    )

    # Training Loop
    best_val_loss = float("inf")

    for epoch in range(Config.EPOCHS):
        print(f"\n[INFO] Epoch {epoch + 1}/{Config.EPOCHS}")

        train_loss = train_epoch(model, train_loader, optimizer, scheduler, scaler)
        val_loss = eval_epoch(model, val_loader)

        print(f"[INFO] Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

        # Save Checkpoint
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            print(
                f"[INFO] Validation improved. Saving model to {Config.MODEL_SAVE_PATH}..."
            )
            torch.save(model.state_dict(), Config.MODEL_SAVE_PATH)

    print("[INFO] Training Complete!")


if __name__ == "__main__":
    main()