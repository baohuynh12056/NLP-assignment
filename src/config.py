import os

import torch


class Config:
    DICTIONARY_URL = (
        "https://raw.githubusercontent.com/adambom/dictionary/master/dictionary.json"
    )
    LOCAL_DATA_FILE = "dictionary_data.json"

    OUTPUT_DIR = "./checkpoints"
    MODEL_SAVE_PATH = os.path.join(OUTPUT_DIR, "best_reverse_dict_model.bin")
    EMBEDDINGS_SAVE_PATH = os.path.join(OUTPUT_DIR, "word_embeddings_index.pt")

    # Backbone
    MODEL_NAME = "bert-base-uncased"

    MAX_LEN_DEF = 96
    MAX_LEN_WORD = 16
    BATCH_SIZE = 64
    EPOCHS = 5
    LEARNING_RATE = 3e-5

    # --- Contrastive Learning ---
    # Base temperature for InfoNCE
    TEMPERATURE = 0.05
    # Hard negative mining: number of hardest negatives to emphasize per sample
    NUM_HARD_NEGATIVES = 5

    # --- Matryoshka Representation Learning ---
    # Full embedding dim from projection head
    EMBEDDING_DIM = 256
    # Sub-dimensions trained simultaneously (MRL); must be ascending subsets of EMBEDDING_DIM
    MRL_DIMS = [64, 128, 256]
    # Loss weights per MRL dimension (smaller dims get lower weight)
    MRL_WEIGHTS = [0.2, 0.3, 0.5]

    # --- Inference ---
    TOP_K = 5
    RETRIEVAL_K = 100  # Candidate pool size before re-ranking
    RERANK_WEIGHT = 0.18  # Frequency prior weight in re-ranking
    QUERY_BLEND_ALPHA = 0.75  # Weight of full query vs. keyword-only embedding

    # Hardware
    if torch.cuda.is_available():
        DEVICE = torch.device("cuda")
    else:
        DEVICE = torch.device("cpu")

    NUM_WORKERS = 4
    SEED = 42


os.makedirs(Config.OUTPUT_DIR, exist_ok=True)
print(f"[INFO] Config loaded. Device: {Config.DEVICE}")
