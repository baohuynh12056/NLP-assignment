import os

import torch


class Config:
    DICTIONARY_URL = (
        "https://raw.githubusercontent.com/adambom/dictionary/master/dictionary.json"
    )

    LOCAL_DATA_FILE = "dictionary_data.json"

    # Directory for saving artifacts
    OUTPUT_DIR = "./checkpoints"

    # Path to save the trained model weights
    MODEL_SAVE_PATH = os.path.join(OUTPUT_DIR, "best_reverse_dict_model.bin")

    # Path to save pre-computed word vectors
    EMBEDDINGS_SAVE_PATH = os.path.join(OUTPUT_DIR, "word_embeddings_index.pt")

    # Backbone model
    MODEL_NAME = "bert-base-uncased"

    # Max length for input definitions/queries
    MAX_LEN_DEF = 96

    # Max length for target words
    MAX_LEN_WORD = 16

    # Batch size
    BATCH_SIZE = 64

    EPOCHS = 5
    LEARNING_RATE = 3e-5

    # Temperature scaling to control distribution sharpness
    TEMPERATURE = 0.05

    # Auto-detect hardware accelerator
    if torch.cuda.is_available():
        DEVICE = torch.device("cuda")  # NVIDIA GPUs
    else:
        DEVICE = torch.device("cpu")

    NUM_WORKERS = 2  # CPU threads for data loading
    SEED = 42  # Random seed for reproducibility

    # Default number of results to return
    TOP_K = 5


# Ensure checkpoint directory exists
os.makedirs(Config.OUTPUT_DIR, exist_ok=True)

print(f"[INFO] Config loaded. Device: {Config.DEVICE}")