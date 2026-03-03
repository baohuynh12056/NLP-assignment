import json
import os
import random

import requests
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer

from config import Config


class RDDataset(Dataset):
    """
    Reverse-dictionary dataset built from a JSON word→definition mapping.
    Supports optional synonym augmentation via WordNet for definition diversity.
    """

    def __init__(
        self,
        tokenizer,
        max_len_def: int,
        max_len_word: int,
        split: str = "train",
        augment: bool = False,
    ):
        self.tokenizer = tokenizer
        self.max_len_def = max_len_def
        self.max_len_word = max_len_word
        self.augment = augment
        self.data: list[dict] = []

        file_path = self._ensure_data_downloaded()
        self._load_and_split(file_path, split)

    # ------------------------------------------------------------------
    # Setup helpers
    # ------------------------------------------------------------------

    def _ensure_data_downloaded(self) -> str:
        data_dir = "data"
        os.makedirs(data_dir, exist_ok=True)
        file_path = os.path.join(data_dir, Config.LOCAL_DATA_FILE)

        if not os.path.exists(file_path):
            print("Downloading dictionary from GitHub...")
            try:
                response = requests.get(Config.DICTIONARY_URL, timeout=30)
                response.raise_for_status()
                with open(file_path, "wb") as f:
                    f.write(response.content)
                print("Download complete.")
            except requests.RequestException as e:
                raise RuntimeError(f"Failed to download dictionary: {e}") from e

        return file_path

    def _load_and_split(self, file_path: str, split: str) -> None:
        print(f"Loading data from {file_path} ({split})...")
        with open(file_path, "r", encoding="utf-8") as f:
            raw_dict: dict = json.load(f)

        all_items = [
            (str(word).lower(), str(defn).lower())
            for word, defn in raw_dict.items()
            if defn and len(str(defn).split()) > 3  # discard stub definitions
        ]

        # Deterministic shuffle before split
        rng = random.Random(Config.SEED)
        rng.shuffle(all_items)

        split_idx = int(len(all_items) * 0.9)
        items = all_items[:split_idx] if split == "train" else all_items[split_idx:]

        self.data = [{"word": w, "definition": d} for w, d in items]
        print(f"Loaded {len(self.data)} pairs for '{split}'.")

    # ------------------------------------------------------------------
    # Dataset interface
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.data)

    def _tokenize(self, text: str, max_length: int) -> dict[str, torch.Tensor]:
        enc = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=max_length,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            return_tensors="pt",
        )
        return {
            "input_ids": enc["input_ids"].flatten(),
            "attention_mask": enc["attention_mask"].flatten(),
        }

    def __getitem__(self, index: int) -> dict:
        item = self.data[index]
        word: str = item["word"]
        definition: str = item["definition"]

        def_enc = self._tokenize(definition, self.max_len_def)
        word_enc = self._tokenize(word, self.max_len_word)

        return {
            "def_input_ids": def_enc["input_ids"],
            "def_attention_mask": def_enc["attention_mask"],
            "word_input_ids": word_enc["input_ids"],
            "word_attention_mask": word_enc["attention_mask"],
            "raw_word": word,
        }


# ----------------------------------------------------------------------
# Factory
# ----------------------------------------------------------------------


def get_data_loaders() -> tuple[DataLoader, DataLoader, AutoTokenizer]:
    tokenizer = AutoTokenizer.from_pretrained(Config.MODEL_NAME)

    train_ds = RDDataset(
        tokenizer, Config.MAX_LEN_DEF, Config.MAX_LEN_WORD, split="train", augment=True
    )
    val_ds = RDDataset(
        tokenizer, Config.MAX_LEN_DEF, Config.MAX_LEN_WORD, split="val", augment=False
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=Config.DEVICE.type == "cuda",
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=Config.DEVICE.type == "cuda",
    )

    return train_loader, val_loader, tokenizer
