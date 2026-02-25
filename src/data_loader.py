import json
import os

import requests
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer

from .config import Config


class RDDataset(Dataset):
    def __init__(self, tokenizer, max_len_defi, max_len_word, split="train"):
        self.tokenizer = tokenizer
        self.max_len_def = max_len_defi
        self.max_len_word = max_len_word
        self.data = []

        # Đường dẫn file local sẽ lưu
        data_dir = "data"
        os.makedirs(data_dir, exist_ok=True)
        file_path = os.path.join(data_dir, Config.LOCAL_DATA_FILE)

        # 1. Tải file từ GitHub nếu chưa có
        if not os.path.exists(file_path):
            print(" Downloading dictionary from GitHub...")
            try:
                response = requests.get(Config.DICTIONARY_URL)
                if response.status_code == 200:
                    with open(file_path, "wb") as f:
                        f.write(response.content)
                    print(" Download complete.")
                else:
                    print(f" Error downloading: Status {response.status_code}")
            except Exception as e:
                print(f" Network error: {e}")

        # 2. Đọc dữ liệu từ file JSON local
        print(f" Loading data from {file_path} ({split})...")
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                raw_dict = json.load(f)  # Format: {"WORD": "DEFINITION", ...}

            # Chuyển đổi dict thành list các cặp
            all_items = []
            for word, definition in raw_dict.items():
                # Lọc dữ liệu rác
                if definition and len(str(definition).split()) > 3:
                    all_items.append((str(word), str(definition)))

            # Chia train/val thủ công (90% train, 10% val)
            split_idx = int(len(all_items) * 0.9)

            if split == "train":
                items_to_use = all_items[:split_idx]
            else:
                items_to_use = all_items[split_idx:]

            for word, definition in items_to_use:
                self.data.append(
                    {"word": word.lower(), "definition": definition.lower()}
                )

            print(f" Loaded {len(self.data)} pairs for {split}.")

        except Exception as e:
            print(f" Error reading file: {e}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        item = self.data[index]
        defi = item["definition"]
        word = item["word"]

        # Tokenize Definition
        defi_encoder = self.tokenizer(
            defi,
            add_special_tokens=True,
            max_length=self.max_len_def,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            return_tensors="pt",
        )

        # Tokenize Word
        word_encoder = self.tokenizer(
            word,
            add_special_tokens=True,
            max_length=self.max_len_word,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            return_tensors="pt",
        )

        return {
            "def_input_ids": defi_encoder["input_ids"].flatten(),
            "def_attention_mask": defi_encoder["attention_mask"].flatten(),
            "word_input_ids": word_encoder["input_ids"].flatten(),
            "word_attention_mask": word_encoder["attention_mask"].flatten(),
            "raw_word": word,
        }


def get_data_loaders():
    tokenizer = AutoTokenizer.from_pretrained(Config.MODEL_NAME)

    # Train set
    train_dataset = RDDataset(
        tokenizer,
        Config.MAX_LEN_DEF,
        Config.MAX_LEN_WORD,
        split="train[:90%]",  # 90% for train
    )

    # Valid set
    val_dataset = RDDataset(
        tokenizer,
        Config.MAX_LEN_DEF,
        Config.MAX_LEN_WORD,
        split="train[90%:]",  # 10% for test
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
    )

    return train_loader, val_loader, tokenizer