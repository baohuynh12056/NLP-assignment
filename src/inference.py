import os

import spacy
import torch
from transformers import AutoTokenizer
from wordfreq import zipf_frequency

from .config import Config
from .model import RDModel


class RDEngine:
    def __init__(self):
        print("Initializing Engine...")
        self.device = Config.DEVICE
        self.tokenizer = AutoTokenizer.from_pretrained(Config.MODEL_NAME)
        self.model = RDModel(Config.MODEL_NAME).to(self.device)

        # Load NLP model for query preprocessing
        self.nlp = spacy.load("en_core_web_sm")

        # Weight for word frequency
        self.lambda_freq = 0.015

        # Load main model
        if os.path.exists(Config.MODEL_SAVE_PATH):
            state_dict = torch.load(Config.MODEL_SAVE_PATH, map_location=self.device)
            self.model.load_state_dict(state_dict)
            self.model.eval()
        else:
            print("[ERROR] Model weights not found! Using random initialization")

        self.words = []
        self.word_embs = None
        self.word_frequencies = None

        self.load_index()

    def load_index(self):
        if os.path.exists(Config.EMBEDDINGS_SAVE_PATH):
            data = torch.load(Config.EMBEDDINGS_SAVE_PATH, map_location=self.device)
            self.words = data["words"]
            self.word_embs = data["embeddings"]
        else:
            print("[ERROR] Index not found!")

        # Compute zipf frequencies for all words
        print("Computing word frequencies...")
        freqs = [zipf_frequency(w, "en") for w in self.words]
        self.word_frequencies = torch.tensor(freqs, dtype=torch.float32).to(self.device)

        print(f"Loaded {len(self.words)} words into index")

    def process_and_encode(self, raw_query):
        """
        Processing pipeline: Nomalizing -> Extracting -> Blending
        """
        # Normalizing
        clean_query = " ".join(raw_query.strip().lower().split())

        # Extract core keywords for augmentation
        doc = self.nlp(clean_query)
        core_tokens = [
            token.text
            for token in doc
            if token.pos_ in ["NOUN", "VERB", "ADJ", "PROPN"] and not token.is_stop
        ]
        keyword_query = " ".join(core_tokens)

        # Helper function to encode text to tensor
        def get_embedding(text):
            inputs = self.tokenizer(
                clean_query,
                return_tensors="pt",
                max_length=Config.MAX_LEN_DEF,
                truncation=True,
                paddings=True,
            ).to(self.device)
            return self.model.encode_text(inputs["input_ids"], inputs["attention_mask"])

        # Encoding
        v_original = get_embedding(clean_query)

        v_keywords = get_embedding(keyword_query)

        # Fallback if no keywords found
        if not core_tokens or clean_query == keyword_query:
            return v_original, clean_query

        # Vector blending
        alpha = 0.75
        # 75% weight on the original, 25% weight on keywords
        v_final = (alpha * v_original) + ((1 - alpha) * v_keywords)

        return v_final, f"{clean_query} (Keywords: {keyword_query})"

    def search(self, raw_query, top_k=5):
        "Search pipeline: Retrieval (top 100) -> Re-ranking (top 5)"
        with torch.no_grad():
            query_emb, debug_query_info = self.process_and_encode(raw_query)
            print(f"Query info: '{debug_query_info}'")

            # Calculate cosine similarity scores
            cosine_scores = torch.matmul(query_emb, self.word_embs.T).squeeze(0)

            # Normalize base scores to [0, 1]
            min_score, max_score = cosine_scores.min(), cosine_scores.max()
            cosine_scores = (cosine_scores - min_score) / (max_score - min_score + 1e-8)

            # Fetch the top 100 most semantically relevant words.
            retrieve_k = 100
            top_scores, top_indices = torch.topk(cosine_scores, k=retrieve_k)

            # Re-rank the top 100 candidates using frequency prior.
            candidates = []
            rerank_weight = 0.18

            for score, idx in zip(
                top_scores.tolist(), top_indices.tolist(), strict=True
            ):
                word = self.words[int(idx)]
                freq = self.word_frequencies[int(idx)].item()

                # Re-rank core = cosine score + (weight * word frequency)
                rerank_score = float(score) + (rerank_weight * freq)
                candidates.append((word, rerank_score))

            candidates.sort(key=lambda x: x[1], reverse=True)

            results = []
            for word, final_score in candidates[:top_k]:
                results.append((word, final_score))

            return results


def main():
    engine = RDEngine()
    print("\nEngine ready! Type 'q' to quit.")

    while True:
        query = input("\nDescribe the word: ").strip()
        if query.lower() == "q":
            break
        if not query:
            continue

        results = engine.search(query, top_k=5)

        print("-" * 50)
        for i, (word, score) in enumerate(results):
            print(f"{i + 1}. {word:<15} (Score: {score:.4f})")
        print("-" * 50)


if __name__ == "__main__":
    main()