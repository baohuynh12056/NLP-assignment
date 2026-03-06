import os

import spacy
import torch
import torch.nn as nn
import torch.nn.functional as F
from nltk.corpus import wordnet as wn
from transformers import AutoModel, AutoTokenizer
from wordfreq import zipf_frequency

from config import Config
from model import RDModel

# -----------------------------------------------------------------------
# Lightweight Cross-Encoder for Re-ranking
# -----------------------------------------------------------------------


class CrossEncoder(nn.Module):
    """
    Lightweight cross-encoder that scores (definition, word) pairs jointly.
    Takes the CLS token of a concatenated "[DEF] <SEP> <WORD>" sequence
    and maps it to a relevance scalar. Used to re-rank the bi-encoder's
    top-K candidates with higher precision.

    This is the standard two-stage retrieve-then-rerank pipeline:
      Stage 1 — Bi-encoder (fast, approximate, ANN-friendly)
      Stage 2 — Cross-encoder (accurate, quadratic cost, applied to top-K only)
    """

    CROSS_ENCODER_PATH = os.path.join(Config.OUTPUT_DIR, "cross_encoder.bin")

    def __init__(self, model_name: str = Config.MODEL_NAME):
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        self.scorer = nn.Linear(self.bert.config.hidden_size, 1)

    def forward(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        cls = self.bert(
            input_ids=input_ids, attention_mask=attention_mask
        ).last_hidden_state[:, 0]
        return self.scorer(cls).squeeze(-1)  # (B,)


# -----------------------------------------------------------------------
# Inference Engine
# -----------------------------------------------------------------------


class RDEngine:
    """
    Full inference pipeline for the reverse dictionary system.

    Search pipeline:
      1. Query preprocessing  — normalise + extract keywords via POS tagging
      2. Query expansion       — augment with WordNet synonyms / hypernyms
      3. Embedding blending    — weighted sum of full-query and keyword vectors
      4. Bi-encoder retrieval  — ANN-style top-K via dot product on pre-built index
      5. Frequency prior       — boost common English words slightly
      6. Cross-encoder re-rank — joint scoring of (query, candidate) pairs
    """

    def __init__(self, mrl_dim: int = Config.EMBEDDING_DIM):
        """
        Args:
            mrl_dim: Embedding dimension to use for retrieval.
                     Smaller dims (e.g. 64) are faster; 256 is most accurate.
        """
        print(f"[INFO] Initialising RDEngine (MRL dim={mrl_dim})...")
        self.device = Config.DEVICE
        self.mrl_dim = mrl_dim

        self.tokenizer = AutoTokenizer.from_pretrained(Config.MODEL_NAME)

        # Bi-encoder (main retrieval model)
        self.model = RDModel(Config.MODEL_NAME).to(self.device)
        self._load_weights(self.model, Config.MODEL_SAVE_PATH)
        self.model.eval()

        # Cross-encoder (re-ranker)
        self.cross_encoder = CrossEncoder(Config.MODEL_NAME).to(self.device)
        if os.path.exists(CrossEncoder.CROSS_ENCODER_PATH):
            self._load_weights(self.cross_encoder, CrossEncoder.CROSS_ENCODER_PATH)
        else:
            print(
                "[WARN] Cross-encoder weights not found; re-ranking will use bi-encoder scores only."
            )
        self.cross_encoder.eval()

        # NLP tools
        self.nlp = spacy.load("en_core_web_sm")

        # Word index
        self.words: list[str] = []
        self.word_embs: torch.Tensor | None = None
        self.word_frequencies: torch.Tensor | None = None
        self._load_index()

    # ------------------------------------------------------------------
    # Initialisation helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _load_weights(model: nn.Module, path: str) -> None:
        if os.path.exists(path):
            state = torch.load(path, map_location="cpu")
            model.load_state_dict(state)
        else:
            print(f"[WARN] Weights not found at {path}; using random init.")

    def _load_index(self) -> None:
        if not os.path.exists(Config.EMBEDDINGS_SAVE_PATH):
            raise FileNotFoundError(
                f"Embedding index not found at {Config.EMBEDDINGS_SAVE_PATH}. "
                "Run the indexing script first."
            )

        data = torch.load(Config.EMBEDDINGS_SAVE_PATH, map_location=self.device)
        self.words = data["words"]
        full_embs: torch.Tensor = data["embeddings"]  # (N, EMBEDDING_DIM)

        # Truncate to requested MRL dimension and re-normalise
        self.word_embs = F.normalize(full_embs[:, : self.mrl_dim], p=2, dim=-1)

        print(f"[INFO] Computing word frequency priors for {len(self.words)} words...")
        freqs = [zipf_frequency(w, "en") for w in self.words]
        self.word_frequencies = torch.tensor(
            freqs, dtype=torch.float32, device=self.device
        )

        print(f"[INFO] Index loaded: {len(self.words)} words @ dim={self.mrl_dim}.")

    # ------------------------------------------------------------------
    # Query processing
    # ------------------------------------------------------------------

    def _expand_with_wordnet(self, tokens: list[str]) -> list[str]:
        """
        Enrich a keyword list with WordNet synonyms and direct hypernyms.
        Expansion is kept small (≤2 entries per token) to avoid topic drift.
        """
        expanded = list(tokens)
        for token in tokens:
            synsets = wn.synsets(token, lang="eng")
            if not synsets:
                continue
            syn = synsets[0]  # most common sense

            # Add one synonym (excluding the token itself)
            for lemma in syn.lemmas():
                candidate = lemma.name().replace("_", " ")
                if candidate != token:
                    expanded.append(candidate)
                    break

            # Add direct hypernym
            hypernyms = syn.hypernyms()
            if hypernyms:
                hp_name = hypernyms[0].lemmas()[0].name().replace("_", " ")
                expanded.append(hp_name)

        return expanded

    def _tokenize_for_model(self, text: str) -> dict[str, torch.Tensor]:
        return self.tokenizer(
            text,
            return_tensors="pt",
            max_length=Config.MAX_LEN_DEF,
            padding=True,
            truncation=True,
        ).to(self.device)

    @torch.no_grad()
    def _get_embedding(self, text: str) -> torch.Tensor:
        enc = self._tokenize_for_model(text)
        return self.model.encode_text(
            enc["input_ids"], enc["attention_mask"], dim=self.mrl_dim
        )

    def _build_query_embedding(self, raw_query: str) -> tuple[torch.Tensor, str]:
        """
        Three-vector blending strategy:
          v_full     — full definition query
          v_keywords — POS-filtered keyword phrase
          v_expanded — WordNet-expanded keyword phrase

        Final embedding = α·v_full + β·v_keywords + γ·v_expanded
        """
        clean = " ".join(raw_query.strip().lower().split())

        # Extract content words via POS tagging
        doc = self.nlp(clean)
        core_tokens = [
            tok.lemma_
            for tok in doc
            if tok.pos_ in ("NOUN", "VERB", "ADJ", "PROPN") and not tok.is_stop
        ]
        keyword_str = " ".join(core_tokens) if core_tokens else clean

        # WordNet expansion
        expanded_tokens = self._expand_with_wordnet(core_tokens)
        expanded_str = " ".join(expanded_tokens) if expanded_tokens else keyword_str

        v_full = self._get_embedding(clean)

        if not core_tokens or clean == keyword_str:
            # No keywords found; fall back to full query only
            debug_info = clean
            return v_full, debug_info

        v_keywords = self._get_embedding(keyword_str)
        v_expanded = self._get_embedding(expanded_str)

        # Weights: full query dominates, keywords second, expansion third
        alpha, beta, gamma = 0.70, 0.20, 0.10
        v_final = alpha * v_full + beta * v_keywords + gamma * v_expanded
        v_final = F.normalize(v_final, p=2, dim=-1)

        debug_info = f"{clean} | keywords: [{keyword_str}] | expanded: [{expanded_str}]"
        return v_final, debug_info

    # ------------------------------------------------------------------
    # Search pipeline
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _bi_encoder_retrieve(
        self, query_emb: torch.Tensor, k: int
    ) -> tuple[list[str], list[float]]:
        """Stage 1: fast dot-product retrieval over the full index."""
        cosine_scores = torch.matmul(query_emb, self.word_embs.T).squeeze(0)

        # Min-max normalise to [0, 1]
        lo, hi = cosine_scores.min(), cosine_scores.max()
        cosine_scores = (cosine_scores - lo) / (hi - lo + 1e-8)

        # Frequency prior (log-scale to soften dominance)
        freq_bonus = Config.RERANK_WEIGHT * torch.log1p(self.word_frequencies)
        combined = cosine_scores + freq_bonus

        top_scores, top_idx = torch.topk(combined, k=k)
        words = [self.words[i] for i in top_idx.tolist()]
        scores = top_scores.tolist()
        return words, scores

    def _cross_encoder_rerank(
        self, query: str, candidates: list[str], bi_scores: list[float]
    ) -> list[tuple[str, float]]:
        """
        Stage 2: joint (query, word) scoring via cross-encoder.
        Falls back to bi-encoder scores if cross-encoder weights are absent.
        """
        if not os.path.exists(CrossEncoder.CROSS_ENCODER_PATH):
            return sorted(zip(candidates, bi_scores), key=lambda x: x[1], reverse=True)

        pairs = [f"{query} [SEP] {w}" for w in candidates]
        enc = self.tokenizer(
            pairs,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=Config.MAX_LEN_DEF + Config.MAX_LEN_WORD + 3,
        ).to(self.device)

        with torch.no_grad():
            ce_scores = self.cross_encoder(enc["input_ids"], enc["attention_mask"])
            ce_scores = torch.sigmoid(ce_scores).tolist()

        # Interpolate bi-encoder and cross-encoder scores
        lam = 0.4  # weight on cross-encoder
        final = [
            (w, (1 - lam) * bi + lam * ce)
            for w, bi, ce in zip(candidates, bi_scores, ce_scores)
        ]
        return sorted(final, key=lambda x: x[1], reverse=True)

    def search(
        self, raw_query: str, top_k: int = Config.TOP_K
    ) -> list[tuple[str, float]]:
        """
        Full search pipeline. Returns a ranked list of (word, score) tuples.
        """
        query_emb, debug_info = self._build_query_embedding(raw_query)
        print(f"[DEBUG] {debug_info}")

        # Stage 1 — bi-encoder retrieval (top RETRIEVAL_K candidates)
        candidates, bi_scores = self._bi_encoder_retrieve(query_emb, Config.RETRIEVAL_K)

        # Stage 2 — cross-encoder re-ranking
        reranked = self._cross_encoder_rerank(raw_query, candidates, bi_scores)

        # Exclude candidate words if any of their individual words are in the query
        query_words = set(raw_query.lower().split())
        reranked = [
            (w, s)
            for w, s in reranked
            if not any(part in query_words for part in w.replace("-", " ").split())
        ]

        return reranked[:top_k]


# -----------------------------------------------------------------------
# CLI demo
# -----------------------------------------------------------------------


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Reverse Dictionary CLI")
    parser.add_argument(
        "--dim",
        type=int,
        default=Config.EMBEDDING_DIM,
        choices=Config.MRL_DIMS,
        help="MRL embedding dimension (trade speed vs. accuracy).",
    )
    parser.add_argument("--top_k", type=int, default=Config.TOP_K)
    args = parser.parse_args()

    engine = RDEngine(mrl_dim=args.dim)
    print(f"\nEngine ready (dim={args.dim}). Type 'q' to quit.\n")

    while True:
        try:
            query = input("Describe the word: ").strip()
        except (KeyboardInterrupt, EOFError):
            break
        if query.lower() == "q" or not query:
            break

        results = engine.search(query, top_k=args.top_k)

        print("-" * 50)
        for rank, (word, score) in enumerate(results, 1):
            print(f"  {rank}. {word:<20} score={score:.4f}")
        print("-" * 50)


if __name__ == "__main__":
    main()
