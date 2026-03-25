"""
search.py
---------
Two-stage retrieval pipeline:
  Stage 1 — Bi-encoder (Qwen3-Embedding-8B) + FAISS GPU → top RERANK_FETCH_K candidates
  Stage 2 — Cross-encoder (bge-reranker-large) → re-score → final top-K
  Stage 3 — MMR diversity filter (optional)
"""

import time
from dataclasses import dataclass

import faiss
import numpy as np
from sentence_transformers import CrossEncoder, SentenceTransformer

from config import (
    EMBED_MODEL,
    QUERY_INSTRUCTION,
    RERANKER_MODEL,
    RERANK_FETCH_K,
    SCORE_THRESHOLD,
    TOP_K,
)
from extract import FunctionRecord, ParamRecord
from index import list_indexes, load_index


# ── Result model ──────────────────────────────────────────────────────────────

@dataclass
class SearchResult:
    rank: int
    score: float          # bi-encoder cosine score (Stage 1)
    rerank_score: float   # cross-encoder score (Stage 2); -inf if skipped
    library: str
    full_name: str
    kind: str
    signature: str
    summary: str
    docstring: str
    params: list[ParamRecord]

    def pretty(self, show_params: int = 5) -> str:
        sep = "-" * 70
        score_str = (
            f"score={self.score:.4f}  rerank={self.rerank_score:.4f}"
            if self.rerank_score > -999
            else f"score={self.score:.4f}"
        )
        lines = [
            sep,
            f"  #{self.rank}  {self.full_name}{self.signature}",
            f"  {score_str}  |  kind={self.kind}  |  lib={self.library}",
            "",
            f"  {self.summary}",
        ]
        if self.params:
            lines.append("")
            lines.append("  Parameters:")
            for p in self.params[:show_params]:
                type_str = f" ({p.type_hint})" if p.type_hint else ""
                desc_str = f" - {p.description[:90]}" if p.description else ""
                lines.append(f"    * {p.name}{type_str}{desc_str}")
            if len(self.params) > show_params:
                lines.append(f"    ... and {len(self.params) - show_params} more")
        return "\n".join(lines)


def display_results(results: list[SearchResult], show_params: int = 5) -> None:
    if not results:
        print("  (no results)")
        return
    for r in results:
        print(r.pretty(show_params=show_params))
    print("-" * 70)


# ── MMR (Maximal Marginal Relevance) ──────────────────────────────────────────

def _mmr(
    query_vec         : np.ndarray,
    candidate_vecs    : np.ndarray,
    candidate_results : list[SearchResult],
    top_k             : int,
    lambda_           : float = 0.6,
) -> list[SearchResult]:
    """Re-rank for diversity. lambda_=1.0 → pure relevance, 0.0 → pure diversity."""
    selected_idx : list[int] = []
    remaining               = list(range(len(candidate_results)))
    query_sims              = candidate_vecs @ query_vec

    while len(selected_idx) < top_k and remaining:
        if not selected_idx:
            best = max(remaining, key=lambda i: query_sims[i])
        else:
            sel_vecs = candidate_vecs[selected_idx]
            scores   = [
                (i, lambda_ * query_sims[i]
                    - (1 - lambda_) * float(np.max(sel_vecs @ candidate_vecs[i])))
                for i in remaining
            ]
            best = max(scores, key=lambda x: x[1])[0]

        selected_idx.append(best)
        remaining.remove(best)

    reranked = [candidate_results[i] for i in selected_idx]
    for rank, r in enumerate(reranked, start=1):
        r.rank = rank
    return reranked


# ── Query expansion ───────────────────────────────────────────────────────────

def _expand_query(query: str) -> str:
    """Prepend domain hint for short queries to anchor embedding in code space."""
    if len(query.strip().split()) <= 4:
        return f"python function {query.strip()}"
    return query.strip()


# ── Safe reconstruct ──────────────────────────────────────────────────────────

def _reconstruct(index: faiss.Index, idx: int) -> np.ndarray:
    """Reconstruct a raw vector from the index by its position.
    GPU indexes (GpuIndexIVF) don't support reconstruct() — falls back to
    a zero vector in that case. MMR quality degrades slightly but won't crash."""
    try:
        vec = np.zeros(index.d, dtype="float32")
        index.reconstruct(idx, vec)
        return vec
    except Exception:
        # GpuIndexIVF does not support reconstruct — return zero vector as fallback
        return np.zeros(index.d, dtype="float32")


# ── Searcher ──────────────────────────────────────────────────────────────────

class FunctionSearcher:
    """
    Parameters
    ----------
    libraries : list[str] | None
        Libraries to search. None = all available indexes.
    use_mmr : bool
        Apply MMR diversity filter on final results.
    mmr_lambda : float
        MMR trade-off: 1.0 = relevance only, 0.0 = diversity only.
    use_reranker : bool
        Apply cross-encoder reranking (Stage 2). Set False to skip for speed.
    """

    def __init__(
        self,
        libraries    : list[str] | None = None,
        use_mmr      : bool  = True,
        mmr_lambda   : float = 0.6,
        use_reranker : bool  = True,
    ) -> None:
        self.use_mmr      = use_mmr
        self.mmr_lambda   = mmr_lambda
        self.use_reranker = use_reranker

        # Stage 1: bi-encoder
        print(f"Loading bi-encoder : {EMBED_MODEL}")
        self._model = SentenceTransformer(
            EMBED_MODEL,
            model_kwargs={
                "attn_implementation": "sdpa",   # PyTorch 2.9 built-in
            },
            tokenizer_kwargs={"padding_side": "left"},       # required for Qwen3
        )

        # Stage 2: cross-encoder reranker
        if use_reranker:
            print(f"Loading reranker   : {RERANKER_MODEL}")
            self._reranker = CrossEncoder(RERANKER_MODEL, max_length=512)
        else:
            self._reranker = None
            print("Reranker disabled.")

        # Load FAISS indexes (already on GPU via load_index)
        targets = libraries if libraries else list_indexes()
        self._stores: list[tuple[faiss.Index, list[FunctionRecord]]] = []

        for lib in targets:
            result = load_index(lib)    # load_index handles CPU→GPU transfer
            if result is None:
                print(f"  [missing] {lib} — run index.py first")
                continue
            index, records = result
            self._stores.append((index, records))
            location = "GPU" if faiss.get_num_gpus() > 0 else "CPU"
            print(f"  Loaded {lib:<25} ({index.ntotal} functions, {location})")

        if not self._stores:
            print("No indexes loaded. Run: python index.py")
        else:
            total = sum(idx.ntotal for idx, _ in self._stores)
            print(f"Ready: {len(self._stores)} libraries | {total:,} functions total")

    # ── Embed ─────────────────────────────────────────────────────────────────

    def _embed_query(self, query: str) -> np.ndarray:
        """Qwen3 requires the instruction prefix on queries (NOT on documents)."""
        prefixed = QUERY_INSTRUCTION + query
        vec = self._model.encode(
            [prefixed],
            normalize_embeddings = True,
            convert_to_numpy     = True,
            show_progress_bar    = False,
        )
        return vec[0].astype("float32")

    # ── Stage 1: FAISS search ─────────────────────────────────────────────────

    def _faiss_search(
        self,
        query_vec       : np.ndarray,
        fetch_k         : int,
        score_threshold : float,
        libraries       : list[str] | None,
    ) -> tuple[list[SearchResult], list[np.ndarray]]:
        """Return (results, raw_vectors) for top fetch_k candidates across all indexes."""
        all_results : list[SearchResult] = []
        all_vecs    : list[np.ndarray]   = []

        for index, records in self._stores:
            lib_name = records[0].library if records else "?"
            if libraries and lib_name not in libraries:
                continue

            k = min(fetch_k, index.ntotal)
            if k == 0:
                continue

            scores, indices = index.search(query_vec[np.newaxis, :], k)

            for score, idx in zip(scores[0], indices[0]):
                if idx < 0 or float(score) < score_threshold:
                    continue

                r       = records[idx]
                summary = next(
                    (ln.strip() for ln in r.docstring.splitlines() if ln.strip()),
                    "No description.",
                )
                all_results.append(SearchResult(
                    rank         = 0,
                    score        = float(score),
                    rerank_score = float("-inf"),
                    library      = r.library,
                    full_name    = r.full_name,
                    kind         = r.kind,
                    signature    = r.signature,
                    summary      = summary,
                    docstring    = r.docstring,
                    params       = r.params,
                ))

                # Reconstruct raw vector for MMR (GPU-safe fallback included)
                all_vecs.append(_reconstruct(index, int(idx)))

        # Global sort by bi-encoder score before reranking
        order       = np.argsort([-r.score for r in all_results])
        all_results = [all_results[i] for i in order]
        all_vecs    = [all_vecs[i]    for i in order]
        return all_results, all_vecs

    # ── Stage 2: Cross-encoder reranking ─────────────────────────────────────

    def _rerank(self, query: str, candidates: list[SearchResult]) -> list[SearchResult]:
        """Re-score each (query, document) pair with full attention cross-encoder."""
        pairs = [
            [query, f"{r.full_name}{r.signature}\n{r.docstring[:300]}"]
            for r in candidates
        ]
        scores = self._reranker.predict(pairs, show_progress_bar=False)
        for r, s in zip(candidates, scores):
            r.rerank_score = float(s)
        candidates.sort(key=lambda r: r.rerank_score, reverse=True)
        return candidates

    # ── Public search API ─────────────────────────────────────────────────────

    def search(
        self,
        query           : str,
        top_k           : int   = TOP_K,
        score_threshold : float = SCORE_THRESHOLD,
        libraries       : list[str] | None = None,
        use_mmr         : bool | None      = None,
    ) -> list[SearchResult]:
        """
        Search for Python functions matching a natural-language description.

        Parameters
        ----------
        query : str
            Natural-language description of the desired function.
        top_k : int
            Number of results to return.
        score_threshold : float
            Min bi-encoder cosine similarity to enter Stage 2.
        libraries : list[str] | None
            Restrict search to specific libraries.
        use_mmr : bool | None
            Override instance-level MMR setting for this call.

        Returns
        -------
        list[SearchResult]  ranked best → worst.
        """
        if not self._stores:
            return []

        t0 = time.perf_counter()

        # Expand & embed
        expanded  = _expand_query(query)
        query_vec = self._embed_query(expanded)

        # Stage 1: FAISS GPU search
        fetch_k          = RERANK_FETCH_K if self._reranker else top_k * 4
        candidates, vecs = self._faiss_search(query_vec, fetch_k, score_threshold, libraries)

        if not candidates:
            print(f"  '{query[:50]}' → 0 results ({time.perf_counter()-t0:.3f}s)")
            return []

        # Stage 2: Cross-encoder rerank
        if self._reranker and len(candidates) > 1:
            candidates = self._rerank(query, candidates)
            # Re-align vecs to match reranked order (needed for MMR)
            id_to_vec  = {id(candidates[i]): vecs[i] for i in range(min(len(vecs), len(candidates)))}
            vecs       = [id_to_vec.get(id(r), vecs[0]) for r in candidates]

        # Stage 3: MMR diversity filter
        _use_mmr = self.use_mmr if use_mmr is None else use_mmr
        if _use_mmr and len(candidates) > top_k and vecs:
            candidate_vecs = np.vstack(vecs[:len(candidates)])
            results = _mmr(query_vec, candidate_vecs, candidates, top_k, self.mmr_lambda)
        else:
            results = candidates[:top_k]
            for rank, r in enumerate(results, start=1):
                r.rank = rank

        elapsed = time.perf_counter() - t0
        stage   = (
            "bi+rerank+mmr" if self._reranker and _use_mmr else
            "bi+rerank"     if self._reranker               else
            "bi+mmr"        if _use_mmr                     else
            "bi"
        )
        print(f"  '{query[:45]}' → {len(results)} results  [{stage}]  ({elapsed:.3f}s)")
        return results

    # ── Convenience ───────────────────────────────────────────────────────────

    def loaded_libraries(self) -> list[str]:
        return [records[0].library for _, records in self._stores if records]

    def total_functions(self) -> int:
        return sum(idx.ntotal for idx, _ in self._stores)