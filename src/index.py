import argparse
import pickle
import time
from pathlib import Path

import faiss
import numpy as np
import torch
from sentence_transformers import SentenceTransformer

from config import BATCH_SIZE, DEFAULT_LIBRARIES, EMBED_MODEL, INDEX_DIR
from extract import FunctionRecord, extract_library

# Switch to approximate IVF search above this vector count
_IVF_THRESHOLD = 50_000
_IVF_NLIST     = 128


# ── FAISS index builders ──────────────────────────────────────────────────────

def _build_flat(embeddings: np.ndarray) -> faiss.Index:
    """Exact cosine search via inner product (vectors must be L2-normalized).
    IndexFlatIP natively supports reconstruct() — safe for GPU transfer."""
    dim   = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    return index


def _build_ivf(embeddings: np.ndarray) -> faiss.Index:
    """Approximate search for large corpora (faster, slightly less accurate)."""
    dim       = embeddings.shape[1]
    quantizer = faiss.IndexFlatIP(dim)
    index     = faiss.IndexIVFFlat(quantizer, dim, _IVF_NLIST, faiss.METRIC_INNER_PRODUCT)
    print(f"    Training IVFFlat (nlist={_IVF_NLIST}, n={len(embeddings)}) ...")
    index.train(embeddings)
    index.add(embeddings)
    index.nprobe = 16
    return index


def _select_index(embeddings: np.ndarray) -> faiss.Index:
    n = len(embeddings)
    if n >= _IVF_THRESHOLD:
        print(f"    {n} vectors → IndexIVFFlat (approximate)")
        return _build_ivf(embeddings)
    print(f"    {n} vectors → IndexFlatIP (exact)")
    return _build_flat(embeddings)


# ── FAISS GPU helpers ─────────────────────────────────────────────────────────

def _to_gpu(index: faiss.Index) -> faiss.Index:
    """Move CPU index to GPU 0 for fast search. Falls back silently if no GPU.
    useFloat16=True halves VRAM usage with minimal accuracy loss."""
    if faiss.get_num_gpus() == 0:
        return index
    res           = faiss.StandardGpuResources()
    co            = faiss.GpuClonerOptions()
    co.useFloat16 = True
    gpu_index     = faiss.index_cpu_to_gpu(res, 0, index, co)
    print(f"    FAISS index moved to GPU 0 (float16)")
    return gpu_index


def _to_cpu(index: faiss.Index) -> faiss.Index:
    """Convert GPU index back to CPU before saving.
    FAISS cannot write GPU indexes directly to disk."""
    try:
        return faiss.index_gpu_to_cpu(index)
    except Exception:
        return index    # already CPU


# ── Embedding ─────────────────────────────────────────────────────────────────

def _embed(model: SentenceTransformer, texts: list[str]) -> np.ndarray:
    """
    Encode texts into L2-normalized float32 vectors.
    Uses all available GPUs via multi-process pool (~2x faster on T4 x2).
    Falls back to single-GPU encode if only 1 GPU available.
    """
    n_gpus = torch.cuda.device_count()

    if n_gpus >= 2:
        # Spawn one process per GPU — each loads its own model copy independently
        # ~2x faster than single GPU on Kaggle T4 x2
        print(f"    Multi-GPU embedding ({n_gpus} GPUs)")
        devices = [f"cuda:{i}" for i in range(n_gpus)]
        pool    = model.start_multi_process_pool(target_devices=devices)
        try:
            embeddings = model.encode_multi_process(
                texts,
                pool                 = pool,
                batch_size           = BATCH_SIZE,
                normalize_embeddings = True,
            )
        finally:
            model.stop_multi_process_pool(pool)  # always clean up, even on error
    else:
        print(f"    Single-GPU embedding (cuda:0)")
        embeddings = model.encode(
            texts,
            batch_size           = BATCH_SIZE,
            show_progress_bar    = True,
            normalize_embeddings = True,
            convert_to_numpy     = True,
        )

    return np.array(embeddings, dtype="float32")


# ── I/O helpers ───────────────────────────────────────────────────────────────

def _index_paths(library: str) -> tuple[Path, Path]:
    return INDEX_DIR / f"{library}.index", INDEX_DIR / f"{library}.meta.pkl"


def _save(index: faiss.Index, records: list[FunctionRecord], library: str) -> None:
    idx_path, meta_path = _index_paths(library)
    faiss.write_index(_to_cpu(index), str(idx_path))   # must be CPU before writing
    with open(meta_path, "wb") as f:
        pickle.dump(records, f, protocol=pickle.HIGHEST_PROTOCOL)
    size_mb = (idx_path.stat().st_size + meta_path.stat().st_size) / 1e6
    print(f"    Saved → {idx_path.parent}  ({size_mb:.1f} MB)")


def load_index(library: str) -> tuple[faiss.Index, list[FunctionRecord]] | None:
    """Load index from disk, then move to GPU 0 for fast search."""
    idx_path, meta_path = _index_paths(library)
    if not idx_path.exists() or not meta_path.exists():
        return None

    index = faiss.read_index(str(idx_path))     # always loads to CPU first
    if hasattr(index, "nprobe"):
        index.nprobe = 16
    index = _to_gpu(index)                       # then move to GPU 0

    with open(meta_path, "rb") as f:
        records: list[FunctionRecord] = pickle.load(f)
    return index, records


# ── Public API ────────────────────────────────────────────────────────────────

def build_index(
    libraries    : list[str] = DEFAULT_LIBRARIES,
    force_rebuild: bool      = False,
) -> dict[str, int]:
    """
    Build FAISS indexes for all given libraries.
    Skips already-indexed libraries unless force_rebuild=True.
    Returns dict of library -> number of functions indexed.
    """
    INDEX_DIR.mkdir(parents=True, exist_ok=True)

    n_gpus = torch.cuda.device_count()
    print(f"GPUs available  : {n_gpus}")
    print(f"Embedding model : {EMBED_MODEL}")

    model = SentenceTransformer(
        EMBED_MODEL,
        device="cpu",
        model_kwargs={
            "attn_implementation": "sdpa",  # PyTorch 2.9 built-in
            "torch_dtype": torch.float16,
        },
        tokenizer_kwargs={"padding_side": "left"},  # required for Qwen3
    )

    report      = {}
    total_start = time.perf_counter()

    for lib in libraries:
        print("-" * 60)
        print(f"[{lib}]")

        idx_path, meta_path = _index_paths(lib)
        if idx_path.exists() and meta_path.exists() and not force_rebuild:
            existing = load_index(lib)
            n        = existing[0].ntotal if existing else 0
            print(f"    Already indexed ({n} vectors). Skipping.")
            report[lib] = n
            continue

        # Extract
        t0      = time.perf_counter()
        records = extract_library(lib)
        if not records:
            print("    No records extracted. Skipping.")
            continue
        print(f"    Extraction: {len(records)} functions in {time.perf_counter()-t0:.1f}s")

        # Embed — uses all GPUs automatically
        t0         = time.perf_counter()
        embeddings = _embed(model, [r.search_text for r in records])
        print(f"    Embedding done in {time.perf_counter()-t0:.1f}s")

        # Build FAISS on CPU, move to GPU 0 for in-session search
        t0    = time.perf_counter()
        index = _select_index(embeddings)
        index = _to_gpu(index)
        print(f"    Indexing done in {time.perf_counter()-t0:.1f}s")

        # _save converts to CPU internally before writing to disk
        _save(index, records, lib)
        report[lib] = len(records)

    elapsed     = time.perf_counter() - total_start
    total_funcs = sum(report.values())
    print("=" * 60)
    print(f"Done! {len(report)} libraries | {total_funcs:,} functions | {elapsed:.1f}s total")
    return report


def list_indexes() -> list[str]:
    """Return names of all libraries that have a built index."""
    if not INDEX_DIR.exists():
        return []
    return [p.stem for p in sorted(INDEX_DIR.glob("*.index"))]


def delete_index(library: str) -> bool:
    """Delete index files for a given library. Returns True if files existed."""
    idx_path, meta_path = _index_paths(library)
    deleted = False
    for p in (idx_path, meta_path):
        if p.exists():
            p.unlink()
            deleted = True
    if deleted:
        print(f"Deleted index for '{library}'")
    return deleted


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build function search indexes.")
    parser.add_argument("--libs",    nargs="+", default=DEFAULT_LIBRARIES)
    parser.add_argument("--rebuild", action="store_true", help="Force rebuild existing indexes")
    parser.add_argument("--list",    action="store_true", help="List existing indexes and exit")
    parser.add_argument("--delete",  nargs="+", metavar="LIB", help="Delete indexes for given libs")
    args = parser.parse_args()

    if args.list:
        libs = list_indexes()
        print(f"\nIndexed libraries ({len(libs)}):")
        for lib in libs:
            result = load_index(lib)
            n = result[0].ntotal if result else 0
            print(f"  {lib:<25}  {n:>6} functions")
    elif args.delete:
        for lib in args.delete:
            delete_index(lib)
    else:
        build_index(libraries=args.libs, force_rebuild=args.rebuild)