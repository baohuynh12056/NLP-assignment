import argparse
import statistics
import time
from typing import Callable

from config import SCORE_THRESHOLD, TOP_K
from search import FunctionSearcher, SearchResult, display_results

# ── ANSI color helpers ────────────────────────────────────────────────────────
_G = "\033[92m"
_R = "\033[91m"
_Y = "\033[93m"
_B = "\033[94m"
_W = "\033[97m"
_DIM = "\033[2m"
_RST = "\033[0m"


def _ok(s="OK"):
    return f"{_G}✓ {s}{_RST}"


def _fail(s="FAIL"):
    return f"{_R}✗ {s}{_RST}"


def _hdr(title: str) -> None:
    bar = "=" * 68
    print(f"\n{_B}{bar}{_RST}")
    print(f"{_B}  {title}{_RST}")
    print(f"{_B}{bar}{_RST}")


# ── Labeled test set ──────────────────────────────────────────────────────────
# (query, expected_substring_in_full_name, hint_library)
TEST_SET: list[tuple[str, str, str]] = [
    # pandas
    ("group rows by a column and compute the mean", "groupby", "pandas"),
    ("merge two dataframes on a common key", "merge", "pandas"),
    ("drop rows that contain missing values", "dropna", "pandas"),
    ("read a CSV file into a dataframe", "read_csv", "pandas"),
    ("reshape dataframe from wide to long format", "melt", "pandas"),
    ("sort dataframe by column values", "sort_values", "pandas"),
    ("pivot table aggregation", "pivot", "pandas"),
    ("apply a function to each row or column", "apply", "pandas"),
    ("rename columns in a dataframe", "rename", "pandas"),
    ("fill missing values with a constant or method", "fillna", "pandas"),
    # numpy
    ("compute the dot product of two arrays", "dot", "numpy"),
    ("find unique elements in an array", "unique", "numpy"),
    ("reshape an array to a new shape", "reshape", "numpy"),
    ("stack arrays vertically", "vstack", "numpy"),
    ("generate evenly spaced numbers over an interval", "linspace", "numpy"),
    ("compute element-wise exponential", "exp", "numpy"),
    ("sort an array along an axis", "sort", "numpy"),
    ("compute the inverse of a matrix", "linalg", "numpy"),
    # sklearn
    ("split dataset into train and test sets", "train_test_split", "sklearn"),
    ("standardize features by removing the mean", "StandardScaler", "sklearn"),
    ("train a random forest classifier", "RandomForest", "sklearn"),
    ("compute accuracy of a classification model", "accuracy_score", "sklearn"),
    ("perform k-means clustering", "KMeans", "sklearn"),
    ("reduce dimensionality using PCA", "PCA", "sklearn"),
    ("encode categorical labels as integers", "LabelEncoder", "sklearn"),
    ("cross-validate a model with k folds", "cross_val", "sklearn"),
    ("find best hyperparameters with grid search", "GridSearchCV", "sklearn"),
    ("impute missing values", "SimpleImputer", "sklearn"),
    # stdlib
    ("flatten a nested list", "chain", "itertools"),
    ("count occurrences of each element", "Counter", "collections"),
    ("cache results of an expensive function", "lru_cache", "functools"),
    ("list all files in a directory recursively", "rglob", "pathlib"),
    ("find all matches of a regex pattern in a string", "findall", "re"),
    ("compute mean of a list of numbers", "mean", "statistics"),
    # scipy
    ("perform a t-test between two groups", "ttest", "scipy"),
    ("compute pearson correlation coefficient", "pearsonr", "scipy"),
    ("solve a linear system of equations", "solve", "scipy"),
    # other
    ("track and log an ML experiment", "log", "mlflow"),
    ("encode text into embeddings", "encode", "sentence_transformers"),
    ("show a progress bar for a loop", "tqdm", "tqdm"),
    ("validate data with a schema", "BaseModel", "pydantic"),
]


# ── Metric helpers ────────────────────────────────────────────────────────────


def _hit(results: list[SearchResult], expected: str) -> bool:
    """True if any result's full_name contains expected (case-insensitive)."""
    exp = expected.lower()
    return any(exp in r.full_name.lower() for r in results)


def _reciprocal_rank(results: list[SearchResult], expected: str) -> float:
    """1/rank of first hit, 0.0 if not found."""
    exp = expected.lower()
    for r in results:
        if exp in r.full_name.lower():
            return 1.0 / r.rank
    return 0.0


# ── Experiment 1: Sanity Check ────────────────────────────────────────────────


def exp_sanity(searcher: FunctionSearcher) -> None:
    """Pass/fail on a small set of obvious query→function pairs."""
    _hdr("Experiment 1 — Sanity Check")

    SANITY: list[tuple[str, str]] = [
        ("read a CSV file into a dataframe", "read_csv"),
        ("merge two dataframes on a key", "merge"),
        ("split data into train and test sets", "train_test_split"),
        ("compute the mean of an array", "mean"),
        ("count occurrences of each element", "Counter"),
        ("reduce dimensionality with PCA", "PCA"),
        ("find all regex matches in a string", "findall"),
        ("generate evenly spaced numbers", "linspace"),
        ("drop rows with missing values", "dropna"),
        ("sort dataframe by column", "sort_values"),
    ]

    passed = 0
    for query, expected in SANITY:
        results = searcher.search(query, top_k=5)
        ok = _hit(results, expected)
        passed += int(ok)
        top = results[0].full_name if results else "(no results)"
        print(f"  {_ok() if ok else _fail()}  {query:<50}  →  {top}")

    rate = passed / len(SANITY) * 100
    col = _G if rate >= 80 else (_Y if rate >= 60 else _R)
    print(f"\n  Result: {col}{passed}/{len(SANITY)}  ({rate:.0f}%){_RST}")


# ── Experiment 2: MRR & Hit@K ─────────────────────────────────────────────────


def exp_mrr(searcher: FunctionSearcher) -> None:
    """MRR and Hit@{1,3,5,10} over the full labeled TEST_SET."""
    _hdr("Experiment 2 — MRR & Hit@K")

    Ks = [1, 3, 5, 10]
    rrs: list[float] = []
    hits: dict[int, int] = {k: 0 for k in Ks}
    missed: list[tuple[str, str]] = []

    for query, expected, _ in TEST_SET:
        results = searcher.search(query, top_k=max(Ks))
        rr = _reciprocal_rank(results, expected)
        rrs.append(rr)
        for k in Ks:
            if _hit(results[:k], expected):
                hits[k] += 1
        if rr == 0:
            top = results[0].full_name if results else "(no results)"
            missed.append((query, top))

    n = len(TEST_SET)
    mrr = statistics.mean(rrs)
    print(f"  Queries evaluated : {n}")
    print(f"  MRR               : {mrr:.4f}")
    for k in Ks:
        pct = hits[k] / n * 100
        bar = "█" * int(pct / 2)
        print(f"  Hit@{k:<3}           : {hits[k]:>3}/{n}  ({pct:5.1f}%)  {bar}")

    if missed:
        print(f"\n  {_Y}Missed ({len(missed)}):{_RST}")
        for q, top in missed:
            print(f"    {_DIM}{q:<55}{_RST}  →  {top}")


# ── Experiment 3: MMR vs Plain ────────────────────────────────────────────────


def exp_mmr_vs_plain(searcher: FunctionSearcher) -> None:
    """Compare MMR re-ranking vs plain cosine ranking on diversity and MRR."""
    _hdr("Experiment 3 — MMR vs Plain Ranking")

    def _ngrams(s: str, n: int = 3) -> set[str]:
        s = s.lower()
        return {s[i : i + n] for i in range(len(s) - n + 1)}

    def _jaccard(a: str, b: str) -> float:
        sa, sb = _ngrams(a), _ngrams(b)
        union = len(sa | sb)
        return len(sa & sb) / union if union else 0.0

    def _diversity(results: list[SearchResult]) -> float:
        names = [r.full_name for r in results]
        if len(names) < 2:
            return 0.0
        pairs = [(a, b) for i, a in enumerate(names) for b in names[i + 1 :]]
        return statistics.mean(1 - _jaccard(a, b) for a, b in pairs)

    SAMPLE = [q for q, _, _ in TEST_SET[:15]]
    mmr_divs, plain_divs, mmr_rrs, plain_rrs = [], [], [], []

    for query in SAMPLE:
        expected = next(e for q, e, _ in TEST_SET if q == query)
        mmr_res = searcher.search(query, top_k=5, use_mmr=True)
        plain_res = searcher.search(query, top_k=5, use_mmr=False)
        mmr_divs.append(_diversity(mmr_res))
        plain_divs.append(_diversity(plain_res))
        mmr_rrs.append(_reciprocal_rank(mmr_res, expected))
        plain_rrs.append(_reciprocal_rank(plain_res, expected))

    avg = lambda xs: statistics.mean(xs) if xs else 0.0
    print(f"  {'Metric':<30}  {'Plain':>10}  {'MMR':>10}  {'Δ':>10}")
    print(f"  {'-' * 30}  {'-' * 10}  {'-' * 10}  {'-' * 10}")

    d_plain, d_mmr = avg(plain_divs), avg(mmr_divs)
    r_plain, r_mmr = avg(plain_rrs), avg(mmr_rrs)
    delta_d = (d_mmr - d_plain) / max(d_plain, 1e-9) * 100
    delta_r = (r_mmr - r_plain) / max(r_plain, 1e-9) * 100

    print(
        f"  {'Diversity (↑)':<30}  {d_plain:>10.4f}  {d_mmr:>10.4f}  "
        f"{_G if delta_d > 0 else _R}{delta_d:>+9.1f}%{_RST}"
    )
    print(
        f"  {'MRR (↑)':<30}  {r_plain:>10.4f}  {r_mmr:>10.4f}  "
        f"{_G if delta_r >= -5 else _Y}{delta_r:>+9.1f}%{_RST}"
    )


# ── Experiment 4: Threshold Sweep ─────────────────────────────────────────────


def exp_threshold(searcher: FunctionSearcher) -> None:
    """Sweep score_threshold to find optimal cutoff via F1(Hit@5, coverage)."""
    _hdr("Experiment 4 — Score Threshold Sweep")

    thresholds = [
        round(t, 2) for t in [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50, 0.60]
    ]
    print(f"  {'Threshold':>10}  {'Hit@5':>8}  {'Coverage':>10}  {'Avg Results':>12}")
    print(f"  {'-' * 10}  {'-' * 8}  {'-' * 10}  {'-' * 12}")

    best_f1, best_t = -1.0, 0.0
    for t in thresholds:
        hits, total_res, covered = 0, 0, 0
        for query, expected, _ in TEST_SET:
            res = searcher.search(query, top_k=5, score_threshold=t)
            total_res += len(res)
            covered += int(bool(res))
            if _hit(res[:5], expected):
                hits += 1

        n = len(TEST_SET)
        hit5 = hits / n
        coverage = covered / n
        avg_res = total_res / n
        f1 = 2 * hit5 * coverage / max(hit5 + coverage, 1e-9)

        marker = f"  {_Y}← best F1{_RST}" if f1 > best_f1 else ""
        if f1 > best_f1:
            best_f1, best_t = f1, t

        col = _G if hit5 >= 0.7 else (_Y if hit5 >= 0.5 else _R)
        print(
            f"  {t:>10.2f}  {col}{hit5:>7.1%}{_RST}  {coverage:>9.1%}  {avg_res:>11.1f}{marker}"
        )

    print(f"\n  Recommended : {_G}{best_t:.2f}{_RST}  (best F1={best_f1:.3f})")
    print(f"  Current     : {SCORE_THRESHOLD:.2f}  (config.SCORE_THRESHOLD)")


# ── Experiment 5: Query Expansion ─────────────────────────────────────────────


def exp_query_expansion(searcher: FunctionSearcher) -> None:
    """Measure Hit@5 improvement from prepending 'python function' to short queries."""
    _hdr("Experiment 5 — Query Expansion Impact")

    from search import _expand_query

    SHORT_QUERIES: list[tuple[str, str]] = [
        ("merge", "merge"),
        ("sort", "sort"),
        ("unique", "unique"),
        ("reshape", "reshape"),
        ("mean", "mean"),
        ("encode", "encode"),
        ("split dataset", "train_test_split"),
        ("read csv", "read_csv"),
        ("drop na", "dropna"),
        ("scale features", "StandardScaler"),
    ]

    print(f"  {'Query':<22}  {'Expanded':<38}  {'Raw':>5}  {'Exp':>5}")
    print(f"  {'-' * 22}  {'-' * 38}  {'-' * 5}  {'-' * 5}")

    raw_hits = exp_hits = 0
    for query, expected in SHORT_QUERIES:
        expanded = _expand_query(query)
        results_raw = searcher.search(query, top_k=5)
        results_exp = searcher.search(expanded, top_k=5)
        h_raw = _hit(results_raw, expected)
        h_exp = _hit(results_exp, expected)
        raw_hits += int(h_raw)
        exp_hits += int(h_exp)
        print(
            f"  {query:<22}  {expanded:<38}  "
            f"{_ok('✓') if h_raw else _fail('✗'):>5}  "
            f"{_ok('✓') if h_exp else _fail('✗'):>5}"
        )

    n = len(SHORT_QUERIES)
    print(f"\n  Raw Hit@5 : {raw_hits}/{n}  ({raw_hits / n:.0%})")
    print(f"  Exp Hit@5 : {exp_hits}/{n}  ({exp_hits / n:.0%})")
    delta = exp_hits - raw_hits
    print(
        f"  Δ         : {_G if delta > 0 else (_Y if delta == 0 else _R)}{delta:+d} queries{_RST}"
    )


# ── Experiment 6: Cross-Library ───────────────────────────────────────────────


def exp_cross_library(searcher: FunctionSearcher) -> None:
    """Check whether results for generic queries span multiple libraries."""
    _hdr("Experiment 6 — Cross-Library Result Distribution")

    CROSS: list[tuple[str, list[str]]] = [
        ("compute mean of numbers", ["numpy", "statistics", "pandas"]),
        ("sort a collection of items", ["numpy", "pandas", "functools"]),
        ("read data from a file", ["pandas", "pathlib", "os", "csv", "io"]),
        ("apply a function to a sequence", ["functools", "pandas", "numpy"]),
        ("generate random numbers", ["numpy", "random"]),
        ("find minimum value", ["numpy", "pandas", "statistics"]),
    ]

    for query, expected_libs in CROSS:
        results = searcher.search(query, top_k=8)
        unique_libs = list(dict.fromkeys(r.library for r in results))
        covered = [l for l in expected_libs if l in unique_libs]
        pct = len(covered) / len(expected_libs) * 100
        col = _G if pct >= 60 else (_Y if pct >= 40 else _R)

        print(f"\n  Query : {_W}{query}{_RST}")
        print(f"  Found : {', '.join(unique_libs[:6])}")
        print(f"  Coverage: {col}{pct:.0f}%{_RST}  ({covered} / {expected_libs})")


# ── Experiment 7: Latency ─────────────────────────────────────────────────────


def exp_latency(searcher: FunctionSearcher, n: int = 30) -> None:
    """Wall-clock latency benchmark including embed + FAISS + rerank."""
    _hdr(f"Experiment 7 — Latency Benchmark  (n={n})")

    queries = ([q for q, _, _ in TEST_SET] * (n // len(TEST_SET) + 1))[:n]
    latencies: list[float] = []

    for query in queries:
        t0 = time.perf_counter()
        searcher.search(query, top_k=TOP_K)
        latencies.append(time.perf_counter() - t0)

    latencies.sort()
    mean_ms = statistics.mean(latencies) * 1000
    med_ms = statistics.median(latencies) * 1000
    p95_ms = latencies[int(0.95 * n)] * 1000
    p99_ms = latencies[min(int(0.99 * n), n - 1)] * 1000
    qps = n / sum(latencies)

    print(f"  Queries      : {n}")
    print(f"  Mean latency : {mean_ms:>8.1f} ms")
    print(f"  Median       : {med_ms:>8.1f} ms")
    print(f"  p95          : {p95_ms:>8.1f} ms")
    print(f"  p99          : {p99_ms:>8.1f} ms")
    print(f"  Throughput   : {qps:>8.1f} queries/sec")

    col = _G if mean_ms < 200 else (_Y if mean_ms < 500 else _R)
    print(
        f"\n  Verdict: {col}{'Fast' if mean_ms < 200 else 'Acceptable' if mean_ms < 500 else 'Slow'}{_RST} "
        f"(mean {mean_ms:.0f} ms/query)"
    )


# ── Experiment 8: Hard Negatives ─────────────────────────────────────────────


def exp_hard_negatives(searcher: FunctionSearcher) -> None:
    """Verify correct answer ranks above a plausible-but-wrong alternative."""
    _hdr("Experiment 8 — Hard Negatives")

    HARD: list[tuple[str, str, str]] = [
        ("sort dataframe rows by column value", "sort_values", "argsort"),
        ("join two dataframes on a shared column", "merge", "concat"),
        ("reduce features using principal component analysis", "PCA", "TruncatedSVD"),
        ("load a comma-separated file into a dataframe", "read_csv", "read_table"),
        ("count frequency of each element in a python list", "Counter", "value_counts"),
        ("memoize a function with a maximum cache size", "lru_cache", "cache"),
        ("split data once into train and test portions", "train_test_split", "KFold"),
        ("remove rows where any value is missing", "dropna", "fillna"),
    ]

    passed = 0
    print(f"  {'Query':<45}  {'Correct':>12}  {'Hard neg':>12}  Result")
    print(f"  {'-' * 45}  {'-' * 12}  {'-' * 12}  {'-' * 6}")

    for query, correct, hard_neg in HARD:
        results = searcher.search(query, top_k=10)
        names = [r.full_name.lower() for r in results]
        correct_rank = next(
            (i + 1 for i, n in enumerate(names) if correct.lower() in n), None
        )
        hardneg_rank = next(
            (i + 1 for i, n in enumerate(names) if hard_neg.lower() in n), None
        )

        ok = correct_rank is not None and (
            hardneg_rank is None or correct_rank < hardneg_rank
        )
        passed += int(ok)

        cr_str = f"rank {correct_rank}" if correct_rank else "not found"
        hn_str = f"rank {hardneg_rank}" if hardneg_rank else "not found"
        print(f"  {query:<45}  {cr_str:>12}  {hn_str:>12}  {_ok() if ok else _fail()}")

    n = len(HARD)
    pct = passed / n * 100
    col = _G if pct >= 75 else (_Y if pct >= 50 else _R)
    print(f"\n  Result: {col}{passed}/{n}  ({pct:.0f}%){_RST}")


# ── Experiment 9: Interactive Demo ────────────────────────────────────────────


def exp_interactive(searcher: FunctionSearcher) -> None:
    """Live search REPL. Flags: --lib <lib>  --k <n>  --no-mmr  |  q to quit."""
    _hdr("Experiment 9 — Interactive Demo")
    print("  Type a query. Flags: --lib <lib>  --k <n>  --no-mmr  |  q to quit\n")

    while True:
        try:
            raw = input(f"  {_B}search>{_RST} ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if raw.lower() in ("q", "quit", "exit"):
            break
        if not raw:
            continue

        tokens = raw.split()
        libs = None
        top_k = TOP_K
        use_mmr = True
        query_tokens: list[str] = []

        i = 0
        while i < len(tokens):
            if tokens[i] == "--lib" and i + 1 < len(tokens):
                libs = [tokens[i + 1]]
                i += 2
            elif tokens[i] == "--k" and i + 1 < len(tokens):
                top_k = int(tokens[i + 1])
                i += 2
            elif tokens[i] == "--no-mmr":
                use_mmr = False
                i += 1
            else:
                query_tokens.append(tokens[i])
                i += 1

        results = searcher.search(
            " ".join(query_tokens), top_k=top_k, libraries=libs, use_mmr=use_mmr
        )
        print()
        display_results(results)
        print()


# ── Registry & runner ─────────────────────────────────────────────────────────

EXPERIMENTS: dict[str, tuple[Callable, str]] = {
    "sanity": (exp_sanity, "Pass/fail on obvious query→function pairs"),
    "mrr": (exp_mrr, "MRR & Hit@K on full test set"),
    "mmr": (exp_mmr_vs_plain, "MMR vs plain: diversity vs relevance"),
    "threshold": (exp_threshold, "Score threshold sweep"),
    "expansion": (exp_query_expansion, "Query expansion impact on short queries"),
    "cross": (exp_cross_library, "Cross-library result distribution"),
    "latency": (exp_latency, "Latency benchmark"),
    "hard": (exp_hard_negatives, "Hard negatives: correct vs plausible-wrong"),
    "demo": (exp_interactive, "Interactive live search REPL"),
}


def run_all(searcher: FunctionSearcher, exclude: list[str] = ("demo",)) -> None:
    for name, (fn, _) in EXPERIMENTS.items():
        if name in exclude:
            continue
        fn(searcher)
    print(f"\n{_G}All experiments complete.{_RST}\n")


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate the function reverse search system."
    )
    parser.add_argument(
        "--exp",
        choices=list(EXPERIMENTS.keys()) + ["all"],
        default="all",
        help="Which experiment to run (default: all)",
    )
    parser.add_argument(
        "--libs", nargs="+", default=None, help="Restrict to these libraries"
    )
    parser.add_argument(
        "--n", type=int, default=30, help="Queries for latency benchmark"
    )
    parser.add_argument("--no-mmr", action="store_true", help="Disable MMR globally")
    args = parser.parse_args()

    print(f"\n{_B}Initializing searcher...{_RST}")
    searcher = FunctionSearcher(libraries=args.libs, use_mmr=not args.no_mmr)
    print(f"  Libraries : {searcher.loaded_libraries()}")
    print(f"  Functions : {searcher.total_functions():,}")

    if args.exp == "all":
        run_all(searcher)
    elif args.exp == "latency":
        exp_latency(searcher, n=args.n)
    else:
        EXPERIMENTS[args.exp][0](searcher)
