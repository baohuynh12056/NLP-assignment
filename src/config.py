from pathlib import Path

# ---------- Embedding model ----------
# Paper: https://arxiv.org/pdf/2506.05176
EMBED_MODEL = "Qwen/Qwen3-Embedding-4B"  # 4B fits T4 16GB on Kaggle

# Qwen3 requires this prefix on queries only — omitting it hurts quality
QUERY_INSTRUCTION = (
    "Instruct: Given a search query, retrieve relevant Python function "
    "documentation that matches the intent.\nQuery: "
)

# Cross-encoder reranker — re-scores FAISS candidates with full attention
# More accurate than bi-encoder but slower; only runs on small candidate set
RERANKER_MODEL = "BAAI/bge-reranker-large"

# ---------- Paths ----------
INDEX_DIR = Path("indexes")
LOG_DIR = Path("logs")

# ---------- Indexing ----------
BATCH_SIZE = 16
MAX_MODULES = 200  # max submodules to walk per library
DOCSTRING_LIMIT = 1024  # max chars of docstring fed into search_text

# ---------- Search ----------
TOP_K = 5
SCORE_THRESHOLD = 0.20  # discard FAISS results below this cosine similarity
RERANK_FETCH_K = 20  # candidates fetched from FAISS before reranking

# ---------- Libraries to index by default (most-used in common) ----------
# Reference: https://flexiple.com/python/python-libraries
DEFAULT_LIBRARIES = [
    # Standard library
    "os",
    "sys",
    "re",
    "math",
    "random",
    "datetime",
    "pathlib",
    "functools",
    "itertools",
    "collections",
    "statistics",
    "json",
    "csv",
    "io",
    "copy",
    "typing",
    "abc",
    "contextlib",
    "dataclasses",
    "enum",
    "warnings",
    "logging",
    "time",
    "threading",
    "multiprocessing",
    "subprocess",
    "hashlib",
    "uuid",
    "string",
    "struct",
    "pickle",
    "shutil",
    "glob",
    "tempfile",
    # Data manipulation
    "pandas",
    "numpy",
    "polars",
    "pyarrow",
    "xarray",
    # Data visualization
    "matplotlib",
    "matplotlib.pyplot",
    "seaborn",
    "plotly",
    "plotly.express",
    "bokeh",
    "altair",
    # Machine learning
    "sklearn",
    "sklearn.linear_model",
    "sklearn.ensemble",
    "sklearn.tree",
    "sklearn.svm",
    "sklearn.neighbors",
    "sklearn.cluster",
    "sklearn.decomposition",
    "sklearn.preprocessing",
    "sklearn.pipeline",
    "sklearn.model_selection",
    "sklearn.metrics",
    "sklearn.feature_extraction",
    "sklearn.feature_selection",
    "sklearn.impute",
    "sklearn.neural_network",
    "sklearn.inspection",
    "xgboost",
    "lightgbm",
    "catboost",
    "imbalanced-learn",
    # Deep learning
    "torch",
    "torch.nn",
    "torch.nn.functional",
    "torch.optim",
    "torch.utils.data",
    "torchvision",
    "torchvision.transforms",
    "torchvision.models",
    "torchmetrics",
    "tensorflow",
    "keras",
    "keras.layers",
    "keras.models",
    # NLP
    "transformers",
    "datasets",
    "tokenizers",
    "sentence_transformers",
    "spacy",
    "nltk",
    "gensim",
    "gensim.models",
    "langchain",
    "langchain.chains",
    "langchain.prompts",
    "langchain.agents",
    "openai",
    "anthropic",
    "tiktoken",
    # Computer vision
    "PIL",
    "PIL.Image",
    "PIL.ImageOps",
    "PIL.ImageFilter",
    "skimage",
    "skimage.io",
    "skimage.transform",
    "skimage.filters",
    # Scientific computing
    "scipy",
    "scipy.stats",
    "scipy.optimize",
    "scipy.linalg",
    "scipy.signal",
    "scipy.sparse",
    "scipy.interpolate",
    "scipy.integrate",
    "sympy",
    "networkx",
    # Data storage & I/O
    "sqlite3",
    "psycopg2",
    "pymongo",
    "redis",
    "h5py",
    "openpyxl",
    "xlrd",
    "fastavro",
    "boto3",
    "fsspec",
    # MLOps
    "ray",
    "ray.tune",
    "mlflow",
    "wandb",
    # Data validation
    "great_expectations",
    "ydata_profiling",
    # Web & API
    "httpx",
    "aiohttp",
    # Utilities
    "tqdm",
    "joblib",
    "dask",
    "dask.dataframe",
    "numba",
    "more_itertools",
    "toolz",
    "attrs",
    "click",
    "pyyaml",
    "toml",
    "python_dotenv",
]
