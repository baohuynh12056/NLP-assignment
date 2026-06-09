import importlib
import inspect
import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from bs4 import BeautifulSoup
from docx import Document
from pypdf import PdfReader


DEFAULT_LIBRARY_MODULES = {
    "pandas": ["pandas"],
    "numpy": ["numpy"],
    "sklearn": ["sklearn", "sklearn.model_selection", "sklearn.preprocessing", "sklearn.metrics"],
    "torch": ["torch", "torch.nn", "torch.optim"],
    "tensorflow": ["tensorflow", "tensorflow.keras", "tensorflow.keras.layers"],
    "matplotlib": ["matplotlib", "matplotlib.pyplot"],
    "scipy": ["scipy", "scipy.optimize", "scipy.stats", "scipy.linalg"],
    "seaborn": ["seaborn"],
    "requests": ["requests"],
    "fastapi": ["fastapi"],
}

DEFAULT_CLASS_TARGETS = {
    "pandas": ["pandas.DataFrame", "pandas.Series"],
    "sklearn": [
        "sklearn.preprocessing.StandardScaler",
    ],
    "fastapi": ["fastapi.FastAPI"],
}


def load_jsonl(path: str) -> List[Dict[str, Any]]:
    records = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def write_jsonl(records: Iterable[Dict[str, Any]], path: str) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def introspect_library(
    library_name: str,
    max_members: int = 1200,
    include_classes: bool = True,
) -> List[Dict[str, Any]]:
    """
    Extracts public function/class docs from an installed Python library.
    This is a lightweight bootstrap parser; curated docs can still be loaded via JSONL.
    """
    records = []
    for module_name in DEFAULT_LIBRARY_MODULES.get(library_name, [library_name]):
        if len(records) >= max_members:
            break
        module_limit = max_members - len(records)
        records.extend(introspect_module(module_name, library_name, module_limit, include_classes))
    for class_path in DEFAULT_CLASS_TARGETS.get(library_name, []):
        if len(records) >= max_members:
            break
        records.extend(
            introspect_class_methods(
                class_path,
                library_name,
                max_members=max_members - len(records),
            )
        )
    return records


def introspect_module(
    module_name: str,
    library_name: str,
    max_members: int = 500,
    include_classes: bool = True,
) -> List[Dict[str, Any]]:
    module = importlib.import_module(module_name)
    root_module = importlib.import_module(library_name)
    version = getattr(root_module, "__version__", None)
    records = []

    for name, obj in inspect.getmembers(module):
        if name.startswith("_"):
            continue
        if not is_documented_public_api(obj):
            continue
        if inspect.isclass(obj) and not include_classes:
            continue

        docstring = inspect.getdoc(obj)
        if not docstring:
            continue

        full_name = f"{module_name}.{name}"
        records.append({
            "library_name": library_name,
            "module_name": getattr(obj, "__module__", library_name),
            "func_name": name,
            "full_name": full_name,
            "signature": safe_signature(obj),
            "docstring": docstring,
            "parameters": {},
            "version": version,
        })

        if len(records) >= max_members:
            break

    return records


def introspect_class_methods(
    class_path: str,
    library_name: str,
    max_members: int = 200,
) -> List[Dict[str, Any]]:
    module_name, class_name = class_path.rsplit(".", 1)
    module = importlib.import_module(module_name)
    root_module = importlib.import_module(library_name)
    version = getattr(root_module, "__version__", None)
    cls = getattr(module, class_name)
    records = []

    for method_name, method in inspect.getmembers(cls):
        if method_name.startswith("_"):
            continue
        if not (
            inspect.isfunction(method)
            or inspect.ismethod(method)
            or isinstance(method, property)
            or inspect.isbuiltin(method)
        ):
            continue

        docstring = inspect.getdoc(method)
        if not docstring:
            continue

        full_name = f"{class_path}.{method_name}"
        records.append({
            "library_name": library_name,
            "module_name": class_path,
            "func_name": method_name,
            "full_name": full_name,
            "signature": safe_signature(method),
            "docstring": docstring,
            "parameters": {},
            "version": version,
        })
        if len(records) >= max_members:
            break

    return records


def introspect_libraries(
    library_names: Iterable[str],
    max_members_per_library: int = 500,
) -> List[Dict[str, Any]]:
    all_records = []
    for library_name in library_names:
        try:
            all_records.extend(introspect_library(library_name, max_members_per_library))
        except Exception as exc:
            print(f"[parsers] Skipping {library_name}: {type(exc).__name__}: {exc}")
    return all_records


def safe_signature(obj: Any) -> Optional[str]:
    try:
        return str(inspect.signature(obj))
    except (TypeError, ValueError):
        return None


def is_documented_public_api(obj: Any) -> bool:
    return (
        inspect.isfunction(obj)
        or inspect.isclass(obj)
        or inspect.isbuiltin(obj)
        or (callable(obj) and inspect.getdoc(obj) is not None)
    )


def parse_document_file(path: str, library_name: str, source_url: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    Parses PDF, HTML, DOCX, Markdown, and text docs into function-like records.
    The parser is deliberately heuristic; LLM augmentation can refine the output later.
    """

    file_path = Path(path)
    suffix = file_path.suffix.lower()
    if suffix == ".pdf":
        sections = parse_pdf(file_path)
    elif suffix in {".html", ".htm"}:
        sections = parse_html(file_path)
    elif suffix == ".docx":
        sections = parse_docx(file_path)
    elif suffix in {".md", ".rst", ".txt"}:
        sections = parse_text(file_path)
    else:
        raise ValueError(f"Unsupported document type: {file_path.suffix}")

    records = []
    for section in sections:
        records.extend(section_to_function_records(section, library_name, str(file_path), source_url))
    return records


def parse_docs_directory(root_dir: str, library_name: str) -> List[Dict[str, Any]]:
    supported = {".pdf", ".html", ".htm", ".docx", ".md", ".rst", ".txt"}
    records = []
    for path in Path(root_dir).rglob("*"):
        if path.is_file() and path.suffix.lower() in supported:
            try:
                records.extend(parse_document_file(str(path), library_name))
            except Exception as exc:
                print(f"[parsers] Failed {path}: {type(exc).__name__}: {exc}")
    return records


def parse_pdf(path: Path) -> List[Dict[str, Any]]:
    reader = PdfReader(str(path))
    sections = []
    for page_number, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        if text.strip():
            sections.extend(split_into_sections(text, title=f"{path.stem} page {page_number}"))
    return sections


def parse_html(path: Path) -> List[Dict[str, Any]]:
    soup = BeautifulSoup(path.read_text(encoding="utf-8", errors="ignore"), "lxml")
    for tag in soup(["script", "style", "nav", "footer"]):
        tag.decompose()

    sections = []
    current_title = path.stem
    current_lines = []
    for element in soup.find_all(["h1", "h2", "h3", "h4", "dt", "p", "pre", "code", "li"]):
        text = element.get_text(" ", strip=True)
        if not text:
            continue
        if element.name in {"h1", "h2", "h3", "h4", "dt"} and current_lines:
            sections.append({"title": current_title, "text": "\n".join(current_lines)})
            current_lines = []
            current_title = text
        else:
            current_lines.append(text)
    if current_lines:
        sections.append({"title": current_title, "text": "\n".join(current_lines)})
    return sections


def parse_docx(path: Path) -> List[Dict[str, Any]]:
    document = Document(str(path))
    sections = []
    current_title = path.stem
    current_lines = []
    for paragraph in document.paragraphs:
        text = paragraph.text.strip()
        if not text:
            continue
        if paragraph.style.name.startswith("Heading") and current_lines:
            sections.append({"title": current_title, "text": "\n".join(current_lines)})
            current_title = text
            current_lines = []
        else:
            current_lines.append(text)
    if current_lines:
        sections.append({"title": current_title, "text": "\n".join(current_lines)})
    return sections


def parse_text(path: Path) -> List[Dict[str, Any]]:
    text = path.read_text(encoding="utf-8", errors="ignore")
    return split_into_sections(text, title=path.stem)


def split_into_sections(text: str, title: str, max_chars: int = 3500) -> List[Dict[str, Any]]:
    blocks = [block.strip() for block in re.split(r"\n{2,}", text) if block.strip()]
    sections = []
    current = []
    current_len = 0
    for block in blocks:
        if current and current_len + len(block) > max_chars:
            sections.append({"title": title, "text": "\n\n".join(current)})
            current = []
            current_len = 0
        current.append(block)
        current_len += len(block)
    if current:
        sections.append({"title": title, "text": "\n\n".join(current)})
    return sections


def section_to_function_records(
    section: Dict[str, Any],
    library_name: str,
    source_path: str,
    source_url: Optional[str] = None,
) -> List[Dict[str, Any]]:
    text = section["text"]
    candidates = extract_function_candidates(section["title"], text, library_name)
    records = []
    for candidate in candidates:
        records.append({
            "library_name": library_name,
            "module_name": candidate.get("module_name"),
            "func_name": candidate["func_name"],
            "full_name": candidate["full_name"],
            "signature": candidate.get("signature"),
            "docstring": text,
            "parameters": extract_parameters(text),
            "source_url": source_url,
            "metadata": {
                "source_path": source_path,
                "section_title": section["title"],
                "parser": "heuristic_document_parser",
            },
        })
    return records


def extract_function_candidates(title: str, text: str, library_name: str) -> List[Dict[str, Any]]:
    combined = f"{title}\n{text[:600]}"
    patterns = [
        rf"({re.escape(library_name)}(?:\.[A-Za-z_][\w]*)+)\s*(\([^)]*\))?",
        r"\b([A-Za-z_][\w]*\.[A-Za-z_][\w]*(?:\.[A-Za-z_][\w]*)*)\s*(\([^)]*\))?",
        r"\b([A-Za-z_][\w]*)\s*(\([^)]{0,200}\))",
    ]
    seen = set()
    candidates = []
    for pattern in patterns:
        for match in re.finditer(pattern, combined):
            name = match.group(1).strip()
            signature = match.group(2)
            if name in seen or name.lower() in {"if", "for", "while", "return"}:
                continue
            seen.add(name)
            full_name = name if name.startswith(f"{library_name}.") else f"{library_name}.{name}"
            module_parts = full_name.split(".")
            candidates.append({
                "full_name": full_name,
                "module_name": ".".join(module_parts[:-1]),
                "func_name": module_parts[-1],
                "signature": signature,
            })
    if not candidates:
        fallback_name = slug_to_func_name(title)
        candidates.append({
            "full_name": f"{library_name}.{fallback_name}",
            "module_name": library_name,
            "func_name": fallback_name,
            "signature": None,
        })
    return candidates[:5]


def extract_parameters(text: str) -> Dict[str, Any]:
    params: Dict[str, Any] = {}
    for match in re.finditer(r"^\s*([A-Za-z_][\w]*)\s*:\s*([^\n]+)", text, flags=re.MULTILINE):
        params[match.group(1)] = {"type": match.group(2).strip()}
    return params


def slug_to_func_name(title: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_]+", "_", title.strip()).strip("_").lower()
    return cleaned or "unknown_function"
