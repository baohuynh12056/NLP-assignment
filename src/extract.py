import importlib
import inspect
import pkgutil
import re
from dataclasses import dataclass, field
from typing import Optional

from config import DOCSTRING_LIMIT, MAX_MODULES

# Map pip package names → actual Python import names
IMPORT_REMAP: dict[str, str] = {
    "imbalanced-learn": "imblearn",
    "scikit-learn": "sklearn",
    "scikit-image": "skimage",
    "pillow": "PIL",
    "pyyaml": "yaml",
    "python-dotenv": "dotenv",
    "opencv-python": "cv2",
}


# ── Data models ───────────────────────────────────────────────────────────────


@dataclass
class ParamRecord:
    name: str
    type_hint: str = ""
    description: str = ""


@dataclass
class FunctionRecord:
    library: str
    module: str
    name: str
    full_name: str
    kind: str
    signature: str
    docstring: str
    params: list[ParamRecord] = field(default_factory=list)
    search_text: str = field(init=False)

    def __post_init__(self):
        # Combine name + summary + param names + body for embedding
        summary = self.docstring.splitlines()[0] if self.docstring else ""
        param_names = " ".join(p.name for p in self.params)
        body = self.docstring[:DOCSTRING_LIMIT]
        self.search_text = f"{self.full_name} {summary} {param_names} {body}"


# ── Docstring parsers ─────────────────────────────────────────────────────────


def parse_numpy_style(lines: list[str]) -> dict[str, ParamRecord]:
    """Parse NumPy-style docstrings: 'param : type\\n    description'."""
    params = {}
    in_params = False
    current = None

    open_tags = {"parameters", "params", "arguments", "args"}
    close_tags = {
        "returns",
        "return",
        "yields",
        "yield",
        "raises",
        "raise",
        "notes",
        "note",
        "examples",
        "example",
        "references",
        "see also",
        "attributes",
    }
    separator_re = re.compile(r"^[-=]{3,}$")

    for line in lines:
        stripped = line.strip()
        lower = stripped.lower()

        if separator_re.match(stripped):
            continue
        if lower in open_tags:
            in_params = True
            current = None
            continue
        if lower in close_tags:
            in_params = False
            current = None
            continue
        if not in_params:
            continue

        if re.match(r"^\S+.*", stripped) and " : " in stripped:
            parts = stripped.split(" : ", 1)
            pname = parts[0].strip()
            th = parts[1].strip() if len(parts) > 1 else ""
            current = ParamRecord(name=pname, type_hint=th)
            params[pname] = current
        elif current and stripped:
            current.description = (current.description + " " + stripped).strip()

    return params


def parse_google_style(lines: list[str]) -> dict[str, ParamRecord]:
    """Parse Google-style docstrings: '    param (type): description'."""
    params = {}
    in_params = False
    current = None

    open_tags = {"args:", "arguments:", "params:", "parameters:"}
    close_tags = {
        "returns:",
        "return:",
        "yields:",
        "yield:",
        "raises:",
        "raise:",
        "note:",
        "notes:",
        "example:",
        "examples:",
        "attributes:",
    }
    param_re = re.compile(r"^\s{4,}(\w+)\s*(?:\(([^)]*)\))?\s*:\s*(.*)")

    for line in lines:
        stripped = line.strip().lower()

        if stripped in open_tags:
            in_params = True
            current = None
            continue
        if stripped in close_tags:
            in_params = False
            current = None
            continue
        if not in_params:
            continue

        m = param_re.match(line)
        if m:
            pname, th, desc = m.group(1), m.group(2) or "", m.group(3)
            current = ParamRecord(name=pname, type_hint=th, description=desc.strip())
            params[pname] = current
        elif current and line.startswith("        "):
            current.description = (current.description + " " + line.strip()).strip()

    return params


def parse_rest_style(lines: list[str]) -> dict[str, ParamRecord]:
    """Parse reST-style docstrings: ':param name: description'."""
    params = {}
    param_re = re.compile(r":param\s+(\w+)\s*:\s*(.*)")
    type_re = re.compile(r":type\s+(\w+)\s*:\s*(.*)")

    for line in lines:
        m = param_re.match(line.strip())
        if m:
            pname, desc = m.group(1), m.group(2)
            params.setdefault(pname, ParamRecord(name=pname)).description = desc.strip()
            continue
        m = type_re.match(line.strip())
        if m:
            pname, th = m.group(1), m.group(2)
            if pname in params:
                params[pname].type_hint = th.strip()

    return params


def parse_params(docstring: str) -> list[ParamRecord]:
    """Auto-detect docstring style and merge results from all matching parsers."""
    if not docstring:
        return []

    lines = docstring.splitlines()
    has_numpy = any(re.match(r"^[-=]{3,}$", l.strip()) for l in lines)
    has_google = any(l.strip().lower() in ("args:", "arguments:") for l in lines)
    has_rest = any(":param " in l for l in lines)

    parsers = []
    if has_numpy:
        parsers.append(parse_numpy_style)
    if has_google:
        parsers.append(parse_google_style)
    if has_rest:
        parsers.append(parse_rest_style)
    if not parsers:
        parsers = [parse_numpy_style, parse_google_style, parse_rest_style]

    merged: dict[str, ParamRecord] = {}
    for parser in parsers:
        for k, v in parser(lines).items():
            if k not in merged:
                merged[k] = v
            elif not merged[k].description and v.description:
                merged[k].description = v.description

    return list(merged.values())


def enrich_type_hints(
    params: list[ParamRecord], sig: inspect.Signature
) -> list[ParamRecord]:
    """Fill missing type hints from the real Python signature annotations."""
    for p in params:
        if p.name in sig.parameters:
            ann = sig.parameters[p.name].annotation
            if ann is not inspect.Parameter.empty and not p.type_hint:
                p.type_hint = ann.__name__ if hasattr(ann, "__name__") else str(ann)
    return params


# ── Object classifier ─────────────────────────────────────────────────────────


def _kind(obj) -> Optional[str]:
    if inspect.isbuiltin(obj):
        return "builtin"
    if inspect.isfunction(obj):
        return "function"
    if inspect.isclass(obj):
        return "class"
    if inspect.ismethod(obj):
        return "method"
    return None


# ── Extraction ────────────────────────────────────────────────────────────────


def extract_from_module(module_path: str, library_name: str) -> list[FunctionRecord]:
    """Extract all public callables from a single module."""
    try:
        module = importlib.import_module(module_path)
    except Exception:
        return []

    records = []
    for name, obj in inspect.getmembers(module):
        if name.startswith("_"):
            continue
        kind = _kind(obj)
        if kind is None:
            continue

        try:
            sig = inspect.signature(obj)
            sig_str = str(sig)
        except (ValueError, TypeError):
            sig = None
            sig_str = "(...)"

        doc = inspect.getdoc(obj) or ""
        params = parse_params(doc)
        if sig:
            params = enrich_type_hints(params, sig)

        records.append(
            FunctionRecord(
                library=library_name,
                module=module_path,
                name=name,
                full_name=f"{module_path}.{name}",
                kind=kind,
                signature=sig_str,
                docstring=doc,
                params=params,
            )
        )
    return records


def extract_library(library_name: str) -> list[FunctionRecord]:
    """Walk all submodules of a library and extract function records."""
    import_name = IMPORT_REMAP.get(library_name, library_name)

    try:
        lib = importlib.import_module(import_name)
    except ImportError:
        print(f"  [Not installed] {library_name}")
        return []

    records = []
    seen_modules: set[str] = set()

    # Root module
    records.extend(extract_from_module(import_name, library_name))
    seen_modules.add(import_name)

    # Submodules
    lib_path = getattr(lib, "__path__", None)
    if lib_path:
        n_walked = 0
        for _finder, modname, _ispkg in pkgutil.walk_packages(
            path=lib_path, prefix=import_name + ".", onerror=lambda x: None
        ):
            if n_walked >= MAX_MODULES:
                break
            if modname in seen_modules:
                continue
            # Skip private / test / vendor submodules
            parts = modname.split(".")
            if any(
                p.startswith("_") or p in ("tests", "test", "testing", "_vendor")
                for p in parts
            ):
                continue

            records.extend(extract_from_module(modname, library_name))
            seen_modules.add(modname)
            n_walked += 1

    # Deduplicate by full_name
    seen_names: set[str] = set()
    unique: list[FunctionRecord] = []
    for r in records:
        if r.full_name not in seen_names:
            seen_names.add(r.full_name)
            unique.append(r)

    print(
        f"  Extracted {len(unique):4d} items from {library_name:<20} ({len(seen_modules)} modules)"
    )
    return unique
