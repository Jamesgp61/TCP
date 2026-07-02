"""
local_embedder.py
=================
Zero-dependency, pure-Python semantic text vectorizer for the
Topological Context Protocol (TCP).

Converts a source-code file into a consistent 8-dimensional dense
vector using a hybrid statistical / TF-IDF-style feature extractor.
The output is designed to be fed directly into Node 0's
``embedding_map.py`` median-threshold functions.

Only standard-library modules are used: ``math``, ``json``, ``re``.
No torch, transformers, scipy, numpy, or any third-party package.
"""

from __future__ import annotations

import math
import json
import re
import os
from typing import Iterable, List, Sequence, Tuple, Union

# ---------------------------------------------------------------------------
# Public constants
# ---------------------------------------------------------------------------

EMBED_DIM = 8
ZERO_VECTOR: List[float] = [0.0] * EMBED_DIM

# Language-agnostic control-flow keyword set used for feature #2.
_CONTROL_FLOW_KEYWORDS = frozenset({
    "if", "elif", "else", "for", "while", "do", "switch", "case",
    "try", "catch", "finally", "return", "break", "continue",
    "goto", "throw", "yield", "await", "async", "match", "when",
    "unless", "until", "foreach", "def", "fn", "func", "fun",
})

# ---------------------------------------------------------------------------
# Binary / non-text detection
# ---------------------------------------------------------------------------

_TEXT_CHARS = set(range(32, 127)) | {7, 8, 9, 10, 11, 12, 13, 27}


def _is_probably_binary(raw: bytes) -> bool:
    """Heuristic: a file is binary if it contains a NUL byte or more than
    10% non-text bytes within the first 8 KiB."""
    if b"\x00" in raw[:8192]:
        return True
    sample = raw[:8192]
    if not sample:
        return False
    non_text = sum(1 for b in sample if b not in _TEXT_CHARS)
    return (non_text / len(sample)) > 0.10


# ---------------------------------------------------------------------------
# Tokenization
# ---------------------------------------------------------------------------

_TOKEN_RE = re.compile(
    r"""
    (?P<comment>      \#[^\n]* | //[^\n]* | /\*.*?\*/ )
  | (?P<string>       \"\"\"(?:\\.|[^\\])*?\"\"\" | '''(?:\\.|[^\\])*?'''
                    | "(?:\\.|[^"\\])*" | '(?:\\.|[^'\\])*' )
  | (?P<number>       \b\d+(?:\.\d+)?(?:[eE][+-]?\d+)?\b )
  | (?P<identifier>   [A-Za-z_][A-Za-z0-9_]* )
  | (?P<op>           [{}()\[\];,.<>:=+\-*/%&|!?^~@] )
    """,
    re.VERBOSE | re.DOTALL,
)


def _tokenize(text: str) -> List[Tuple[str, str]]:
    """Return a list of (kind, value) tuples."""
    tokens: List[Tuple[str, str]] = []
    for m in _TOKEN_RE.finditer(text):
        kind = m.lastgroup
        value = m.group()
        tokens.append((kind, value))
    return tokens


# ---------------------------------------------------------------------------
# Math helpers
# ---------------------------------------------------------------------------

def _sigmoid(x: float) -> float:
    """Numerically stable sigmoid."""
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


def _shannon_entropy(counts: Sequence[int]) -> float:
    total = sum(counts)
    if total <= 0:
        return 0.0
    h = 0.0
    for c in counts:
        if c <= 0:
            continue
        p = c / total
        h -= p * math.log2(p)
    return h


# ---------------------------------------------------------------------------
# Feature extraction -> 8 dimensions
# ---------------------------------------------------------------------------

def _extract_features(text: str) -> List[float]:
    """Extract 8 raw features from source text and normalize each to [0,1].

    Feature map (dimension : semantic meaning)
    ------------------------------------------
    0 : lexical diversity      (unique identifiers / total identifiers)
    1 : control-flow density   (branch/loop keywords per token)
    2 : comment density        (comment chars / total chars)
    3 : identifier richness    (average identifier length)
    4 : numeric literal density
    5 : string literal density
    6 : structural nesting depth
    7 : token-type Shannon entropy
    """
    if not text:
        return list(ZERO_VECTOR)

    tokens = _tokenize(text)
    if not tokens:
        return list(ZERO_VECTOR)

    n_tokens = len(tokens)
    n_chars = len(text)

    # --- 0. Lexical diversity ---------------------------------------------
    identifiers = [v for k, v in tokens if k == "identifier"]
    n_ident = len(identifiers)
    unique_ident = len(set(identifiers))
    f0 = (unique_ident / n_ident) if n_ident > 0 else 0.0

    # --- 1. Control-flow keyword density ----------------------------------
    cf_count = sum(
        1 for k, v in tokens
        if k == "identifier" and v in _CONTROL_FLOW_KEYWORDS
    )
    cf_density = cf_count / n_tokens
    f1 = _sigmoid((cf_density - 0.05) * 20.0)

    # --- 2. Comment density -----------------------------------------------
    comment_chars = sum(len(v) for k, v in tokens if k == "comment")
    comment_density = comment_chars / max(1, n_chars)
    f2 = _sigmoid((comment_density - 0.10) * 15.0)

    # --- 3. Average identifier length -------------------------------------
    avg_ident_len = (
        sum(len(v) for v in identifiers) / n_ident
    ) if n_ident > 0 else 0.0
    f3 = _sigmoid((avg_ident_len - 6.0) * 0.5)

    # --- 4. Numeric literal density ---------------------------------------
    numbers = [v for k, v in tokens if k == "number"]
    numeric_density = len(numbers) / n_tokens
    f4 = _sigmoid((numeric_density - 0.03) * 30.0)

    # --- 5. String literal density ----------------------------------------
    strings = [v for k, v in tokens if k == "string"]
    string_density = len(strings) / n_tokens
    f5 = _sigmoid((string_density - 0.02) * 30.0)

    # --- 6. Structural nesting depth --------------------------------------
    depth = 0
    max_depth = 0
    for ch in text:
        if ch in "{([":
            depth += 1
            if depth > max_depth:
                max_depth = depth
        elif ch in "})]":
            if depth > 0:
                depth -= 1
    f6 = _sigmoid((max_depth - 4.0) * 0.5)

    # --- 7. Token-type Shannon entropy ------------------------------------
    type_counts: dict = {}
    for k, _ in tokens:
        type_counts[k] = type_counts.get(k, 0) + 1
    entropy = _shannon_entropy(list(type_counts.values()))
    f7 = max(0.0, min(1.0, entropy / 2.5))

    return [f0, f1, f2, f3, f4, f5, f6, f7]


# ---------------------------------------------------------------------------
# Public embedding API
# ---------------------------------------------------------------------------

def embed_text(text: str) -> List[float]:
    """Embed a raw text string into an 8-dimensional dense vector."""
    return _extract_features(text)


def embed_file(path: Union[str, "os.PathLike"]) -> List[float]:
    """Embed a file on disk.

    Safety guarantees:
      * Missing / unreadable file  -> ZERO_VECTOR
      * Empty file                 -> ZERO_VECTOR
      * Binary / non-text asset    -> ZERO_VECTOR

    Returning a zero vector (rather than raising) keeps downstream
    median-threshold logic stable: zero vectors naturally fall below
    any non-trivial median and are filtered out.
    """
    try:
        with open(path, "rb") as fh:
            raw = fh.read()
    except (OSError, IOError, TypeError):
        return list(ZERO_VECTOR)

    if not raw:
        return list(ZERO_VECTOR)

    if _is_probably_binary(raw):
        return list(ZERO_VECTOR)

    try:
        text = raw.decode("utf-8", errors="strict")
    except UnicodeDecodeError:
        try:
            text = raw.decode("latin-1", errors="replace")
        except Exception:
            return list(ZERO_VECTOR)

    return _extract_features(text)


def embed_batch(
    paths: Iterable[Union[str, "os.PathLike"]],
) -> List[List[float]]:
    """Embed many files. Non-text / empty files yield zero vectors."""
    return [embed_file(p) for p in paths]


# Alias for callers that prefer a different name.
vectorize = embed_file


# ---------------------------------------------------------------------------
# Node 0 interface: median-threshold bridge  (embedding_map.py)
# ---------------------------------------------------------------------------

def _median(values: List[float]) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    n = len(s)
    mid = n // 2
    if n % 2 == 1:
        return s[mid]
    return (s[mid - 1] + s[mid]) / 2.0


def median_threshold_mask(
    vectors: List[List[float]],
    dim: int = 0,
) -> List[bool]:
    """Return a boolean mask: ``True`` for vectors whose ``dim``-th
    component is >= the median of that component across the batch.

    This is the canonical entry point expected by ``embedding_map.py``.
    """
    if not vectors:
        return []
    column = [v[dim] for v in vectors]
    m = _median(column)
    return [v[dim] >= m for v in vectors]


def median_threshold_split(
    vectors: List[List[float]],
    dim: int = 0,
) -> Tuple[List[List[float]], List[List[float]]]:
    """Split a batch into ``(above_or_equal, below)`` based on the
    median of the chosen dimension."""
    mask = median_threshold_mask(vectors, dim=dim)
    above = [v for v, keep in zip(vectors, mask) if keep]
    below = [v for v, keep in zip(vectors, mask) if not keep]
    return above, below


def build_embedding_map(
    paths: Iterable[Union[str, "os.PathLike"]],
) -> dict:
    """End-to-end Node 0 entry point.

    1. Embeds every path.
    2. Computes the per-dimension median vector.
    3. Returns a dict ready to be consumed by ``embedding_map.py``::

           {
               "dim": 8,
               "vectors":  [[...], ...],
               "median":   [m0, ..., m7],
               "mask":     [bool, ...],   # True = above median on dim 0
           }
    """
    vectors = embed_batch(paths)
    if not vectors:
        return {"dim": EMBED_DIM, "vectors": [], "median": list(ZERO_VECTOR), "mask": []}

    per_dim_medians = [
        _median([v[d] for v in vectors])
        for d in range(EMBED_DIM)
    ]
    mask = median_threshold_mask(vectors, dim=0)
    return {
        "dim": EMBED_DIM,
        "vectors": vectors,
        "median": per_dim_medians,
        "mask": mask,
    }


# ---------------------------------------------------------------------------
# JSON serialization for hand-off to embedding_map.py
# ---------------------------------------------------------------------------

def to_json(vectors: List[List[float]]) -> str:
    """Serialize vectors for hand-off to ``embedding_map.py``."""
    return json.dumps({"dim": EMBED_DIM, "vectors": vectors})


def from_json(payload: str) -> List[List[float]]:
    """Inverse of :func:`to_json`."""
    obj = json.loads(payload)
    return obj["vectors"]


def map_to_json(paths: Iterable[Union[str, "os.PathLike"]]) -> str:
    """Convenience: build the full embedding map and serialize to JSON."""
    return json.dumps(build_embedding_map(paths))


# ---------------------------------------------------------------------------
# CLI / self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("usage: python local_embedder.py <file> [<file> ...]")
        sys.exit(1)

    result = build_embedding_map(sys.argv[1:])
    print(json.dumps(result, indent=2))
