"""Small text utilities: tokens, normalisation, shingles, bag-of-words vectors."""
from __future__ import annotations

import hashlib
import math
import re
from collections import Counter
from typing import Callable, Dict, Iterable, List, Sequence, Set

_WORD = re.compile(r"[A-Za-z0-9]+(?:'[A-Za-z]+)?")
_SENT_END = re.compile(r"(?<=[.!?])\s+")

TokenCounter = Callable[[str], int]

STOPWORDS = frozenset(
    "a an and are as at be by for from has have in is it its of on or that the "
    "this to was were will with".split()
)


def approx_tokens(text: str) -> int:
    """Cheap, model-agnostic token estimate (~4 characters per token).

    Good enough for budgeting; plug in a real tokenizer (tiktoken, the
    Anthropic token-count API, ...) via ``token_counter=`` for exact numbers.
    """
    if not text:
        return 0
    return max(1, math.ceil(len(text) / 4))


def words(text: str) -> List[str]:
    return [w.lower() for w in _WORD.findall(text)]


def normalize(text: str) -> str:
    return " ".join(words(text))


def fingerprint(text: str) -> str:
    """Hash of the normalised text: equal for chunks that differ only in case,
    punctuation or whitespace."""
    return hashlib.sha1(normalize(text).encode("utf-8")).hexdigest()


def shingles(text: str, k: int = 5) -> Set[tuple]:
    ws = words(text)
    if len(ws) < k:
        return {tuple(ws)} if ws else set()
    return {tuple(ws[i : i + k]) for i in range(len(ws) - k + 1)}


def jaccard(a: Set, b: Set) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def bow_vector(text: str) -> Dict[str, float]:
    """Bag-of-words vector without stopwords (default similarity for MMR)."""
    return dict(Counter(w for w in words(text) if w not in STOPWORDS))


def cosine(a, b) -> float:
    """Cosine similarity for sparse dict vectors or dense sequences."""
    if isinstance(a, dict):
        if len(a) > len(b):
            a, b = b, a
        dot = sum(v * b.get(k, 0.0) for k, v in a.items())
        na = math.sqrt(sum(v * v for v in a.values()))
        nb = math.sqrt(sum(v * v for v in b.values()))
    else:
        if len(a) != len(b):
            raise ValueError("vectors have different lengths")
        dot = sum(x * y for x, y in zip(a, b))
        na = math.sqrt(sum(x * x for x in a))
        nb = math.sqrt(sum(y * y for y in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


def split_sentences(text: str) -> List[str]:
    return [s for s in _SENT_END.split(text.strip()) if s]


def truncate_to_tokens(text: str, max_tokens: int, count: TokenCounter,
                       marker: str = " …") -> str:
    """Cut ``text`` so ``count(result) <= max_tokens``.

    Prefers whole sentences; falls back to whole words. Returns "" when not
    even one word fits.
    """
    if max_tokens <= 0:
        return ""
    if count(text) <= max_tokens:
        return text
    out = ""
    for sent in split_sentences(text):
        cand = (out + " " + sent).strip()
        if count(cand + marker) > max_tokens:
            break
        out = cand
    if out:
        return out + marker
    out = ""
    for w in text.split():
        cand = (out + " " + w).strip()
        if count(cand + marker) > max_tokens:
            break
        out = cand
    return out + marker if out else ""


def join_nonempty(parts: Iterable[str], sep: str) -> str:
    return sep.join(p for p in parts if p)


__all__: Sequence[str] = [
    "approx_tokens", "words", "normalize", "fingerprint", "shingles", "jaccard",
    "bow_vector", "cosine", "split_sentences", "truncate_to_tokens", "TokenCounter",
]
