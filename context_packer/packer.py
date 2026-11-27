"""Pack retrieved chunks into a token-budgeted, cited LLM context."""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence

from .text import (TokenCounter, approx_tokens, bow_vector, cosine, fingerprint,
                   jaccard, shingles, truncate_to_tokens)

ORDERS = ("score", "edges", "source")


@dataclass
class Chunk:
    """One retrieved passage."""
    text: str
    id: str = ""
    score: float = 0.0
    source: str = ""
    position: int = 0          # order inside the source document
    metadata: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, d: Dict[str, Any], index: int = 0) -> "Chunk":
        known = {"text", "id", "score", "source", "position", "metadata"}
        meta = dict(d.get("metadata") or {})
        meta.update({k: v for k, v in d.items() if k not in known})
        return cls(text=str(d["text"]), id=str(d.get("id") or f"c{index}"),
                   score=float(d.get("score", 0.0)), source=str(d.get("source", "")),
                   position=int(d.get("position", 0)), metadata=meta)


@dataclass
class Dropped:
    chunk: Chunk
    reason: str               # "empty" | "duplicate" | "near-duplicate" | "min_score" | "budget" | "max_chunks"
    detail: str = ""


@dataclass
class Packed:
    """Result of packing: the context string plus everything needed to audit it."""
    text: str
    chunks: List[Chunk]                 # in final order; chunk n has citation [n]
    citations: Dict[int, Chunk]
    tokens: int
    budget: int
    dropped: List[Dropped]
    truncated_ids: List[str]

    def sources(self) -> List[str]:
        seen: List[str] = []
        for c in self.chunks:
            if c.source and c.source not in seen:
                seen.append(c.source)
        return seen

    def summary(self) -> Dict[str, Any]:
        reasons: Dict[str, int] = {}
        for d in self.dropped:
            reasons[d.reason] = reasons.get(d.reason, 0) + 1
        return {"kept": len(self.chunks), "dropped": reasons, "tokens": self.tokens,
                "budget": self.budget, "truncated": len(self.truncated_ids),
                "sources": self.sources()}


Embedder = Callable[[str], Any]


class ContextPacker:
    """Turn a ranked list of chunks into the context block of a RAG prompt.

    Steps: drop empty / low-score chunks -> remove exact and near duplicates
    (keeps the higher score) -> optional MMR re-ranking for diversity ->
    greedy fill of the token budget (optionally truncating the last chunk on a
    sentence boundary) -> reorder ("score", "edges" for lost-in-the-middle, or
    "source" to keep documents together) -> render with [n] citation markers.
    """

    def __init__(self, budget: int, *, token_counter: TokenCounter = approx_tokens,
                 template: str = "[{n}] ({source}) {text}", separator: str = "\n\n",
                 header: str = "", near_dup_threshold: float = 0.8, shingle_size: int = 5,
                 mmr_lambda: Optional[float] = None, embedder: Optional[Embedder] = None,
                 order: str = "score", min_score: Optional[float] = None,
                 max_chunks: Optional[int] = None, allow_truncate: bool = True,
                 min_truncated_tokens: int = 20) -> None:
        if budget <= 0:
            raise ValueError("budget must be > 0")
        if order not in ORDERS:
            raise ValueError(f"order must be one of {ORDERS}")
        if mmr_lambda is not None and not 0.0 <= mmr_lambda <= 1.0:
            raise ValueError("mmr_lambda must be in [0, 1]")
        if not 0.0 < near_dup_threshold <= 1.0:
            raise ValueError("near_dup_threshold must be in (0, 1]")
        self.budget = budget
        self.count = token_counter
        self.template = template
        self.separator = separator
        self.header = header
        self.near_dup_threshold = near_dup_threshold
        self.shingle_size = shingle_size
        self.mmr_lambda = mmr_lambda
        self.embed = embedder or bow_vector
        self.order = order
        self.min_score = min_score
        self.max_chunks = max_chunks
        self.allow_truncate = allow_truncate
        self.min_truncated_tokens = min_truncated_tokens

    # ---- stages -------------------------------------------------------------
    def dedupe(self, chunks: Sequence[Chunk]) -> "tuple[List[Chunk], List[Dropped]]":
        ranked = sorted(chunks, key=lambda c: -c.score)  # stable: ties keep input order
        kept: List[Chunk] = []
        kept_sh: List[set] = []
        seen: Dict[str, Chunk] = {}
        dropped: List[Dropped] = []
        for c in ranked:
            fp = fingerprint(c.text)
            if fp in seen:
                dropped.append(Dropped(c, "duplicate", f"same text as {seen[fp].id}"))
                continue
            sh = shingles(c.text, self.shingle_size)
            dup_of = None
            for k, ksh in zip(kept, kept_sh):
                if jaccard(sh, ksh) >= self.near_dup_threshold:
                    dup_of = k
                    break
            if dup_of is not None:
                dropped.append(Dropped(c, "near-duplicate", f"overlaps {dup_of.id}"))
                continue
            seen[fp] = c
            kept.append(c)
            kept_sh.append(sh)
        return kept, dropped

    def mmr(self, chunks: Sequence[Chunk], query: Optional[str]) -> List[Chunk]:
        """Maximal Marginal Relevance: lambda*relevance - (1-lambda)*max_sim_to_selected.

        Relevance is the retriever score scaled to [0, 1]; with a ``query`` it is
        the average of that and query similarity.
        """
        lam = self.mmr_lambda
        if lam is None or len(chunks) < 2:
            return list(chunks)
        vecs = [self.embed(c.text) for c in chunks]
        scores = [c.score for c in chunks]
        lo, hi = min(scores), max(scores)
        rel = [(s - lo) / (hi - lo) if hi > lo else 1.0 for s in scores]
        if query:
            qv = self.embed(query)
            rel = [(r + cosine(qv, v)) / 2 for r, v in zip(rel, vecs)]
        remaining = list(range(len(chunks)))
        picked: List[int] = []
        while remaining:
            def gain(i: int) -> float:
                red = max((cosine(vecs[i], vecs[j]) for j in picked), default=0.0)
                return lam * rel[i] - (1 - lam) * red
            best = max(remaining, key=lambda i: (gain(i), -i))
            picked.append(best)
            remaining.remove(best)
        return [chunks[i] for i in picked]

    def render_one(self, n: int, c: Chunk) -> str:
        return self.template.format(n=n, id=c.id, source=c.source or "unknown",
                                    text=c.text, score=c.score, **_safe_meta(c.metadata))

    def _reorder(self, chunks: List[Chunk]) -> List[Chunk]:
        if self.order == "edges":
            # best chunks at the start and the end, weakest in the middle
            front: List[Chunk] = []
            back: List[Chunk] = []
            for i, c in enumerate(chunks):
                (front if i % 2 == 0 else back).append(c)
            return front + back[::-1]
        if self.order == "source":
            first: Dict[str, int] = {}
            for i, c in enumerate(chunks):
                first.setdefault(c.source, i)
            return sorted(chunks, key=lambda c: (first[c.source], c.position))
        return chunks

    # ---- main entry ---------------------------------------------------------
    def pack(self, chunks: Sequence[Any], query: Optional[str] = None) -> Packed:
        items = [c if isinstance(c, Chunk) else Chunk.from_dict(c, i)
                 for i, c in enumerate(chunks)]
        dropped: List[Dropped] = []
        live: List[Chunk] = []
        for c in items:
            if not c.text.strip():
                dropped.append(Dropped(c, "empty"))
            elif self.min_score is not None and c.score < self.min_score:
                dropped.append(Dropped(c, "min_score", f"{c.score} < {self.min_score}"))
            else:
                live.append(c)
        live, dups = self.dedupe(live)
        dropped += dups
        live = self.mmr(live, query)

        # The separator and header cost tokens too, so measure the real rendered
        # text instead of summing per-chunk counts.
        selected: List[Chunk] = []
        truncated: List[str] = []
        for c in live:
            if self.max_chunks is not None and len(selected) >= self.max_chunks:
                dropped.append(Dropped(c, "max_chunks"))
                continue
            trial = selected + [c]
            if self._tokens(self._reorder(trial)) <= self.budget:
                selected = trial
                continue
            if self.allow_truncate:
                cut = self._fit_truncated(selected, c)
                if cut is not None:
                    selected = selected + [cut]
                    truncated.append(c.id)
                    continue
            dropped.append(Dropped(c, "budget"))

        final = self._reorder(selected)
        text = self._render(final)
        return Packed(text=text, chunks=final, citations={i + 1: c for i, c in enumerate(final)},
                      tokens=self.count(text) if text else 0, budget=self.budget,
                      dropped=dropped, truncated_ids=truncated)

    # ---- helpers ------------------------------------------------------------
    def _render(self, ordered: List[Chunk]) -> str:
        body = self.separator.join(self.render_one(i + 1, c) for i, c in enumerate(ordered))
        if self.header and body:
            return self.header + self.separator + body
        return body

    def _tokens(self, ordered: List[Chunk]) -> int:
        return self.count(self._render(ordered))

    def _fit_truncated(self, selected: List[Chunk], c: Chunk) -> Optional[Chunk]:
        overhead_chunk = Chunk(text="", id=c.id, score=c.score, source=c.source,
                               position=c.position, metadata=c.metadata)
        base = self._tokens(self._reorder(selected + [overhead_chunk]))
        room = self.budget - base
        if room < self.min_truncated_tokens:
            return None
        lo_text = truncate_to_tokens(c.text, room, self.count)
        # shrink until the rendered whole fits (token counts are not additive)
        while lo_text:
            cand = Chunk(text=lo_text, id=c.id, score=c.score, source=c.source,
                         position=c.position, metadata={**c.metadata, "truncated": True})
            if self._tokens(self._reorder(selected + [cand])) <= self.budget:
                if self.count(lo_text) < self.min_truncated_tokens:
                    return None
                return cand
            room -= 1
            lo_text = truncate_to_tokens(c.text, room, self.count)
        return None


def _safe_meta(meta: Dict[str, Any]) -> Dict[str, Any]:
    reserved = {"n", "id", "source", "text", "score"}
    return {k: v for k, v in meta.items() if isinstance(k, str) and k not in reserved}


_CITE = re.compile(r"\[(\d+(?:\s*[,\-–]\s*\d+)*)\]")


def extract_citations(answer: str) -> List[int]:
    """Citation numbers used in an answer: "[1]", "[2, 3]", "[4-6]" -> [1, 2, 3, 4, 5, 6]."""
    out: List[int] = []
    for m in _CITE.finditer(answer):
        for part in re.split(r"\s*,\s*", m.group(1)):
            if re.search(r"[\-–]", part):
                a, b = (int(x) for x in re.split(r"\s*[\-–]\s*", part))
                rng = range(a, b + 1) if a <= b else range(a, b - 1, -1)
                out.extend(rng)
            else:
                out.append(int(part))
    seen: List[int] = []
    for n in out:
        if n not in seen:
            seen.append(n)
    return seen


def check_citations(answer: str, packed: Packed) -> Dict[str, Any]:
    """Map an LLM answer's [n] markers back to chunks and flag invalid ones."""
    used = extract_citations(answer)
    valid = [n for n in used if n in packed.citations]
    invalid = [n for n in used if n not in packed.citations]
    unused = [n for n in packed.citations if n not in used]
    return {
        "cited": valid,
        "invalid": invalid,
        "uncited": unused,
        "sources": sorted({packed.citations[n].source for n in valid if packed.citations[n].source}),
        "ok": bool(valid) and not invalid,
    }
