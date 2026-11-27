# rag-context-packer

Fit retrieved chunks into an LLM prompt **without wasting tokens**: remove exact and
near-duplicate chunks, re-rank for diversity with **MMR**, fill a **token budget**
(truncating the last chunk on a sentence boundary), order chunks to avoid the
**lost-in-the-middle** problem, and render them with **`[n]` citations** you can check
in the model's answer. Dependency-free Python (3.10+).

Retrievers usually return the top-k chunks. In real RAG systems many of those are
copies (the same FAQ paragraph in three docs), several say almost the same thing,
and the whole list does not fit the context window that is left after the system
prompt and chat history. This library is the step between the retriever and the
prompt.

```
retriever top-k ─► drop empty / low score ─► dedupe (hash + shingle Jaccard)
               ─► MMR (optional) ─► greedy fill of token budget (+ truncate last)
               ─► order: score | edges | source ─► "[1] (source) text ..." + audit report
```

## Quick start

```python
from context_packer import ContextPacker, check_citations

chunks = [  # from your vector DB / BM25 / hybrid retriever
    {"id": "a1", "source": "refund-policy.md", "score": 0.91, "text": "Customers can request a full refund within 30 days..."},
    {"id": "b1", "source": "faq.md",           "score": 0.89, "text": "Customers can request a full refund within 30 days!..."},
    {"id": "c1", "source": "shipping.md",      "score": 0.52, "text": "Standard shipping takes 3 to 5 business days..."},
]

packer = ContextPacker(budget=1500, order="edges", mmr_lambda=0.7,
                       header="Answer using only these sources. Cite them like [1].")
packed = packer.pack(chunks, query="Can I get a refund after 30 days?")

prompt = f"{packed.text}\n\nQuestion: Can I get a refund after 30 days?"
print(packed.summary())   # {'kept': 2, 'dropped': {'duplicate': 1}, 'tokens': ..., 'sources': [...]}

answer = llm(prompt)      # any model / SDK
print(check_citations(answer, packed))
# {'cited': [1], 'invalid': [], 'uncited': [2], 'sources': ['refund-policy.md'], 'ok': True}
```

Run the offline demo: `python examples/demo.py`.

## Features

| Stage | What it does |
|---|---|
| Filter | drops empty chunks and chunks under `min_score` |
| Exact dedupe | hash of normalised text (case, punctuation and whitespace ignored); keeps the higher score |
| Near-duplicate dedupe | word k-shingles + Jaccard ≥ `near_dup_threshold` (default 0.8) |
| MMR | `mmr_lambda` trades relevance vs. redundancy; relevance = scaled retriever score (+ query similarity if a `query` is given); bag-of-words cosine by default or your own `embedder` |
| Budget | measures the **rendered** text (template, header and separators included), so the result never exceeds `budget`; smaller later chunks can still fill gaps; `max_chunks` cap |
| Truncation | the chunk that does not fit is cut on a sentence (then word) boundary when at least `min_truncated_tokens` fit; marked `metadata["truncated"]` |
| Ordering | `score` (best first), `edges` (best at start **and** end, weakest in the middle), `source` (group by document, then `position`) |
| Rendering | `template="[{n}] ({source}) {text}"` – any field or metadata key can be used |
| Audit | `Packed.dropped` gives every removed chunk with a reason; `summary()`; `check_citations()` flags invented `[n]` markers |

Token counting uses a ~4 chars/token estimate by default. Pass `token_counter=` for an
exact count (for example `lambda t: len(enc.encode(t))` with tiktoken, or a cached call
to a provider's token-count endpoint).

## CLI

```bash
python -m context_packer examples/chunks.jsonl --budget 130 --order edges --mmr 0.7 \
    --query "refund after 30 days" --json
```

## Tests

```bash
python -m unittest discover -s tests -v
```

## License

MIT
