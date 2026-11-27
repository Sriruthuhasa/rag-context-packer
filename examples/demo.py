"""Offline demo: pack retrieved chunks for a RAG prompt and check the answer's citations."""
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from context_packer import ContextPacker, check_citations  # noqa: E402

rows = [json.loads(l) for l in (pathlib.Path(__file__).parent / "chunks.jsonl").read_text().splitlines() if l.strip()]

packer = ContextPacker(budget=130, order="edges", mmr_lambda=0.7,
                       header="Answer using only these sources. Cite them like [1].")
packed = packer.pack(rows, query="Can I get a refund after 30 days?")

print(packed.text)
print("\nsummary:", packed.summary())
for d in packed.dropped:
    print(f"dropped {d.chunk.id}: {d.reason} {d.detail}".rstrip())

# Pretend this came back from the LLM:
answer = "Full refunds are possible within 30 days [1]; after that, store credit may be offered [3]. See [7]."
print("\ncitation check:", check_citations(answer, packed))
assert packed.tokens <= packer.budget
