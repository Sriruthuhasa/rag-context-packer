"""CLI: python -m context_packer chunks.jsonl --budget 800 [--query ...] [--order edges] [--mmr 0.7] [--json]"""
from __future__ import annotations

import argparse
import json
import sys

from .packer import ORDERS, ContextPacker


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="context_packer", description=__doc__)
    p.add_argument("file", help="JSONL with one chunk per line (text, id, score, source, position); '-' = stdin")
    p.add_argument("--budget", type=int, required=True, help="max tokens for the context")
    p.add_argument("--query", default=None)
    p.add_argument("--order", choices=ORDERS, default="score")
    p.add_argument("--mmr", type=float, default=None, help="MMR lambda in [0,1] (off by default)")
    p.add_argument("--near-dup", type=float, default=0.8, help="shingle Jaccard threshold")
    p.add_argument("--min-score", type=float, default=None)
    p.add_argument("--max-chunks", type=int, default=None)
    p.add_argument("--no-truncate", action="store_true")
    p.add_argument("--json", action="store_true", help="print a JSON report instead of the context")
    a = p.parse_args(argv)

    fh = sys.stdin if a.file == "-" else open(a.file, encoding="utf-8")
    with fh:
        rows = [json.loads(line) for line in fh if line.strip()]
    packer = ContextPacker(a.budget, order=a.order, mmr_lambda=a.mmr, near_dup_threshold=a.near_dup,
                           min_score=a.min_score, max_chunks=a.max_chunks,
                           allow_truncate=not a.no_truncate)
    packed = packer.pack(rows, query=a.query)
    if a.json:
        print(json.dumps({
            "context": packed.text,
            "summary": packed.summary(),
            "citations": {n: {"id": c.id, "source": c.source} for n, c in packed.citations.items()},
            "dropped": [{"id": d.chunk.id, "reason": d.reason, "detail": d.detail} for d in packed.dropped],
        }, indent=2))
    else:
        print(packed.text)
        print(f"\n# {json.dumps(packed.summary())}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
