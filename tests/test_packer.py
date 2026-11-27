import io
import json
import os
import sys
import tempfile
import unittest
from contextlib import redirect_stdout, redirect_stderr

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from context_packer import (Chunk, ContextPacker, approx_tokens, check_citations,  # noqa: E402
                            cosine, extract_citations, truncate_to_tokens)
from context_packer.__main__ import main  # noqa: E402
from context_packer.text import fingerprint, jaccard, shingles  # noqa: E402


def wc(text):  # exact, easy-to-reason token counter for tests
    return len(text.split())


def mk(i, text, score, source="doc", position=0):
    return Chunk(text=text, id=f"c{i}", score=score, source=source, position=position)


LONG = ("Retrieval augmented generation grounds answers in documents. "
        "Chunks are retrieved by a vector index. The best chunks go into the prompt. "
        "Too many chunks waste tokens. Duplicates waste even more tokens.")


class TextTests(unittest.TestCase):
    def test_approx_tokens(self):
        self.assertEqual(approx_tokens(""), 0)
        self.assertEqual(approx_tokens("abc"), 1)
        self.assertEqual(approx_tokens("a" * 9), 3)

    def test_fingerprint_ignores_case_and_punctuation(self):
        self.assertEqual(fingerprint("Hello, World!"), fingerprint("hello   world"))
        self.assertNotEqual(fingerprint("hello world"), fingerprint("hello there"))

    def test_jaccard_and_shingles(self):
        a = shingles("one two three four five six", 3)
        self.assertEqual(len(a), 4)
        self.assertEqual(jaccard(a, a), 1.0)
        self.assertEqual(jaccard(a, set()), 0.0)
        self.assertEqual(shingles("short text", 5), {("short", "text")})

    def test_cosine_dense_and_sparse(self):
        self.assertAlmostEqual(cosine([1, 0], [1, 0]), 1.0)
        self.assertAlmostEqual(cosine([1, 0], [0, 1]), 0.0)
        self.assertAlmostEqual(cosine({"a": 1}, {"a": 2, "b": 0}), 1.0)
        self.assertEqual(cosine([0, 0], [1, 1]), 0.0)
        with self.assertRaises(ValueError):
            cosine([1], [1, 2])

    def test_truncate_prefers_sentences(self):
        out = truncate_to_tokens(LONG, 12, wc)
        self.assertTrue(out.endswith("…"))
        self.assertLessEqual(wc(out), 12)
        self.assertTrue(out.startswith("Retrieval augmented generation grounds answers in documents."))

    def test_truncate_falls_back_to_words(self):
        out = truncate_to_tokens("one two three four five six seven", 4, wc)
        self.assertEqual(out, "one two three …")
        self.assertEqual(truncate_to_tokens("x y", 0, wc), "")
        self.assertEqual(truncate_to_tokens("fits", 5, wc), "fits")


class PackTests(unittest.TestCase):
    def test_respects_budget_and_numbers_citations(self):
        chunks = [mk(i, f"fact number {i} " + "word " * 8, 1 - i / 10, f"s{i}") for i in range(6)]
        p = ContextPacker(40, token_counter=wc, allow_truncate=False).pack(chunks)
        self.assertLessEqual(p.tokens, 40)
        self.assertEqual(p.tokens, wc(p.text))
        self.assertEqual(sorted(p.citations), list(range(1, len(p.chunks) + 1)))
        self.assertIn("[1] (s0) fact number 0", p.text)
        self.assertTrue(all(d.reason == "budget" for d in p.dropped))
        self.assertEqual(len(p.chunks) + len(p.dropped), 6)

    def test_separator_and_header_are_counted(self):
        chunks = [mk(i, "a b c", 1.0 - i / 10) for i in range(3)]
        # each rendered chunk: "[n] (doc) a b c" = 5 words; header 3 words
        packer = ContextPacker(13, token_counter=wc, header="use these sources",
                               separator=" | ", allow_truncate=False)
        p = packer.pack(chunks)
        # header(3) + sep(1) + 5 = 9; adding another: +1 sep +5 = 15 > 13
        self.assertEqual(len(p.chunks), 1)
        self.assertLessEqual(wc(p.text), 13)

    def test_exact_duplicate_keeps_higher_score(self):
        chunks = [mk(0, "Refunds within 30 days.", 0.5), mk(1, "refunds WITHIN 30 days!", 0.9)]
        p = ContextPacker(100).pack(chunks)
        self.assertEqual([c.id for c in p.chunks], ["c1"])
        self.assertEqual(p.dropped[0].reason, "duplicate")
        self.assertIn("c1", p.dropped[0].detail)

    def test_near_duplicate_removed(self):
        base = "the quick brown fox jumps over the lazy dog near the old river bank today"
        chunks = [mk(0, base, 0.9), mk(1, base + " again", 0.8),
                  mk(2, "completely different content about invoices and tax rules", 0.7)]
        p = ContextPacker(200, near_dup_threshold=0.7).pack(chunks)
        self.assertEqual([c.id for c in p.chunks], ["c0", "c2"])
        self.assertEqual(p.dropped[0].reason, "near-duplicate")

    def test_empty_and_min_score_dropped(self):
        chunks = [mk(0, "   ", 0.9), mk(1, "low score text", 0.1), mk(2, "good text", 0.8)]
        p = ContextPacker(100, min_score=0.2).pack(chunks)
        self.assertEqual([c.id for c in p.chunks], ["c2"])
        self.assertEqual({d.reason for d in p.dropped}, {"empty", "min_score"})

    def test_max_chunks(self):
        chunks = [mk(i, f"unique text {i} alpha{i}", 1 - i / 10) for i in range(5)]
        p = ContextPacker(1000, max_chunks=2).pack(chunks)
        self.assertEqual(len(p.chunks), 2)
        self.assertEqual(p.summary()["dropped"], {"max_chunks": 3})

    def test_truncates_last_chunk_to_fill_budget(self):
        chunks = [mk(0, "short first chunk here", 0.9, "a"), mk(1, LONG, 0.8, "b")]
        p = ContextPacker(25, token_counter=wc, min_truncated_tokens=5).pack(chunks)
        self.assertEqual(len(p.chunks), 2)
        self.assertEqual(p.truncated_ids, ["c1"])
        self.assertTrue(p.chunks[1].metadata.get("truncated"))
        self.assertLessEqual(p.tokens, 25)

    def test_no_truncation_when_room_too_small(self):
        chunks = [mk(0, "w " * 20, 0.9), mk(1, LONG, 0.8)]
        p = ContextPacker(26, token_counter=wc, min_truncated_tokens=10).pack(chunks)
        self.assertEqual(len(p.chunks), 1)
        self.assertEqual(p.dropped[0].reason, "budget")

    def test_smaller_later_chunk_still_fits(self):
        chunks = [mk(0, "a " * 10, 0.9), mk(1, "b " * 50, 0.8), mk(2, "c d", 0.7)]
        p = ContextPacker(20, token_counter=wc, allow_truncate=False).pack(chunks)
        self.assertEqual([c.id for c in p.chunks], ["c0", "c2"])

    def test_edges_order_puts_best_at_both_ends(self):
        chunks = [mk(i, f"distinct topic {i} " + chr(97 + i) * 3, 1 - i / 10) for i in range(5)]
        p = ContextPacker(1000, order="edges").pack(chunks)
        self.assertEqual([c.id for c in p.chunks], ["c0", "c2", "c4", "c3", "c1"])

    def test_source_order_groups_documents_by_position(self):
        chunks = [mk(0, "alpha one", 0.9, "A", 3), mk(1, "beta one", 0.8, "B", 0),
                  mk(2, "alpha two", 0.7, "A", 1)]
        p = ContextPacker(1000, order="source").pack(chunks)
        self.assertEqual([c.id for c in p.chunks], ["c2", "c0", "c1"])
        self.assertEqual(p.sources(), ["A", "B"])

    def test_mmr_prefers_diverse_chunks(self):
        chunks = [mk(0, "python agents tools memory planning", 0.95),
                  mk(1, "python agents tools memory loops", 0.94),
                  mk(2, "billing invoices tax refunds", 0.90)]
        no_mmr = ContextPacker(1000, near_dup_threshold=1.0).pack(chunks)
        self.assertEqual([c.id for c in no_mmr.chunks], ["c0", "c1", "c2"])
        with_mmr = ContextPacker(1000, near_dup_threshold=1.0, mmr_lambda=0.5).pack(chunks)
        self.assertEqual([c.id for c in with_mmr.chunks], ["c0", "c2", "c1"])

    def test_mmr_lambda_one_is_pure_relevance(self):
        chunks = [mk(0, "x y z", 0.9), mk(1, "x y w", 0.8), mk(2, "q r s", 0.1)]
        p = ContextPacker(1000, mmr_lambda=1.0, near_dup_threshold=1.0).pack(chunks)
        self.assertEqual([c.id for c in p.chunks], ["c0", "c1", "c2"])

    def test_mmr_uses_query_and_custom_embedder(self):
        vecs = {"cats": [1.0, 0.0], "dogs": [0.0, 1.0], "q": [0.0, 1.0]}
        chunks = [mk(0, "cats", 0.5), mk(1, "dogs", 0.5)]
        p = ContextPacker(1000, mmr_lambda=1.0, embedder=lambda t: vecs[t]).pack(chunks, query="q")
        self.assertEqual(p.chunks[0].id, "c1")

    def test_accepts_dicts_and_extra_fields_go_to_metadata(self):
        p = ContextPacker(1000, template="[{n}] {title}: {text}").pack(
            [{"text": "hello", "score": 1, "title": "Greeting"}])
        self.assertEqual(p.text, "[1] Greeting: hello")
        self.assertEqual(p.chunks[0].id, "c0")

    def test_empty_input(self):
        p = ContextPacker(10, header="H").pack([])
        self.assertEqual((p.text, p.tokens, p.chunks), ("", 0, []))

    def test_bad_arguments(self):
        for kw in ({"budget": 0}, {"budget": 5, "order": "random"},
                   {"budget": 5, "mmr_lambda": 2}, {"budget": 5, "near_dup_threshold": 0}):
            with self.assertRaises(ValueError):
                ContextPacker(**kw)


class CitationTests(unittest.TestCase):
    def test_extract(self):
        self.assertEqual(extract_citations("a [1] b [2, 3] c [5-6] d [1]"), [1, 2, 3, 5, 6])
        self.assertEqual(extract_citations("no cites, [x] or [ ]"), [])

    def test_check(self):
        p = ContextPacker(1000).pack([mk(0, "alpha", 0.9, "A"), mk(1, "beta", 0.8, "B")])
        r = check_citations("Alpha is true [1]. Gamma [4].", p)
        self.assertEqual(r["cited"], [1])
        self.assertEqual(r["invalid"], [4])
        self.assertEqual(r["uncited"], [2])
        self.assertEqual(r["sources"], ["A"])
        self.assertFalse(r["ok"])
        self.assertTrue(check_citations("both [1-2]", p)["ok"])
        self.assertFalse(check_citations("no citations", p)["ok"])


class CliTests(unittest.TestCase):
    def test_cli_json(self):
        rows = [{"id": "a", "text": "first chunk text", "score": 0.9, "source": "x.md"},
                {"id": "b", "text": "first chunk text", "score": 0.5, "source": "y.md"}]
        with tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False) as f:
            f.write("\n".join(json.dumps(r) for r in rows) + "\n")
        try:
            out = io.StringIO()
            with redirect_stdout(out), redirect_stderr(io.StringIO()):
                self.assertEqual(main([f.name, "--budget", "50", "--json"]), 0)
            report = json.loads(out.getvalue())
            self.assertEqual(report["summary"]["kept"], 1)
            self.assertEqual(report["dropped"][0]["reason"], "duplicate")
            self.assertEqual(report["citations"]["1"]["source"], "x.md")
        finally:
            os.unlink(f.name)


if __name__ == "__main__":
    unittest.main()
