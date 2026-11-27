"""rag-context-packer: fit retrieved chunks into a token budget with dedupe, MMR, ordering and citations."""
from .packer import (Chunk, ContextPacker, Dropped, Packed, check_citations,
                     extract_citations)
from .text import approx_tokens, cosine, truncate_to_tokens

__all__ = ["Chunk", "ContextPacker", "Dropped", "Packed", "check_citations",
           "extract_citations", "approx_tokens", "cosine", "truncate_to_tokens"]
__version__ = "0.1.0"
