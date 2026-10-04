import re
from typing import List, Tuple


def _find_boundary_index(window: str) -> int:
    """Return the end position of the last useful boundary within a text window."""
    patterns = [r"\n\s*\n+", r"[.!?]\s+", r";\s+", r"\n+"]
    best_index = -1

    for pattern in patterns:
        matches = list(re.finditer(pattern, window))
        if not matches:
            continue
        last_match = matches[-1]
        boundary_end = last_match.end()
        if boundary_end > best_index:
            best_index = boundary_end

    return best_index if best_index > 0 else -1


def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> List[Tuple[str, int, int]]:
    """Split text into RAG-friendly chunks while preserving overlap and useful boundaries.

    Returns a list of tuples: (chunk_text, start_char_idx, end_char_idx)
    """
    if not text or not text.strip():
        raise ValueError("text must not be empty")
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if overlap < 0:
        raise ValueError("overlap must be non-negative")

    chunks: List[Tuple[str, int, int]] = []
    start = 0
    text_len = len(text)

    while start < text_len:
        end = min(start + chunk_size, text_len)
        window = text[start:end]

        boundary_index = _find_boundary_index(window)
        if boundary_index > 0 and boundary_index < len(window):
            end = start + boundary_index
            window = text[start:end]

        chunk = window.strip()
        if not chunk:
            start = end
            continue

        chunks.append((chunk, start, start + len(chunk)))

        if end >= text_len:
            break

        next_start = end - overlap
        if next_start <= start:
            next_start = end
        start = next_start

    return chunks
