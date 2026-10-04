import re
from pathlib import Path
from typing import List

import fitz


def normalize_text(text: str) -> str:
    """Clean and normalize extracted PDF text for downstream chunking and retrieval."""
    if text is None:
        raise ValueError("Text input is required")

    cleaned = text.replace("\r\n", "\n").replace("\r", "\n")
    cleaned = cleaned.replace("\t", " ")
    cleaned = re.sub(r"[ \t]+", " ", cleaned)
    cleaned = re.sub(r"\n\s*\n+", "\n\n", cleaned)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned)
    cleaned = cleaned.replace(" \n", "\n").replace("\n ", "\n")
    cleaned = re.sub(r"\n{2,}", "\n\n", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned)
    cleaned = cleaned.strip()

    if not cleaned:
        raise ValueError("No text available after normalization")

    return cleaned


def load_pdf_text(path: str) -> str:
    """Load and normalize text from a PDF file using PyMuPDF (fitz).

    Returns the concatenated text of all pages.
    Raises ValueError for empty, unreadable, or non-PDF files.
    """
    file_path = Path(path)
    if not file_path.exists():
        raise ValueError(f"PDF file not found: {path}")
    if file_path.stat().st_size == 0:
        raise ValueError("PDF file is empty")

    try:
        doc = fitz.open(str(file_path))
    except (fitz.FileDataError, RuntimeError, ValueError) as exc:
        raise ValueError(f"Unreadable PDF file: {path}") from exc

    try:
        if doc.page_count == 0:
            raise ValueError("PDF has no pages")

        pages: List[str] = []
        for page in doc:
            text = page.get_text()
            if text:
                pages.append(text)

        full_text = "\n".join(pages).strip()
        if not full_text:
            raise ValueError("No extractable text found in PDF")

        return normalize_text(full_text)
    finally:
        doc.close()
