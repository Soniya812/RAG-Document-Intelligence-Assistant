import os
from typing import List

from sentence_transformers import SentenceTransformer


class Embedder:
    def __init__(self, model_name: str = None):
        self.model_name = model_name or os.getenv(
            "EMBEDDING_MODEL",
            "sentence-transformers/all-MiniLM-L6-v2",
        )
        self.model = SentenceTransformer(self.model_name)

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            raise ValueError("texts must not be empty")

        cleaned = [str(text).strip() for text in texts]
        cleaned = [text for text in cleaned if text]
        if not cleaned:
            raise ValueError("texts must contain at least one non-empty value")

        arr = self.model.encode(cleaned, convert_to_numpy=True, show_progress_bar=False)
        return [list(map(float, vector)) for vector in arr]

    def embed_query(self, text: str) -> List[float]:
        if text is None or not str(text).strip():
            raise ValueError("query text must not be empty")

        vec = self.model.encode([str(text).strip()], convert_to_numpy=True, show_progress_bar=False)[0]
        return list(map(float, vec))
