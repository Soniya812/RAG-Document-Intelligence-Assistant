from typing import List, Dict, Any

import chromadb


class VectorStore:
    def __init__(self, persist_dir: str = "./data/chroma"):
        self.persist_dir = persist_dir
        self.client = chromadb.PersistentClient(path=self.persist_dir)
        self.collection = None
        self.collection_name = None

    def create_collection(self, name: str = "documents"):
        self.collection_name = name
        if self.collection is None:
            self.collection = self.client.get_or_create_collection(name)
        elif getattr(self.collection, "name", None) != name:
            self.collection = self.client.get_or_create_collection(name)
        return self.collection

    def add_documents(
        self,
        ids: List[str],
        embeddings: List[List[float]],
        metadatas: List[Dict[str, Any]],
        documents: List[str],
    ):
        if self.collection is None:
            raise RuntimeError("Collection not created")
        if not ids or not embeddings or not documents:
            raise ValueError("ids, embeddings, and documents must not be empty")
        if len(ids) != len(embeddings) or len(ids) != len(documents):
            raise ValueError("ids, embeddings, and documents lengths must match")
        if metadatas is not None and len(metadatas) != len(ids):
            raise ValueError("metadatas length must match ids length")

        self.collection.add(
            ids=ids,
            embeddings=embeddings,
            metadatas=metadatas or [{} for _ in ids],
            documents=documents,
        )

    def query(self, embedding: List[float], n_results: int = 5):
        if self.collection is None:
            raise RuntimeError("Collection not created")
        if not embedding:
            raise ValueError("embedding must not be empty")
        if n_results <= 0:
            raise ValueError("n_results must be positive")

        results = self.collection.query(query_embeddings=[embedding], n_results=n_results)
        return results

    def persist(self):
        if hasattr(self.client, "persist"):
            self.client.persist()
