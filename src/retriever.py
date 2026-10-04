from typing import Any, Dict, List


class Retriever:
    def __init__(self, vectorstore):
        self.vectorstore = vectorstore

    def retrieve(self, query_embedding: List[float], top_k: int = 5) -> List[Dict[str, Any]]:
        if not query_embedding:
            raise ValueError("query_embedding must not be empty")
        if top_k <= 0:
            raise ValueError("top_k must be positive")

        result = self.vectorstore.query(query_embedding, n_results=top_k)
        docs: List[Dict[str, Any]] = []

        ids = result.get("ids", [])
        distances = result.get("distances", [])
        metadatas = result.get("metadatas", [])
        documents = result.get("documents", [])

        for i in range(len(ids)):
            for j in range(len(ids[i])):
                doc_id = ids[i][j]
                docs.append(
                    {
                        "id": doc_id,
                        "distance": distances[i][j] if i < len(distances) and j < len(distances[i]) else None,
                        "metadata": metadatas[i][j] if i < len(metadatas) and j < len(metadatas[i]) else None,
                        "document": documents[i][j] if i < len(documents) and j < len(documents[i]) else None,
                    }
                )

        return docs
