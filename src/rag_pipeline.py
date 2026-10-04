import uuid
from typing import Any, Dict

from .chunking import chunk_text
from .embeddings import Embedder
from .llm import LLM
from .pdf_loader import load_pdf_text
from .retriever import Retriever
from .vectorstore import VectorStore


class RAGPipeline:
    def __init__(self, embedder: Embedder = None, vectorstore: VectorStore = None, llm: LLM = None):
        self.embedder = embedder or Embedder()
        self.vectorstore = vectorstore or VectorStore()
        if llm is None:
            try:
                self.llm = LLM()
            except ValueError:
                self.llm = None
        else:
            self.llm = llm

    def ingest_document(self, path: str, collection_name: str = "documents") -> Dict[str, Any]:
        text = load_pdf_text(path)
        chunks = chunk_text(text)
        texts = [chunk[0] for chunk in chunks]
        ids = [str(uuid.uuid4()) for _ in texts]
        metadatas = [{"start": chunk[1], "end": chunk[2]} for chunk in chunks]

        embeddings = self.embedder.embed_texts(texts)
        self.vectorstore.create_collection(collection_name)
        self.vectorstore.add_documents(
            ids=ids,
            embeddings=embeddings,
            metadatas=metadatas,
            documents=texts,
        )
        self.vectorstore.persist()
        return {"count": len(texts), "ids": ids}

    def answer_question(self, question: str, top_k: int = 5, collection_name: str = None) -> Dict[str, Any]:
        if self.llm is None:
            raise RuntimeError("LLM not configured")

        target_collection = collection_name or self.vectorstore.collection_name or "documents"

        if self.vectorstore.collection is None:
            self.vectorstore.create_collection(target_collection)
        elif self.vectorstore.collection_name != target_collection:
            self.vectorstore.create_collection(target_collection)

        q_emb = self.embedder.embed_query(question)
        retriever = Retriever(self.vectorstore)
        docs = retriever.retrieve(q_emb, top_k=top_k)
        contexts = [doc["document"] for doc in docs if doc.get("document")]
        answer = self.llm.generate_answer(question, contexts)
        return {"answer": answer, "sources": docs}
