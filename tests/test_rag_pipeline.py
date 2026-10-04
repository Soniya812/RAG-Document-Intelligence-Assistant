import fitz
import pytest

from app import get_runtime_config
from src.embeddings import Embedder
from src.llm import LLM
from src.rag_pipeline import RAGPipeline
from src.vectorstore import VectorStore


class DummyLLM:
    def generate_answer(self, question, contexts, max_tokens=300):
        combined = " | ".join(contexts)
        return f"Answer based on: {combined}"


def test_rag_pipeline_integration_with_dummy_llm(tmp_path):
    pdf_path = tmp_path / "sample.pdf"
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 72), "This document explains how to train a model and evaluate it.")
    doc.save(pdf_path)
    doc.close()

    embedder = Embedder(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = VectorStore(persist_dir=str(tmp_path / "chroma"))
    llm = DummyLLM()

    pipeline = RAGPipeline(embedder=embedder, vectorstore=vectorstore, llm=llm)
    ingest_result = pipeline.ingest_document(str(pdf_path), collection_name="demo")

    assert ingest_result["count"] >= 1
    assert len(ingest_result["ids"]) == ingest_result["count"]

    answer_result = pipeline.answer_question("How is a model trained?", top_k=2)
    assert "answer" in answer_result
    assert "Answer based on:" in answer_result["answer"]
    assert len(answer_result["sources"]) >= 1
    assert answer_result["sources"][0]["document"]


def test_runtime_config_reports_missing_api_key(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("CHROMA_PERSIST_DIR", "./tmp/test-chroma")

    config, missing = get_runtime_config()
    assert config["CHROMA_PERSIST_DIR"] == "./tmp/test-chroma"
    assert missing == ["OPENAI_API_KEY"]


def test_llm_requires_api_key(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(ValueError, match="OPENAI_API_KEY not provided"):
        LLM()


def test_llm_handles_empty_contexts(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-api-key")
    llm = LLM(api_key="test-api-key")
    answer = llm.generate_answer("What is this?", [])
    assert answer == "I don't know based on the provided context."


def test_rag_pipeline_invalid_pdf_raises(tmp_path):
    bad_pdf = tmp_path / "broken.pdf"
    bad_pdf.write_bytes(b"not a valid pdf")

    pipeline = RAGPipeline(
        embedder=Embedder(model_name="sentence-transformers/all-MiniLM-L6-v2"),
        vectorstore=VectorStore(persist_dir=str(tmp_path / "chroma")),
        llm=DummyLLM(),
    )

    with pytest.raises(ValueError):
        pipeline.ingest_document(str(bad_pdf), collection_name="demo")


def test_rag_pipeline_answers_without_context_are_handled(tmp_path):
    pipeline = RAGPipeline(
        embedder=Embedder(model_name="sentence-transformers/all-MiniLM-L6-v2"),
        vectorstore=VectorStore(persist_dir=str(tmp_path / "chroma")),
        llm=DummyLLM(),
    )
    pipeline.vectorstore.create_collection("empty-demo")

    answer_result = pipeline.answer_question("What is this about?", top_k=2)
    assert "answer" in answer_result
    assert "Answer based on:" in answer_result["answer"]
    assert answer_result["sources"] == []