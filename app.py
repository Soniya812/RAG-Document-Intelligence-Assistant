import os
from pathlib import Path
import uuid

import streamlit as st
from dotenv import load_dotenv

from src.embeddings import Embedder
from src.llm import LLM
from src.rag_pipeline import RAGPipeline
from src.vectorstore import VectorStore


load_dotenv()


def get_runtime_config():
    config = {
        "OPENAI_API_KEY": (os.getenv("OPENAI_API_KEY") or "").strip(),
        "CHROMA_PERSIST_DIR": os.getenv("CHROMA_PERSIST_DIR", "./data/chroma"),
    }
    missing = [key for key, value in config.items() if key == "OPENAI_API_KEY" and not value]
    return config, missing


def build_collection_name(file_name: str) -> str:
    stem = Path(file_name).stem
    safe_name = "".join(ch if ch.isalnum() else "_" for ch in stem).strip("_")
    return safe_name[:40] if safe_name else "document"


def validate_uploaded_pdf(uploaded_file) -> None:
    if uploaded_file is None:
        raise ValueError("Please upload a PDF file.")
    if uploaded_file.size is None or uploaded_file.size <= 0:
        raise ValueError("The uploaded PDF is empty.")
    if uploaded_file.type and uploaded_file.type.lower() != "application/pdf":
        raise ValueError("Only PDF files are supported.")
    if not uploaded_file.name.lower().endswith(".pdf"):
        raise ValueError("The selected file must have a .pdf extension.")


def save_uploaded_pdf(uploaded_file) -> str:
    validate_uploaded_pdf(uploaded_file)

    upload_dir = Path("./data/uploads")
    upload_dir.mkdir(parents=True, exist_ok=True)

    original_name = Path(uploaded_file.name).name
    unique_name = original_name
    if (upload_dir / unique_name).exists():
        unique_name = f"{uuid.uuid4().hex}_{original_name}"

    save_path = upload_dir / unique_name
    with save_path.open("wb") as file_obj:
        file_obj.write(uploaded_file.getbuffer())

    return str(save_path)


def build_rag_pipeline(persist_dir: str, llm_enabled: bool = False):
    embedder = Embedder()
    vectorstore = VectorStore(persist_dir=persist_dir)

    if llm_enabled:
        llm = LLM()
    else:
        llm = None

    return RAGPipeline(embedder=embedder, vectorstore=vectorstore, llm=llm)


def main():
    st.title("RAG Document Intelligence Assistant - V1")

    config, missing_config = get_runtime_config()
    persist_dir = st.sidebar.text_input(
        "Chroma persist dir",
        value=config["CHROMA_PERSIST_DIR"],
    )

    st.sidebar.header("Configuration")
    st.sidebar.caption("Set OPENAI_API_KEY in your environment or .env file before asking questions.")
    if missing_config:
        st.sidebar.warning(
            "OPENAI_API_KEY is not configured. Add it to your .env file or environment before asking questions."
        )

    st.subheader("1. Upload a PDF")
    uploaded_file = st.file_uploader("Choose a PDF document", type=["pdf"], key="pdf_uploader")

    if uploaded_file is not None:
        try:
            file_path = save_uploaded_pdf(uploaded_file)
            previous_path = st.session_state.get("uploaded_pdf_path")
            if previous_path and Path(previous_path).name != uploaded_file.name:
                st.session_state["document_indexed"] = False
                st.session_state.pop("indexed_document_path", None)

            st.session_state["uploaded_pdf_path"] = file_path
            st.session_state["active_collection"] = build_collection_name(uploaded_file.name)
            st.session_state["upload_status"] = "success"
            st.success(f"Uploaded successfully: {Path(file_path).name}")
            st.caption(f"Saved to: {file_path}")
            st.caption(f"Active collection: {st.session_state['active_collection']}")
        except ValueError as exc:
            st.session_state["upload_status"] = "error"
            st.error(str(exc))
            st.session_state.pop("uploaded_pdf_path", None)
            st.session_state["document_indexed"] = False

    uploaded_pdf_path = st.session_state.get("uploaded_pdf_path")
    if uploaded_pdf_path:
        if st.button("Process document and build vector index", key="ingest_button"):
            collection_name = st.session_state.get("active_collection", "documents")
            try:
                with st.spinner("Extracting text, chunking, embedding, and indexing the document..."):
                    rag = build_rag_pipeline(persist_dir=persist_dir, llm_enabled=False)
                    result = rag.ingest_document(uploaded_pdf_path, collection_name=collection_name)
                st.session_state["document_indexed"] = True
                st.session_state["indexed_document_path"] = uploaded_pdf_path
                st.success(f"Document indexed successfully. {result['count']} chunks added to the vector store in collection '{collection_name}'.")
            except Exception as exc:
                st.session_state["document_indexed"] = False
                st.session_state["last_error"] = str(exc)
                st.error(f"Document processing failed: {exc}")

    st.subheader("2. Ask a question")
    if st.session_state.get("indexed_document_path"):
        st.caption(f"Current indexed document: {Path(st.session_state['indexed_document_path']).name}")

    question = st.text_input("Ask a question about the uploaded document", key="question_input")
    can_answer = bool(st.session_state.get("document_indexed")) and bool(question) and not missing_config

    if missing_config and st.session_state.get("document_indexed") and question:
        st.info("Add OPENAI_API_KEY to your environment or .env file to enable grounded answer generation.")

    if can_answer and st.button("Get Answer", key="answer_button"):
        collection_name = st.session_state.get("active_collection", "documents")
        try:
            with st.spinner("Retrieving relevant context and generating a grounded answer..."):
                rag = build_rag_pipeline(persist_dir=persist_dir, llm_enabled=True)
                rag.vectorstore.create_collection(collection_name)
                result = rag.answer_question(question, top_k=3, collection_name=collection_name)

            st.subheader("Answer")
            st.write(result["answer"])

            st.subheader("Sources used")
            if result.get("sources"):
                for index, source in enumerate(result["sources"], start=1):
                    st.markdown(f"**Chunk {index}**")
                    st.write(source.get("document", ""))
                    if source.get("id"):
                        st.caption(f"Chunk ID: {source['id']}")
                    if source.get("metadata"):
                        st.caption(f"Metadata: {source['metadata']}")
                    if source.get("distance") is not None:
                        st.caption(f"Distance: {source['distance']}")
            else:
                st.info("No relevant source chunks were found for this question.")
        except Exception as exc:
            st.session_state["last_error"] = str(exc)
            st.error(f"Answer generation failed: {exc}")

    if not st.session_state.get("document_indexed") and uploaded_pdf_path:
        st.info("Upload a PDF and process it to build the vector index before asking questions.")


if __name__ == "__main__":
    main()
