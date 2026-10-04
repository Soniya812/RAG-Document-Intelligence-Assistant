# RAG Document Intelligence Assistant (V1)

A minimal Retrieval-Augmented Generation (RAG) application for locally querying PDFs.

Overview
--------
This project is a Streamlit-based app that lets you upload a PDF, extract text, chunk it, embed the chunks locally with SentenceTransformers, store vectors in a local ChromaDB instance, and answer questions with the retrieved context using the OpenAI API.

Architecture
------------
- `app.py`: Streamlit UI and orchestration.
- `src/pdf_loader.py`: PDF text extraction using PyMuPDF.
- `src/chunking.py`: chunking logic with overlap.
- `src/embeddings.py`: local SentenceTransformers embeddings wrapper.
- `src/vectorstore.py`: local ChromaDB wrapper for persistence and queries.
- `src/retriever.py`: retrieval logic for top-k chunk results.
- `src/llm.py`: OpenAI-backed answer generation wrapper.
- `src/rag_pipeline.py`: end-to-end pipeline orchestrator.

Environment and configuration
-----------------------------
The app expects these environment variables:

- `OPENAI_API_KEY`: required for answer generation. If it is missing, the Streamlit app shows a configuration warning and disables the answer button until it is set.
- `CHROMA_PERSIST_DIR`: optional override for the local Chroma persistence directory. Defaults to `./data/chroma` when not set.

The current implementation does not read `EMBEDDING_MODEL` from the environment. Use the values in `.env.example` only for the supported keys above.

Quickstart
----------
1. Create a Python virtual environment and activate it.

```bash
python -m venv .venv
.\.venv\Scripts\activate
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Copy `.env.example` to `.env` and set a valid `OPENAI_API_KEY`.

```bash
copy .env.example .env
```

4. Run the Streamlit app:

```bash
streamlit run app.py
```

5. Upload a PDF, process it to build the index, and then ask a question.

Expected behavior and failure modes
------------------------------------
- Missing or blank `OPENAI_API_KEY` prevents grounded answer generation and shows a clear configuration warning in the UI.
- Invalid or unreadable PDFs raise a `ValueError` during upload/ingestion and are shown as user-facing errors in the app.
- Empty or no-context retrieval results are handled gracefully and the LLM layer returns a fallback answer if no context is available.
- The app does not perform OCR; scanned PDFs without embedded text will be treated as unreadable by the current PDF loader.

Testing
-------
Run the project tests with the configured project virtual environment:

```bash
.\.venv\Scripts\python.exe -m pytest -q
```

Troubleshooting
---------------
- If the app reports `OPENAI_API_KEY not provided`, add the key to `.env` and restart the app.
- If the app says a PDF is invalid or unreadable, verify the file is a real PDF and contains extractable text.
- If you change the Chroma persist directory, ensure the path is writable and that the app is pointed at the intended storage location.

Notes & Next Steps
------------------
- This version uses local embeddings and ChromaDB for simplicity and offline capability.
- Future improvements: better chunking and retrieval quality, enhanced UI state management, caching, and optional remote embeddings/LLM configurations.
