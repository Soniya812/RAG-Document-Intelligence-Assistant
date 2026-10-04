import pytest

from src.vectorstore import VectorStore


def test_vectorstore_add_and_query(tmp_path):
    persist_dir = str(tmp_path / "chroma")
    store = VectorStore(persist_dir=persist_dir)
    store.create_collection(name="demo")

    documents = [
        "Alpha beta gamma",
        "Delta epsilon zeta",
    ]
    embeddings = [
        [1.0, 0.0],
        [0.0, 1.0],
    ]
    metadatas = [
        {"page": 1},
        {"page": 2},
    ]

    store.add_documents(
        ids=["doc-1", "doc-2"],
        embeddings=embeddings,
        metadatas=metadatas,
        documents=documents,
    )

    results = store.query([1.0, 0.0], n_results=2)
    assert "ids" in results
    assert len(results["ids"][0]) >= 1
    assert results["documents"][0][0] == "Alpha beta gamma"


def test_vectorstore_validates_empty_inputs(tmp_path):
    store = VectorStore(persist_dir=str(tmp_path / "chroma"))
    store.create_collection(name="demo")

    try:
        store.add_documents([], [], [], [])
        assert False, "Expected ValueError for empty vectors"
    except ValueError:
        assert True

    try:
        store.query([], n_results=2)
        assert False, "Expected ValueError for empty query embedding"
    except ValueError:
        assert True


def test_vectorstore_tracks_active_collection_name(tmp_path):
    store = VectorStore(persist_dir=str(tmp_path / "chroma"))
    store.create_collection(name="demo")
    assert store.collection_name == "demo"

    store.create_collection(name="updated-demo")
    assert store.collection_name == "updated-demo"


def test_vectorstore_persists_collection_state(tmp_path):
    persist_dir = str(tmp_path / "persisted-chroma")

    first_store = VectorStore(persist_dir=persist_dir)
    first_store.create_collection(name="persisted-demo")
    first_store.add_documents(
        ids=["doc-1"],
        embeddings=[[1.0, 0.0]],
        metadatas=[{"page": 1}],
        documents=["Alpha beta gamma"],
    )
    first_store.persist()

    reopened_store = VectorStore(persist_dir=persist_dir)
    reopened_store.create_collection(name="persisted-demo")
    results = reopened_store.query([1.0, 0.0], n_results=1)

    assert reopened_store.collection_name == "persisted-demo"
    assert results["documents"][0][0] == "Alpha beta gamma"
