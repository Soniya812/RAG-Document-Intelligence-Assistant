from src.retriever import Retriever


class FakeVectorStore:
    def __init__(self, results):
        self.results = results

    def query(self, embedding, n_results=5):
        assert embedding == [1.0, 0.0]
        if n_results == 2:
            return {
                "ids": [["chunk-3", "chunk-1"]],
                "distances": [[0.25, 0.50]],
                "metadatas": [[{"page": 3}, {"page": 1}]],
                "documents": [["Third chunk text", "First chunk text"]],
            }
        return self.results


def test_retriever_successful_retrieval():
    fake_store = FakeVectorStore(
        {
            "ids": [["chunk-1", "chunk-2"]],
            "distances": [[0.11, 0.72]],
            "metadatas": [[{"page": 1}, {"page": 2}]],
            "documents": [["First chunk text", "Second chunk text"]],
        }
    )
    retriever = Retriever(fake_store)

    results = retriever.retrieve([1.0, 0.0], top_k=2)
    assert len(results) == 2
    assert results[0]["id"] == "chunk-3"
    assert results[0]["document"] == "Third chunk text"
    assert results[0]["metadata"]["page"] == 3
    assert results[0]["distance"] == 0.25


def test_retriever_top_k_behavior():
    fake_store = FakeVectorStore({
        "ids": [["chunk-3", "chunk-1"]],
        "distances": [[0.25, 0.50]],
        "metadatas": [[{"page": 3}, {"page": 1}]],
        "documents": [["Third chunk text", "First chunk text"]],
    })
    retriever = Retriever(fake_store)

    results = retriever.retrieve([1.0, 0.0], top_k=2)
    assert len(results) == 2
    assert [item["id"] for item in results] == ["chunk-3", "chunk-1"]


def test_retriever_empty_query_raises():
    retriever = Retriever(FakeVectorStore({}))
    try:
        retriever.retrieve([], top_k=2)
        assert False, "Expected ValueError for empty query embedding"
    except ValueError:
        assert True
