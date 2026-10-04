from src.embeddings import Embedder


class FakeSentenceTransformer:
    def __init__(self, model_name):
        self.model_name = model_name

    def encode(self, texts, convert_to_numpy=True, show_progress_bar=False):
        result = []
        for text in texts:
            value = float(len(str(text)))
            result.append([value, value + 1.0])
        return result


def test_embedder_uses_configured_model(monkeypatch):
    monkeypatch.setattr("src.embeddings.SentenceTransformer", FakeSentenceTransformer)

    embedder = Embedder(model_name="demo-model")
    assert embedder.model_name == "demo-model"

    vectors = embedder.embed_texts(["short", "medium length"])
    assert len(vectors) == 2
    assert vectors[0][0] == 5.0
    assert vectors[1][0] == 13.0

    query_vector = embedder.embed_query("sample")
    assert len(query_vector) == 2
    assert query_vector[0] == 6.0


def test_embedder_empty_input_raises():
    monkeypatch = None
    # The wrapper should reject empty inputs before calling the model.
    class SafeSentenceTransformer:
        def __init__(self, model_name):
            self.model_name = model_name

        def encode(self, texts, convert_to_numpy=True, show_progress_bar=False):
            return []

    import src.embeddings as embeddings_module
    original = embeddings_module.SentenceTransformer
    embeddings_module.SentenceTransformer = SafeSentenceTransformer
    try:
        embedder = Embedder(model_name="demo-model")
        try:
            embedder.embed_texts([])
            assert False, "Expected ValueError for empty text list"
        except ValueError:
            assert True

        try:
            embedder.embed_query("   ")
            assert False, "Expected ValueError for blank query text"
        except ValueError:
            assert True
    finally:
        embeddings_module.SentenceTransformer = original
