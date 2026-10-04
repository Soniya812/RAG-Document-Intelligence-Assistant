from src.chunking import chunk_text


def test_chunking_basic():
    text = "a" * 1200
    chunks = chunk_text(text, chunk_size=500, overlap=50)
    assert len(chunks) >= 2
    first = chunks[0]
    second = chunks[1]
    assert first[2] - second[1] == 50


def test_chunking_small():
    text = "short text"
    chunks = chunk_text(text, chunk_size=1000)
    assert len(chunks) == 1
    assert chunks[0][0] == text


def test_chunking_empty_text_raises():
    try:
        chunk_text("   ")
        assert False, "Expected ValueError for empty text"
    except ValueError:
        assert True
