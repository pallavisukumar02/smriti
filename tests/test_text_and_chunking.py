from smriti.chunking import chunk_pages
from smriti.loaders import Page, load_file
from smriti.text import tokenize


def test_tokenize_drops_stopwords_and_stems():
    assert tokenize("The cats are jumping quickly") == ["cat", "jump", "quickly"]


def test_chunks_respect_size_and_overlap():
    para = "Sentence number {}. " * 1
    text = "\n\n".join(para.format(i) * 12 for i in range(20))
    chunks = chunk_pages([Page(text, 1)], "d1", "doc.txt", size=300, overlap=50)
    assert len(chunks) > 3
    assert all(len(c.text) <= 300 + 60 for c in chunks)
    assert all(c.page == 1 for c in chunks)
    assert [c.index for c in chunks] == list(range(len(chunks)))


def test_long_paragraph_is_split_by_sentence():
    text = " ".join(f"This is sentence {i}." for i in range(200))
    chunks = chunk_pages([Page(text)], "d", "long.txt", size=400, overlap=80)
    assert len(chunks) >= 8


def test_overlap_must_be_smaller_than_size():
    import pytest

    with pytest.raises(ValueError):
        chunk_pages([Page("x")], "d", "n", size=100, overlap=100)


def test_loaders_text_json_html():
    assert "hello" in load_file("a.txt", b"hello world")[0].text
    assert '"k": 1' in load_file("a.json", b'{"k":1}')[0].text
    html = load_file("a.html", b"<html><script>bad()</script><p>Visible text</p></html>")[0].text
    assert "Visible text" in html and "bad()" not in html
