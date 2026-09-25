from smriti.bm25 import BM25
from smriti.fusion import reciprocal_rank_fusion
from smriti.loaders import Page
from smriti.pipeline import Smriti
from smriti.retriever import HybridRetriever
from smriti.chunking import chunk_pages


def test_bm25_ranks_matching_doc_first():
    bm = BM25().fit(["the cat sat on the mat", "dogs chase balls in parks", "a cat and a dog"])
    top = bm.top_k("cat mat", 3)
    assert top[0][0] == 0
    assert all(s > 0 for _, s in top)


def test_rrf_rewards_agreement():
    fused = reciprocal_rank_fusion([[1, 2, 3], [3, 1, 4]])
    ids = [d for d, _ in fused]
    assert ids[0] == 1  # ranked high by both lists
    assert set(ids) == {1, 2, 3, 4}


def test_hybrid_retriever_add_search_remove(embedder):
    r = HybridRetriever(embedder)
    a = chunk_pages([Page("Invoices are due within 30 days of receipt.", 1)], "a", "billing.pdf")
    b = chunk_pages([Page("The office cafeteria serves lunch at noon.", 1)], "b", "office.txt")
    r.add(a + b)
    hits = r.search("when are invoices due", k=2)
    assert hits[0].chunk.doc_id == "a"
    assert hits[0].sparse_rank == 1
    r.remove_doc("a")
    assert len(r) == 1 and len(r.store) == 1
    assert all(h.chunk.doc_id == "b" for h in r.search("invoices", k=2))


def test_pipeline_end_to_end_without_llm(embedder):
    rag = Smriti(embedder, chunk_size=300, overlap=50)
    doc = rag.add_file("notes.txt", b"Project Falcon launches on 12 March.\n\nBudget is 40 lakh rupees.")
    assert doc.n_chunks >= 1
    hits = rag.retrieve("When does Falcon launch?", k=3)
    assert "Falcon" in hits[0].chunk.text
    msgs = rag.build_messages("When does Falcon launch?", hits, [])
    assert msgs[0]["role"] == "system" and "[1]" in msgs[-1]["content"]
    # same file again is a no-op; changed file with same name replaces it
    rag.add_file("notes.txt", b"Project Falcon launches on 12 March.\n\nBudget is 40 lakh rupees.")
    rag.add_file("notes.txt", b"Falcon was delayed to May.")
    assert len(rag.documents) == 1
    rag.reindex(500, 100)
    assert len(rag.retriever) >= 1
