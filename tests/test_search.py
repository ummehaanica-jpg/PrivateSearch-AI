import pymupdf
import pytest

from privatesearch.hybrid import build_fts_query, rrf_fuse


def test_duplicate_prevention(engine):
    data = b"Cats are small carnivorous mammals kept as pets by many families."
    first = engine.ingest("cats.txt", data)
    second = engine.ingest("renamed.txt", data)
    assert first.status == "added" and second.status == "duplicate"
    assert len(engine.list_documents()) == 1


def test_persistence_across_restart(settings, loaded_engine):
    from privatesearch.embeddings import HashingEmbedder
    from privatesearch.engine import PrivateSearchEngine

    before = loaded_engine.search("vector similarity search", "hybrid", 3)
    restarted = PrivateSearchEngine(settings, embedder=HashingEmbedder())
    after = restarted.search("vector similarity search", "hybrid", 3)
    assert len(restarted.list_documents()) == len(loaded_engine.list_documents())
    assert [r.chunk_id for r in before] == [r.chunk_id for r in after]


def test_keyword_semantic_hybrid_find_the_right_document(loaded_engine):
    for mode in ("keyword", "semantic", "hybrid"):
        results = loaded_engine.search("FAISS flat index cosine", mode, 3)
        assert results and results[0].chunk.doc_name == "vector_databases.docx", mode


def test_special_characters_do_not_break_fts(loaded_engine):
    for nasty in ['AND OR "', "agent*", "(((", "a:b -c", "'; DROP TABLE chunks; --"]:
        loaded_engine.search(nasty, "hybrid", 3)  # must not raise
    assert loaded_engine.stats()["chunks"] > 0


def test_empty_and_long_queries_are_rejected(loaded_engine):
    with pytest.raises(ValueError):
        loaded_engine.search("   ")
    with pytest.raises(ValueError):
        loaded_engine.search("x" * 501)


def test_delete_removes_every_trace(loaded_engine):
    doc = next(d for d in loaded_engine.list_documents() if d.name == "study_habits.txt")
    assert loaded_engine.search("spaced repetition", "keyword", 3)
    stored = loaded_engine.settings.uploads_dir / doc.stored_name
    assert stored.exists()
    chunks_before = len(loaded_engine.index)
    assert loaded_engine.delete_document(doc.id)
    assert not stored.exists()
    assert len(loaded_engine.index) < chunks_before
    assert not loaded_engine.search("spaced repetition", "keyword", 3)
    assert not any(r.chunk.doc_name == "study_habits.txt"
                   for r in loaded_engine.search("study", "semantic", 20))
    with loaded_engine.db.connect() as conn:
        assert conn.execute("SELECT COUNT(*) FROM embeddings WHERE chunk_id NOT IN (SELECT id FROM chunks)").fetchone()[0] == 0
        assert conn.execute("SELECT COUNT(*) FROM chunks_fts WHERE chunks_fts MATCH 'repetition'").fetchone()[0] == 0


def test_rebuild_index_keeps_search_working(loaded_engine):
    before = {d.name: d.n_chunks for d in loaded_engine.list_documents()}
    report = loaded_engine.rebuild_index()
    assert report["documents"] == len(before)
    assert {d.name: d.n_chunks for d in loaded_engine.list_documents()} == before
    assert loaded_engine.search("agent orchestration supervisor", "hybrid", 3)


def test_history_and_clear(loaded_engine):
    loaded_engine.search("alpha beta", "hybrid")
    loaded_engine.search("alpha beta", "hybrid")
    assert loaded_engine.history() == [("alpha beta", "hybrid")]
    loaded_engine.clear_history()
    assert loaded_engine.history() == []


def test_embedding_model_change_marks_index_stale(settings, loaded_engine):
    from privatesearch.embeddings import HashingEmbedder
    from privatesearch.engine import PrivateSearchEngine

    other = HashingEmbedder()
    other.name = "some-other-model"
    engine = PrivateSearchEngine(settings, embedder=other)
    assert engine.index_stale
    assert engine.ingest("new.txt", b"fresh text about something new and different").status == "error"
    assert engine.search("agents", "keyword", 3)  # keyword search still works
    engine.rebuild_index()
    assert not engine.index_stale


def test_rrf_prefers_items_ranked_by_both():
    fused = rrf_fuse({"k": [1, 2, 3], "s": [3, 1, 4]}, {"k": 0.5, "s": 0.5})
    assert max(fused, key=fused.get) in (1, 3) and fused[1] > fused[2] and fused[3] > fused[4]


def test_fts_query_is_quoted():
    assert build_fts_query('the "AI" agents (and) tools') == '"ai" OR "agents" OR "tools"'
