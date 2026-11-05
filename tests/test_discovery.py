from privatesearch.textutil import content_stems, stem

from privatesearch.llm import OllamaClient, LLMError
import pytest


def test_connections_are_cross_document_and_evidence_backed(loaded_engine):
    result = loaded_engine.discover("AI agents", link_threshold=0.15)
    assert result.passages and result.connections
    for edge in result.connections:
        a, b = result.node(edge.source), result.node(edge.target)
        assert a.chunk.doc_id != b.chunk.doc_id
        assert edge.weight >= 0.15
        ids, mat = loaded_engine.get_vectors([a.chunk.id, b.chunk.id])
        assert abs(float(mat[0] @ mat[1]) - edge.weight) < 1e-4
        for term in edge.shared_terms:  # shared terms must literally appear in BOTH passages
            for node in (a, b):
                assert all(stem(w) in content_stems(node.chunk.text) for w in term.split())


def test_no_connections_below_threshold(loaded_engine):
    result = loaded_engine.discover("AI agents", link_threshold=0.99)
    assert result.connections == []


def test_unrelated_documents_are_not_linked_to_agents(loaded_engine):
    result = loaded_engine.discover("AI agents", link_threshold=0.2)
    linked_docs = {result.node(e.source).chunk.doc_name for e in result.connections} | \
                  {result.node(e.target).chunk.doc_name for e in result.connections}
    assert "study_habits.txt" not in linked_docs


def test_concepts_span_multiple_documents(loaded_engine):
    result = loaded_engine.discover("AI agents", link_threshold=0.15)
    for concept in result.concepts:
        assert len(concept.doc_names) >= 2
        for chunk_id in concept.chunk_ids:
            text = loaded_engine.db.get_chunks([chunk_id])[chunk_id].text
            assert all(stem(w) in content_stems(text) for w in concept.term.split())


def test_empty_library_and_no_match(engine, loaded_engine):
    assert "No passages" in engine.discover("anything").message
    assert "No passages" in loaded_engine.discover("zxqv plorbnak").message


def test_remote_llm_is_blocked_by_default():
    with pytest.raises(LLMError):
        OllamaClient("http://example.com:11434", "llama3.2:3b")
    assert OllamaClient("http://localhost:11434", "llama3.2:3b").host == "http://localhost:11434"
    assert OllamaClient("http://127.0.0.1:11434", "m").host
