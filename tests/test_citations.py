import pymupdf

from privatesearch.qa import NOT_FOUND_TEXT, grounding_score, verify_citations
from privatesearch.textutil import normalize_space

from conftest import FakeLLM


def make_pdf(tmp_path):
    pdf = pymupdf.open()
    pdf.new_page().insert_text((72, 100), "Tomatoes need full sun and regular watering in summer.")
    pdf.new_page().insert_text((72, 100), "Quantum entanglement links two photons so that measuring one affects the other.")
    path = tmp_path / "mixed.pdf"
    pdf.save(path)
    return path


def test_page_number_and_passage_text_are_accurate(engine, tmp_path):
    path = make_pdf(tmp_path)
    assert engine.ingest_path(path).status == "added"
    result = engine.search("entanglement photons", "hybrid", 3)[0]
    assert result.chunk.doc_name == "mixed.pdf" and result.chunk.page == 2
    original_page2 = normalize_space(pymupdf.open(path)[1].get_text())
    assert normalize_space(result.chunk.text) in original_page2
    tomato = engine.search("tomatoes watering", "keyword", 3)[0]
    assert tomato.chunk.page == 1


def test_every_result_text_comes_from_its_cited_document(loaded_engine):
    stored = {d.name: d for d in loaded_engine.list_documents()}
    for query in ("agent orchestration", "overfitting regularisation", "disk encryption"):
        for r in loaded_engine.search(query, "hybrid", 5):
            assert r.chunk.doc_name in stored
            row = loaded_engine.db.get_chunks([r.chunk_id])[r.chunk_id]
            assert row.text == r.chunk.text and row.doc_id == stored[r.chunk.doc_name].id


def test_llm_answer_citations_map_to_sources(loaded_engine):
    llm = FakeLLM("Supervisors route subtasks to specialist workers [1].")
    loaded_engine._llm = llm
    answer = loaded_engine.ask("What does a supervisor agent do?")
    assert answer.mode == "llm" and answer.sources
    number, result = answer.sources[0]
    assert number == 1 and f"[{number}]" in answer.text
    assert f"[1] (file: {result.chunk.doc_name}" in llm.prompts[0]
    assert result.chunk.text in llm.prompts[0]


def test_invented_citation_numbers_are_removed(loaded_engine):
    loaded_engine._llm = FakeLLM("Supervisors route work [1] and agents never fail [9].")
    answer = loaded_engine.ask("What does a supervisor agent do?")
    assert "[9]" not in answer.text and any("do not exist" in w for w in answer.warnings)


def test_missing_citations_are_flagged(loaded_engine):
    loaded_engine._llm = FakeLLM("Supervisors route work to specialists.")
    answer = loaded_engine.ask("What does a supervisor agent do?")
    assert any("did not cite" in w for w in answer.warnings)


def test_unanswerable_question_says_so(loaded_engine):
    loaded_engine._llm = FakeLLM("should never be asked")
    answer = loaded_engine.ask("zxqv plorbnak glimmerwhistle")
    assert answer.mode == "not_found" and answer.text == NOT_FOUND_TEXT and not answer.sources
    loaded_engine._llm = FakeLLM("NOT_FOUND")
    assert loaded_engine.ask("What does a supervisor agent do?").mode == "not_found"


def test_extractive_fallback_quotes_documents_verbatim(loaded_engine):
    answer = loaded_engine.ask("How does hybrid search work?", use_llm=False)
    assert answer.mode == "extractive" and answer.sources
    for number, result in answer.sources:
        assert f"[{number}]" in answer.text
    body = normalize_space(answer.text.replace("[1]", "").replace("[2]", "").replace("[3]", ""))
    corpus = " ".join(normalize_space(r.chunk.text) for _, r in answer.sources)
    assert all(piece.strip() in corpus for piece in body.split(". ")[:1])


def test_llm_failure_falls_back_to_extractive(loaded_engine):
    from privatesearch.llm import LLMError

    class Broken(FakeLLM):
        def generate(self, system, prompt):
            raise LLMError("Could not reach Ollama.")

    loaded_engine._llm = Broken("")
    answer = loaded_engine.ask("How does hybrid search work?")
    assert answer.mode == "extractive" and answer.warnings


def test_verify_and_grounding_helpers():
    text, valid, invalid = verify_citations("A [2] b [2] c [7]", 3)
    assert valid == [2] and invalid == [7] and "[7]" not in text
    assert grounding_score("Hybrid search combines keyword matching [1]", ["Hybrid search combines keyword matching with vectors."]) == 1.0
    assert grounding_score("Bananas orbit Jupiter nightly", ["Hybrid search combines keyword matching."]) < 0.3
