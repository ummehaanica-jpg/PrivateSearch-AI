"""Streamlit pages. All logic lives in `privatesearch/`; these functions only collect input and render output."""
from __future__ import annotations

from pathlib import Path

import streamlit as st

from privatesearch.config import ALLOWED_EXTENSIONS, PROJECT_ROOT
from privatesearch.engine import PrivateSearchEngine
from privatesearch.llm import OllamaClient
from privatesearch.textutil import content_stems, tokenize
from ui.components import chip, empty_state, esc, highlight, page_head, passage_block, passage_card, stat
from ui.graph import build_figure

PAGES = ["Search", "Ask", "Discover", "Documents", "Privacy & settings"]
SAMPLE_DIR = PROJECT_ROOT / "sample_docs"
MODE_LABELS = {"Hybrid": "hybrid", "Keywords": "keyword", "Meaning": "semantic"}
MODE_ICONS = {"hybrid": "hybrid", "keyword": "keywords", "semantic": "meaning", "ask": "question", "discover": "connections"}


def goto(page: str) -> None:
    st.session_state["page"] = page


def _terms(text: str) -> list[str]:
    return [t for t in tokenize(text) if len(t) > 2]


def _no_documents(engine: PrivateSearchEngine, key: str) -> bool:
    """Show the first-run empty state. Returns True when the library is empty."""
    if engine.stats()["documents"] > 0:
        return False
    st.markdown(empty_state("Your library is empty",
                            "Add PDF, Word, text or Markdown files and they will be searchable in seconds. "
                            "Nothing is uploaded anywhere: files are copied into this app's local data folder."),
                unsafe_allow_html=True)
    c1, c2, _ = st.columns([1, 1.3, 2])
    c1.button("Add documents", key=f"{key}_add", on_click=goto, args=("Documents",), type="primary")
    if c2.button("Try the sample documents", key=f"{key}_sample"):
        load_samples(engine)
        st.rerun()
    return True


def load_samples(engine: PrivateSearchEngine) -> None:
    files = sorted(p for p in SAMPLE_DIR.glob("*") if p.suffix.lower() in ALLOWED_EXTENSIONS)
    if not files:
        st.error("The sample_docs folder is missing or empty.")
        return
    with st.spinner("Indexing sample documents…"):
        for path in files:
            engine.ingest_path(path)


def _index_banner(engine: PrivateSearchEngine) -> None:
    if engine.index_stale:
        st.warning("Your index was built with a different embedding model, so meaning-based search is paused. "
                   "Open Documents and choose Rebuild index.")
        st.button("Go to Documents", key="stale_goto", on_click=goto, args=("Documents",))


# ----------------------------------------------------------------------------- sidebar
def render_sidebar(engine: PrivateSearchEngine) -> None:
    from ui.theme import BRAND

    with st.sidebar:
        st.markdown(BRAND, unsafe_allow_html=True)
        st.radio("Navigate", PAGES, key="page", label_visibility="collapsed")
        stats = engine.stats()
        st.markdown('<div class="ps-side-title">Library</div>', unsafe_allow_html=True)
        st.caption(f"{stats['documents']} documents, {stats['chunks']} passages")
        for doc in engine.list_documents()[:4]:
            st.caption(f"• {doc.name}")
        if stats["documents"] > 4:
            st.caption(f"and {stats['documents'] - 4} more")
        st.button("Manage documents", key="side_docs", on_click=goto, args=("Documents",))

        st.markdown('<div class="ps-side-title">Recent searches</div>', unsafe_allow_html=True)
        history = engine.history(6)
        if not history:
            st.caption("Searches you run will appear here.")
        for i, (query, mode) in enumerate(history):
            label = query if len(query) <= 30 else query[:28] + "…"
            st.button(label, key=f"hist_{i}", help=f"Run again ({MODE_ICONS.get(mode, mode)})",
                      on_click=_rerun_history, args=(query, mode))
        if history:
            st.button("Clear history", key="side_clear", on_click=engine.clear_history)

        st.markdown('<div class="ps-side-title">Privacy</div>', unsafe_allow_html=True)
        st.markdown(chip("Everything stays on this computer", "good") + "<br>" + chip("No telemetry"),
                    unsafe_allow_html=True)
        if engine.embedder.fallback_reason:
            st.warning("Using the offline fallback model. Results are word-based, not semantic. "
                       "See Privacy & settings to fix this.")


def _rerun_history(query: str, mode: str) -> None:
    target = {"ask": "Ask", "discover": "Discover"}.get(mode, "Search")
    st.session_state["page"] = target
    st.session_state[{"Ask": "ask_text", "Discover": "disc_text"}.get(target, "search_text")] = query
    if mode in MODE_LABELS.values():
        st.session_state["search_mode_label"] = next(k for k, v in MODE_LABELS.items() if v == mode)
    st.session_state["_autorun"] = target


# ------------------------------------------------------------------------------ search
def render_search(engine: PrivateSearchEngine) -> None:
    stats = engine.stats()
    st.markdown(
        '<div class="ps-hero"><h1>Search everything you’ve read. It never leaves this computer.</h1>'
        '<p>Find passages by keyword or by meaning, ask questions in plain language, and see how ideas connect '
        'across your files, with the source of every answer one click away.</p></div>', unsafe_allow_html=True)
    _index_banner(engine)
    if _no_documents(engine, "home"):
        return

    with st.form("search_form"):
        st.text_input("Search your documents", key="search_text", label_visibility="collapsed",
                      placeholder="Search your documents, for example “AI agents” or “how does hybrid search work”")
        c1, c2 = st.columns([3, 1])
        c1.radio("Method", list(MODE_LABELS), key="search_mode_label", horizontal=True,
                 help="Hybrid combines exact keywords with meaning, and usually works best.")
        submitted = c2.form_submit_button("Search", type="primary")
    auto = st.session_state.pop("_autorun", None) == "Search"
    if submitted or auto:
        _run_search(engine)

    state = st.session_state.get("search_state")
    if not state:
        a, b, c = st.columns(3)
        a.markdown(stat(stats["documents"], "documents indexed"), unsafe_allow_html=True)
        b.markdown(stat(stats["chunks"], "searchable passages"), unsafe_allow_html=True)
        c.markdown(stat(engine.index.backend.upper(), "vector index running locally"), unsafe_allow_html=True)
        return
    results = state["results"]
    if state.get("error"):
        st.error(state["error"])
        return
    if not results:
        st.markdown(empty_state("No matching passages",
                                "Try fewer or different words, or switch the method to Meaning to find "
                                "passages that use other wording."), unsafe_allow_html=True)
        return
    st.caption(f"{len(results)} passages for “{state['query']}”")
    terms = _terms(state["query"])
    for rank, result in enumerate(results, start=1):
        st.markdown(passage_card(result, terms, rank), unsafe_allow_html=True)
    st.button("See how these ideas connect", key="to_discover", on_click=_send_to_discover, args=(state["query"],))


def _run_search(engine: PrivateSearchEngine) -> None:
    query = st.session_state.get("search_text", "")
    mode = MODE_LABELS[st.session_state.get("search_mode_label", "Hybrid")]
    try:
        with st.spinner("Searching…"):
            results = engine.search(query, mode, top_k=int(engine.get_pref("top_k", "8")))
        st.session_state["search_state"] = {"query": query.strip(), "mode": mode, "results": results}
    except ValueError as exc:
        st.session_state["search_state"] = {"query": query, "mode": mode, "results": [], "error": str(exc)}


def _send_to_discover(query: str) -> None:
    st.session_state["page"] = "Discover"
    st.session_state["disc_text"] = query
    st.session_state["_autorun"] = "Discover"


# --------------------------------------------------------------------------------- ask
def render_ask(engine: PrivateSearchEngine) -> None:
    page_head("Ask your documents", "Get an answer written only from your own files, with numbered citations "
                                    "that point to the exact passage.")
    _index_banner(engine)
    if _no_documents(engine, "ask"):
        return
    llm = engine.get_llm()
    status = llm.status() if isinstance(llm, OllamaClient) else None
    if status is not None and not (status.running and status.has_model):
        st.info(f"{status.message} Until then, answers are built from sentences quoted directly from your documents.")

    with st.form("ask_form"):
        st.text_input("Your question", key="ask_text", label_visibility="collapsed",
                      placeholder="Ask a question, for example “What does a supervisor agent do?”")
        submitted = st.form_submit_button("Get answer", type="primary")
    auto = st.session_state.pop("_autorun", None) == "Ask"
    if submitted or auto:
        question = st.session_state.get("ask_text", "")
        try:
            with st.spinner("Reading your documents and writing an answer…"):
                st.session_state["answer"] = engine.ask(question, use_llm=bool(status and status.running and status.has_model) or not isinstance(llm, OllamaClient))
        except ValueError as exc:
            st.session_state["answer"] = None
            st.error(str(exc))
    answer = st.session_state.get("answer")
    if answer is None:
        return

    badge = {"llm": chip(f"Written by {answer.model} on this computer", "good"),
             "extractive": chip("Quoted from your documents", "violet"),
             "not_found": chip("Not found in your documents", "warn")}[answer.mode]
    st.markdown(badge, unsafe_allow_html=True)
    if answer.mode == "not_found":
        st.markdown(empty_state("No answer in your documents", answer.text + " Try rephrasing, or add a document "
                                "that covers this topic."), unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="ps-answer">{highlight(answer.text, [])}</div>', unsafe_allow_html=True)
    for warning in answer.warnings:
        st.warning(warning)
    if answer.grounding is not None and answer.mode == "llm":
        st.caption(f"Grounding check: {answer.grounding:.0%} of the answer’s key terms appear in the cited passages. "
                   "This is a quick heuristic, so read the sources.")
    terms = _terms(answer.question)
    if answer.sources:
        st.markdown("##### Sources")
        for number, result in answer.sources:
            c = result.chunk
            with st.expander(f"[{number}] {c.doc_name}" + (f", page {c.page}" if c.page else ""), expanded=True):
                st.markdown(passage_block(c.text, terms), unsafe_allow_html=True)
    if answer.other_sources:
        with st.expander(f"{len(answer.other_sources)} more passages were retrieved but not cited"):
            for number, result in answer.other_sources:
                st.markdown(f"**[{number}] {esc(result.chunk.location)}**", unsafe_allow_html=True)
                st.markdown(passage_block(result.chunk.text, terms), unsafe_allow_html=True)


# ----------------------------------------------------------------------------- discover
def render_discover(engine: PrivateSearchEngine) -> None:
    page_head("Knowledge discovery",
              "Search a concept and see related passages from other documents, even when they use different words. "
              "Every link shows the evidence behind it.")
    _index_banner(engine)
    if _no_documents(engine, "disc"):
        return
    default_threshold = engine.embedder.link_threshold
    with st.form("disc_form"):
        st.text_input("Concept", key="disc_text", label_visibility="collapsed",
                      placeholder="Enter a concept, for example “AI agents”")
        c1, c2, c3 = st.columns(3)
        threshold = c1.slider("Connection strictness", 0.05, 0.9, float(default_threshold), 0.01,
                              help="Higher values keep only very close connections.")
        seeds = c2.slider("Matching passages", 3, 15, 8)
        concepts = c3.toggle("Show shared concepts", value=True)
        submitted = st.form_submit_button("Find connections", type="primary")
    auto = st.session_state.pop("_autorun", None) == "Discover"
    if submitted or auto:
        try:
            with st.spinner("Finding connections…"):
                st.session_state["disc"] = engine.discover(st.session_state.get("disc_text", ""), seed_k=seeds,
                                                           link_threshold=threshold)
                st.session_state["disc_concepts"] = concepts
        except ValueError as exc:
            st.session_state["disc"] = None
            st.error(str(exc))
    result = st.session_state.get("disc")
    if result is None:
        st.markdown(empty_state("Nothing to map yet", "Enter a concept above. The map shows your search in the "
                                "centre, matching passages around it, and lines between passages from different "
                                "documents that say related things."), unsafe_allow_html=True)
        return
    if result.message:
        st.markdown(empty_state("No map to draw", result.message), unsafe_allow_html=True)
        return
    if not result.semantic:
        st.warning("The offline fallback model is active, so links reflect similar word patterns rather than "
                   "meaning. Install the sentence-transformers model for true semantic discovery.")

    docs = {n.chunk.doc_name for n in result.passages}
    a, b, c = st.columns(3)
    a.markdown(stat(len(result.passages), "passages"), unsafe_allow_html=True)
    b.markdown(stat(len(docs), "documents"), unsafe_allow_html=True)
    c.markdown(stat(len(result.connections), "cross-document links"), unsafe_allow_html=True)
    if not result.connections:
        st.info("No cross-document links met the strictness setting. Lower it to see weaker connections.")

    selected = st.session_state.get("disc_pick")
    fig = build_figure(result, st.session_state.get("disc_concepts", True), selected)
    event = st.plotly_chart(fig, on_select="rerun", selection_mode="points", key="disc_map")
    clicked = None
    try:
        points = event.selection.points if event and event.selection else []
        if points:
            cd = points[0].get("customdata")
            clicked = cd[0] if isinstance(cd, (list, tuple)) else cd
    except (AttributeError, KeyError, IndexError, TypeError):
        clicked = None
    st.caption("Hover over a circle for its passage or over a line for the reason it exists. "
               "Click a circle to inspect it.")

    passage_ids = [n.id for n in result.passages]
    if clicked in passage_ids:
        picked = clicked
    else:
        picked = st.selectbox("Inspect a passage", passage_ids, format_func=lambda i: result.node(i).chunk.location
                              + (" (matches your search)" if result.node(i).role == "seed" else " (related)"),
                              key="disc_pick_box")
    _inspect(result, picked)

    st.markdown("##### Why these passages connect")
    for edge in result.connections[:10]:
        na, nb = result.node(edge.source), result.node(edge.target)
        with st.expander(f"{na.chunk.location}  and  {nb.chunk.location}  (similarity {edge.weight:.2f})"):
            st.markdown(f"**{edge.explanation}**")
            st.markdown(f'<div class="ps-pair"><div class="ps-card">{highlight(na.chunk.text, edge.shared_terms, "shared")}'
                        f'</div><div class="ps-card">{highlight(nb.chunk.text, edge.shared_terms, "shared")}</div></div>',
                        unsafe_allow_html=True)
    if result.concepts:
        st.markdown("##### Concepts that appear across documents")
        st.caption("Terms found in passages from at least two different documents, ranked by how distinctive they are.")
        for concept in result.concepts:
            with st.expander(f"{concept.term}  in {len(concept.doc_names)} documents"):
                st.caption(", ".join(concept.doc_names))
                for node in result.passages:
                    if node.chunk.id in concept.chunk_ids:
                        st.markdown(f"**{esc(node.chunk.location)}**", unsafe_allow_html=True)
                        st.markdown(passage_block(node.chunk.text, [concept.term]), unsafe_allow_html=True)


def _inspect(result, node_id: str) -> None:
    node = result.node(node_id)
    if node is None or node.chunk is None:
        return
    st.markdown(f'<div class="ps-card accent"><div class="ps-head"><span class="ps-doc">{esc(node.chunk.location)}</span>'
                f'{chip("matches your search" if node.role == "seed" else "related passage", "violet")}</div>'
                f'{passage_block(node.chunk.text, _terms(result.query))}'
                f'<div class="ps-meta">{esc(node.reason)}</div></div>', unsafe_allow_html=True)
    links = [e for e in result.edges_for(node_id) if e.kind == "semantic-link"]
    if not links:
        st.caption("This passage has no cross-document links at the current strictness.")
    for edge in sorted(links, key=lambda e: -e.weight):
        other = result.node(edge.target if edge.source == node_id else edge.source)
        st.markdown(f"**Connected to {esc(other.chunk.location)}** (similarity {edge.weight:.2f})", unsafe_allow_html=True)
        st.caption(edge.explanation)
        st.markdown(passage_block(other.chunk.text, edge.shared_terms, "shared"), unsafe_allow_html=True)


# --------------------------------------------------------------------------- documents
def render_documents(engine: PrivateSearchEngine) -> None:
    page_head("Documents", "Add files to your private library, review what is indexed, or remove anything you no longer want searchable.")
    _index_banner(engine)
    nonce = st.session_state.setdefault("uploader_nonce", 0)
    files = st.file_uploader("Drop files here", type=[e.lstrip(".") for e in ALLOWED_EXTENSIONS],
                             accept_multiple_files=True, key=f"uploader_{nonce}",
                             help=f"PDF, DOCX, TXT or Markdown, up to {engine.settings.max_upload_mb} MB each.")
    c1, c2, _ = st.columns([1, 1.4, 2])
    if files and c1.button(f"Index {len(files)} file{'s' if len(files) != 1 else ''}", type="primary"):
        progress = st.progress(0.0)
        outcomes = []
        with st.status("Reading and indexing…", expanded=True) as status:
            for n, upload in enumerate(files, start=1):
                st.write(f"Processing {upload.name}")
                outcomes.append(engine.ingest(upload.name, upload.getvalue()))
                progress.progress(n / len(files))
            status.update(label="Finished", state="complete")
        st.session_state["last_ingest"] = outcomes
        st.session_state["uploader_nonce"] = nonce + 1
        st.rerun()
    if c2.button("Add the sample documents"):
        load_samples(engine)
        st.rerun()
    for outcome in st.session_state.pop("last_ingest", []):
        {"added": st.success, "duplicate": st.info, "error": st.error}[outcome.status](f"{outcome.filename}: {outcome.message}")

    docs = engine.list_documents()
    st.markdown("##### Indexed documents")
    if not docs:
        st.markdown(empty_state("Nothing indexed yet", "Upload a file above to get started."), unsafe_allow_html=True)
    for doc in docs:
        with st.container(border=True):
            a, b, c = st.columns([5, 3, 1.4])
            a.markdown(f"**{esc(doc.name)}**", unsafe_allow_html=True)
            pages = f"{doc.n_pages} pages, " if doc.n_pages else ""
            b.caption(f"{pages}{doc.n_chunks} passages, {doc.size_bytes / 1024:.0f} KB, added {doc.added_at[:10]}")
            with c.popover("Remove"):
                st.write("Delete this document and all of its search data?")
                if st.button("Remove permanently", key=f"rm_{doc.id}", type="primary"):
                    engine.delete_document(doc.id)
                    st.session_state.pop("search_state", None)
                    st.session_state.pop("answer", None)
                    st.session_state.pop("disc", None)
                    st.rerun()

    st.markdown("##### Index maintenance")
    st.caption(f"Vector index: {engine.index.backend.upper()}, {len(engine.index)} vectors, model {engine.embedder.name}.")
    if st.button("Rebuild index", help="Re-reads every stored file, splits it again and recomputes all embeddings."):
        bar = st.progress(0.0, text="Starting…")
        report = engine.rebuild_index(lambda f, m: bar.progress(min(f, 1.0), text=m))
        st.success(f"Rebuilt the index for {report['documents']} documents ({report['passages']} passages).")
        st.session_state.pop("disc", None)


# ----------------------------------------------------------------------------- privacy
def render_privacy(engine: PrivateSearchEngine) -> None:
    page_head("Privacy and settings", "What stays on your computer, which local model is used, and how to erase everything.")
    st.markdown(
        "| Data | Where it lives | Leaves your computer? |\n|---|---|---|\n"
        f"| Your files | `{Path(engine.settings.uploads_dir).as_posix()}` | No |\n"
        f"| Search index and history | `{Path(engine.settings.db_path).as_posix()}` | No |\n"
        "| Embeddings | Same database, plus an in-memory FAISS index | No |\n"
        "| Questions and answers | Processed by Ollama on `localhost` | No |\n"
        "| Telemetry | None. Analytics are switched off in every library used | No |\n")
    st.caption("The only network use is a one-time download of the embedding model (about 90 MB), which sends no "
               "documents. After that you can set PSAI_OFFLINE=1 to block all downloads.")
    if engine.embedder.fallback_reason:
        st.warning("The sentence-transformers model could not be loaded, so the offline fallback is active.")
        st.code(engine.embedder.fallback_reason)
        st.markdown("Connect to the internet once and restart the app so the model can download, or run "
                    "`pip install sentence-transformers`.")

    st.markdown("##### Local language model")
    llm = engine.get_llm()
    if isinstance(llm, OllamaClient):
        status = llm.status()
        st.markdown(chip("Ollama running", "good") if status.running else chip("Ollama not running", "warn"),
                    unsafe_allow_html=True)
        if status.running and status.models:
            current = engine.get_pref("ollama_model", engine.settings.ollama_model)
            choice = st.selectbox("Model used for answers", status.models,
                                  index=status.models.index(current) if current in status.models else 0)
            if choice != current:
                engine.set_pref("ollama_model", choice)
                st.rerun()
        st.caption(status.message)
    else:
        st.warning("The configured Ollama address is not on this computer, so it is blocked to protect your privacy.")

    st.markdown("##### Search behaviour")
    top_k = st.slider("Results per search", 3, 20, int(engine.get_pref("top_k", "8")))
    weight = st.slider("Hybrid balance: keywords (0) to meaning (1)", 0.0, 1.0,
                       float(engine.get_pref("semantic_weight", "0.6")), 0.05)
    if top_k != int(engine.get_pref("top_k", "8")) or weight != float(engine.get_pref("semantic_weight", "0.6")):
        engine.set_pref("top_k", top_k)
        engine.set_pref("semantic_weight", weight)
        st.toast("Settings saved")

    st.markdown("##### Erase data")
    c1, c2 = st.columns(2)
    if c1.button("Clear search history"):
        engine.clear_history()
        st.session_state.pop("search_state", None)
        st.success("Search history cleared.")
    with c2.popover("Delete all documents and history"):
        st.write("This removes every indexed document, its stored copy and your history. It cannot be undone.")
        if st.text_input("Type DELETE to confirm", key="wipe_confirm") == "DELETE" and st.button("Delete everything", type="primary"):
            engine.delete_everything()
            for key in ("search_state", "answer", "disc"):
                st.session_state.pop(key, None)
            st.success("Everything was deleted.")
