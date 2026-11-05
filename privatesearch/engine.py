"""PrivateSearchEngine: the single entry point the UI (and the tests) talk to."""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Callable

import numpy as np

from . import qa
from .chunking import chunk_pages
from .config import Settings, apply_privacy_environment
from .database import Database
from .discovery import DiscoveryResult, discover
from .embeddings import BaseEmbedder, load_embedder
from .extraction import ExtractionError, extract_text, sanitize_filename, validate_filename
from .hybrid import build_fts_query, rrf_fuse, validate_query
from .llm import LLMClient, LLMError, OllamaClient
from .models import Answer, DocumentRow, IngestResult, SearchResult
from .textutil import content_stems, stem, tokenize, is_content_word
from .vector_index import VectorIndex

EMBEDDER_KEY = "embedder_id"
SEARCH_MODES = ("hybrid", "keyword", "semantic")


class PrivateSearchEngine:
    def __init__(self, settings: Settings | None = None, embedder: BaseEmbedder | None = None,
                 llm: LLMClient | None = None):
        self.settings = settings or Settings.from_env()
        apply_privacy_environment(self.settings)
        self.settings.ensure_dirs()
        self.db = Database(self.settings.db_path)
        self.embedder = embedder or load_embedder(self.settings)
        self._llm = llm
        self.index = VectorIndex(self.embedder.dim)
        self.index_stale = False
        self.stored_embedder_id: str | None = None
        self._load_index()

    # ------------------------------------------------------------------ index
    def _load_index(self) -> None:
        stored = self.db.get_meta(EMBEDDER_KEY)
        self.stored_embedder_id = stored
        stats = self.db.stats()
        if stored is None or stats["documents"] == 0:
            self.db.set_meta(EMBEDDER_KEY, self.embedder.name)
            self.stored_embedder_id = self.embedder.name
            self.index_stale = False
            return
        if stored != self.embedder.name:
            # Vectors were made by a different model; mixing them would silently corrupt results.
            self.index_stale = True
            return
        ids, matrix = self.db.load_embeddings()
        self.index = VectorIndex(self.embedder.dim)
        if ids:
            self.index.add(ids, matrix)
        self.index_stale = False

    def rebuild_index(self, progress: Callable[[float, str], None] | None = None) -> dict[str, int]:
        """Re-read every stored original, re-chunk, re-embed and rebuild FAISS from scratch."""
        report = lambda f, m: progress(f, m) if progress else None
        documents = self.db.list_documents()
        rechunked = 0
        for n, doc in enumerate(documents, start=1):
            report(0.6 * (n - 1) / max(len(documents), 1), f"Reading {doc.name}")
            path = self._stored_path(doc.stored_name)
            if path is None or not path.is_file():
                continue
            try:
                extracted = extract_text(path, self.settings.max_pdf_pages)
            except ExtractionError:
                continue
            chunks = chunk_pages(extracted.pages, self.settings.chunk_words, self.settings.chunk_overlap_words)
            if not chunks:
                continue
            vectors = self.embedder.encode([c.text for c in chunks])
            self.db.replace_document_chunks(doc.id, [(c.page, c.text) for c in chunks], vectors, extracted.n_pages)
            rechunked += 1
        ids, texts = self.db.all_chunk_texts()
        self.index = VectorIndex(self.embedder.dim)
        for start in range(0, len(ids), 64):
            report(0.6 + 0.4 * start / max(len(ids), 1), "Computing embeddings")
            vectors = self.embedder.encode(texts[start:start + 64])
            self.db.set_embeddings(ids[start:start + 64], vectors)
            self.index.add(ids[start:start + 64], vectors)
        self.db.set_meta(EMBEDDER_KEY, self.embedder.name)
        self.stored_embedder_id, self.index_stale = self.embedder.name, False
        report(1.0, "Done")
        return {"documents": len(documents), "rechunked": rechunked, "passages": len(ids)}

    def get_vectors(self, chunk_ids: list[int]) -> tuple[list[int], np.ndarray]:
        return self.db.load_embeddings(chunk_ids)

    def _stored_path(self, stored_name: str) -> Path | None:
        path = (self.settings.uploads_dir / stored_name).resolve()
        try:
            path.relative_to(self.settings.uploads_dir.resolve())
        except ValueError:
            return None
        return path

    # -------------------------------------------------------------- documents
    def ingest(self, filename: str, data: bytes) -> IngestResult:
        name = sanitize_filename(filename)
        try:
            ext = validate_filename(name)
            if not data:
                raise ExtractionError("The file is empty.")
            if len(data) > self.settings.max_upload_bytes:
                raise ExtractionError(f"The file is larger than {self.settings.max_upload_mb} MB.")
            if self.index_stale:
                raise ExtractionError("The index was built with a different embedding model. "
                                      "Rebuild the index on the Documents page first.")
            sha = hashlib.sha256(data).hexdigest()
            existing = self.db.find_by_hash(sha)
            if existing:
                return IngestResult(name, "duplicate", f"Already indexed as '{existing.name}'.", existing.id,
                                    existing.n_chunks)
            stored_name = f"{sha}{ext}"
            path = self._stored_path(stored_name)
            assert path is not None
            tmp = path.with_suffix(path.suffix + ".part")
            tmp.write_bytes(data)
            tmp.replace(path)
            try:
                extracted = extract_text(path, self.settings.max_pdf_pages)
                chunks = chunk_pages(extracted.pages, self.settings.chunk_words, self.settings.chunk_overlap_words)
                if not chunks:
                    raise ExtractionError("No readable text was found in this file.")
                vectors = self.embedder.encode([c.text for c in chunks])
                doc_id, chunk_ids = self.db.add_document(
                    name=name, sha256=sha, ext=ext, stored_name=stored_name, size_bytes=len(data),
                    n_pages=extracted.n_pages, chunks=[(c.page, c.text) for c in chunks], vectors=vectors)
            except Exception:
                path.unlink(missing_ok=True)
                raise
            self.index.add(chunk_ids, vectors)
            return IngestResult(name, "added", f"Indexed {len(chunks)} passages.", doc_id, len(chunks))
        except ExtractionError as exc:
            return IngestResult(name, "error", str(exc))
        except Exception as exc:  # never let one bad file crash the app
            return IngestResult(name, "error", f"Unexpected problem while reading this file: {exc}")

    def ingest_path(self, path: Path) -> IngestResult:
        return self.ingest(path.name, Path(path).read_bytes())

    def list_documents(self) -> list[DocumentRow]:
        return self.db.list_documents()

    def delete_document(self, doc_id: int) -> bool:
        chunk_ids, stored_name = self.db.delete_document(doc_id)
        if stored_name is None:
            return False
        self.index.remove(chunk_ids)
        path = self._stored_path(stored_name)
        if path is not None:
            path.unlink(missing_ok=True)
        return True

    def delete_everything(self) -> None:
        self.db.wipe()
        self.index = VectorIndex(self.embedder.dim)
        for f in self.settings.uploads_dir.glob("*"):
            if f.is_file():
                f.unlink(missing_ok=True)
        self.db.set_meta(EMBEDDER_KEY, self.embedder.name)
        self.index_stale = False

    def stats(self) -> dict[str, int]:
        return self.db.stats()

    # ----------------------------------------------------------------- search
    def search(self, query: str, mode: str = "hybrid", top_k: int = 10, semantic_weight: float | None = None,
               record_history: bool = True) -> list[SearchResult]:
        query = validate_query(query)
        if mode not in SEARCH_MODES:
            raise ValueError(f"Unknown search mode '{mode}'.")
        alpha = float(self.get_pref("semantic_weight", "0.6")) if semantic_weight is None else semantic_weight
        pool = max(50, top_k * 5)

        keyword_hits = self.db.fts_search(build_fts_query(query), pool) if mode in ("hybrid", "keyword") else []
        bm25 = dict(keyword_hits)

        cosines: dict[int, float] = {}
        semantic_ids: list[int] = []
        if mode in ("hybrid", "semantic") and not self.index_stale and len(self.index):
            hits = self.index.search(self.embedder.encode_query(query), min(pool * 4, len(self.index)))
            cosines = dict(hits)
            semantic_ids = [cid for cid, s in hits if s >= self.embedder.min_score][:pool]
        if mode == "hybrid" and keyword_hits and not self.index_stale:
            # make sure keyword-only hits also carry a cosine so the UI and QA gate can use it
            missing = [cid for cid, _ in keyword_hits if cid not in cosines]
            if missing:
                ids, mat = self.db.load_embeddings(missing)
                q = self.embedder.encode_query(query)
                cosines.update({i: float(v @ q) for i, v in zip(ids, mat)})

        keyword_ids = [cid for cid, _ in keyword_hits]
        weights = {"keyword": 1.0 - alpha, "semantic": alpha}
        if mode == "keyword":
            weights = {"keyword": 1.0}
        elif mode == "semantic":
            weights = {"semantic": 1.0}
        fused = rrf_fuse({"keyword": keyword_ids, "semantic": semantic_ids}, weights)
        if not fused:
            if record_history:
                self.db.add_history(query, mode)
            return []
        ranked = sorted(fused, key=lambda c: -fused[c])[:top_k]
        best_possible = sum(weights.values()) / 61
        chunks = self.db.get_chunks(ranked)
        kw_rank = {cid: r for r, cid in enumerate(keyword_ids, start=1)}
        sem_rank = {cid: r for r, cid in enumerate(semantic_ids, start=1)}
        query_stems = {stem(t) for t in tokenize(query) if is_content_word(t)}

        results = []
        for cid in ranked:
            chunk = chunks.get(cid)
            if chunk is None:
                continue
            chunk_stems = content_stems(chunk.text)
            matched = sorted(s for s in query_stems if s in chunk_stems)
            coverage = len(matched) / len(query_stems) if query_stems else 0.0
            results.append(SearchResult(
                chunk=chunk, score=min(fused[cid] / best_possible, 1.0), keyword_rank=kw_rank.get(cid),
                semantic_rank=sem_rank.get(cid), bm25=bm25.get(cid), cosine=cosines.get(cid),
                matched_terms=matched, coverage=coverage))
        if record_history:
            self.db.add_history(query, mode)
        return results

    # --------------------------------------------------------------------- QA
    def get_llm(self) -> LLMClient | None:
        if self._llm is not None:
            return self._llm
        model = self.get_pref("ollama_model", self.settings.ollama_model)
        try:
            return OllamaClient(self.settings.ollama_host, model, allow_remote=self.settings.allow_remote_llm)
        except LLMError:
            return None

    def ask(self, question: str, top_k: int = 6, use_llm: bool = True) -> Answer:
        question = validate_query(question)
        results = self.search(question, "hybrid", top_k=top_k, record_history=False)
        answer = qa.answer_question(question, results, self.get_llm() if use_llm else None, self.embedder)
        self.db.add_history(question, "ask")
        return answer

    # -------------------------------------------------------------- discovery
    def discover(self, query: str, **kwargs) -> DiscoveryResult:
        query = validate_query(query)
        if self.index_stale:
            return DiscoveryResult(query=query, message="The index must be rebuilt before connections can be found.")
        result = discover(self, query, **kwargs)
        self.db.add_history(query, "discover")
        return result

    # ---------------------------------------------------- history / preferences
    def history(self, limit: int = 8) -> list[tuple[str, str]]:
        return self.db.recent_history(limit)

    def clear_history(self) -> None:
        self.db.clear_history()

    def get_pref(self, key: str, default: str) -> str:
        return self.db.get_meta(f"pref:{key}", default) or default

    def set_pref(self, key: str, value: object) -> None:
        self.db.set_meta(f"pref:{key}", str(value))
