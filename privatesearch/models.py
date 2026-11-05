"""Plain data containers shared across modules."""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class DocumentRow:
    id: int
    name: str
    sha256: str
    ext: str
    stored_name: str
    size_bytes: int
    n_pages: int | None
    n_chunks: int
    added_at: str


@dataclass(frozen=True)
class ChunkRow:
    id: int
    doc_id: int
    doc_name: str
    chunk_index: int
    page: int | None
    text: str

    @property
    def location(self) -> str:
        return f"{self.doc_name}, p. {self.page}" if self.page else self.doc_name


@dataclass
class SearchResult:
    chunk: ChunkRow
    score: float  # fused score scaled to 0..1
    keyword_rank: int | None = None
    semantic_rank: int | None = None
    bm25: float | None = None
    cosine: float | None = None
    matched_terms: list[str] = field(default_factory=list)
    coverage: float = 0.0  # share of query terms found in the passage

    @property
    def chunk_id(self) -> int:
        return self.chunk.id


@dataclass
class IngestResult:
    filename: str
    status: str  # added | duplicate | error
    message: str
    doc_id: int | None = None
    n_chunks: int = 0


@dataclass
class Answer:
    question: str
    text: str
    mode: str  # llm | extractive | not_found
    sources: list[tuple[int, SearchResult]] = field(default_factory=list)  # (citation number, result)
    other_sources: list[tuple[int, SearchResult]] = field(default_factory=list)
    grounding: float | None = None
    warnings: list[str] = field(default_factory=list)
    model: str | None = None
