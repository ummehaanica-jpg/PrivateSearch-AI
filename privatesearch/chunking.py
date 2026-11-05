"""Sentence-aware chunking. Chunks never cross page boundaries, so page citations stay exact."""
from __future__ import annotations

from dataclasses import dataclass

from .extraction import ExtractedPage
from .textutil import split_sentences


@dataclass
class Chunk:
    index: int
    page: int | None
    text: str

    @property
    def n_words(self) -> int:
        return len(self.text.split())


def _explode_long(sentence: str, limit: int) -> list[str]:
    words = sentence.split()
    if len(words) <= limit:
        return [sentence]
    return [" ".join(words[i:i + limit]) for i in range(0, len(words), limit)]


def chunk_page(text: str, target_words: int, overlap_words: int) -> list[str]:
    sentences: list[str] = []
    for sentence in split_sentences(text):
        sentences.extend(_explode_long(sentence, max(target_words, 20)))

    chunks: list[str] = []
    current: list[str] = []
    current_words = 0
    has_new = False
    for sentence in sentences:
        words = len(sentence.split())
        if current and current_words + words > target_words and has_new:
            chunks.append(" ".join(current))
            # carry the last sentences forward as overlap
            carried: list[str] = []
            carried_words = 0
            for previous in reversed(current):
                previous_words = len(previous.split())
                if carried_words + previous_words > overlap_words:
                    break
                carried.insert(0, previous)
                carried_words += previous_words
            current, current_words, has_new = carried, carried_words, False
        current.append(sentence)
        current_words += words
        has_new = True
    if current and has_new:
        chunks.append(" ".join(current))
    return chunks


def chunk_pages(pages: list[ExtractedPage], target_words: int = 160, overlap_words: int = 30) -> list[Chunk]:
    if target_words < 20:
        raise ValueError("target_words must be at least 20")
    overlap_words = max(0, min(overlap_words, target_words // 2))
    chunks: list[Chunk] = []
    for page in pages:
        for text in chunk_page(page.text, target_words, overlap_words):
            if text.strip():
                chunks.append(Chunk(len(chunks), page.page, text.strip()))
    return chunks
