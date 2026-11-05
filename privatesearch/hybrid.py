"""Hybrid ranking: Reciprocal Rank Fusion (RRF) of keyword (BM25) and semantic (cosine) rankings."""
from __future__ import annotations

import re

from .textutil import STOPWORDS

MAX_QUERY_CHARS = 500


def validate_query(query: str) -> str:
    query = (query or "").strip()
    if not query:
        raise ValueError("Type something to search for.")
    if len(query) > MAX_QUERY_CHARS:
        raise ValueError(f"Queries are limited to {MAX_QUERY_CHARS} characters.")
    return query


def build_fts_query(query: str, max_terms: int = 12) -> str:
    """Turn free text into a safe FTS5 query: quoted tokens joined by OR.

    Quoting every token means characters such as " ( ) * : - cannot be interpreted as FTS5 syntax.
    """
    tokens = re.findall(r"\w+", query.lower())
    useful = [t for t in tokens if t not in STOPWORDS] or tokens
    unique = list(dict.fromkeys(useful))[:max_terms]
    return " OR ".join(f'"{t}"' for t in unique)


def rrf_fuse(rankings: dict[str, list[int]], weights: dict[str, float], k: int = 60) -> dict[int, float]:
    """score(d) = sum_over_rankers weight * 1 / (k + rank). Ranks start at 1."""
    scores: dict[int, float] = {}
    for name, ranked_ids in rankings.items():
        weight = weights.get(name, 1.0)
        for rank, item in enumerate(ranked_ids, start=1):
            scores[item] = scores.get(item, 0.0) + weight / (k + rank)
    return scores
