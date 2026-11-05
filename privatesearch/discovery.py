"""Knowledge Discovery: evidence-based connections between passages in different documents.

Nothing here is invented. Every connection is backed by
  1. the cosine similarity of two stored passage embeddings (above a threshold), and
  2. the literal terms the two passages share (possibly none - then we say so).
The UI always shows both passages so the user can verify the link.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

from .models import ChunkRow
from .textutil import content_stems, extract_terms, stem, tokenize

if TYPE_CHECKING:  # pragma: no cover
    from .engine import PrivateSearchEngine


@dataclass
class GraphNode:
    id: str
    kind: str  # query | passage | concept
    label: str
    chunk: ChunkRow | None = None
    role: str = ""  # seed | related (passages only)
    reason: str = ""
    score: float = 0.0


@dataclass
class GraphEdge:
    source: str
    target: str
    kind: str  # query-match | semantic-link | concept-mention
    weight: float
    explanation: str
    shared_terms: list[str] = field(default_factory=list)


@dataclass
class Concept:
    term: str
    doc_names: list[str]
    chunk_ids: list[int]
    score: float


@dataclass
class DiscoveryResult:
    query: str
    nodes: list[GraphNode] = field(default_factory=list)
    edges: list[GraphEdge] = field(default_factory=list)
    concepts: list[Concept] = field(default_factory=list)
    link_threshold: float = 0.0
    semantic: bool = True
    message: str = ""

    @property
    def passages(self) -> list[GraphNode]:
        return [n for n in self.nodes if n.kind == "passage"]

    @property
    def connections(self) -> list[GraphEdge]:
        return sorted((e for e in self.edges if e.kind == "semantic-link"), key=lambda e: -e.weight)

    def node(self, node_id: str) -> GraphNode | None:
        return next((n for n in self.nodes if n.id == node_id), None)

    def edges_for(self, node_id: str) -> list[GraphEdge]:
        return [e for e in self.edges if node_id in (e.source, e.target)]


def passage_id(chunk_id: int) -> str:
    return f"p{chunk_id}"


def _short(chunk: ChunkRow) -> str:
    name = chunk.doc_name if len(chunk.doc_name) <= 22 else chunk.doc_name[:20] + "…"
    return f"{name} p.{chunk.page}" if chunk.page else name


def explain_link(similarity: float, shared: list[str], semantic: bool) -> str:
    basis = "embedding similarity" if semantic else "word-pattern similarity (fallback model, not true semantics)"
    if shared:
        return (f"Passages from different documents with {basis} {similarity:.2f}. "
                f"Both mention: {', '.join(shared)}.")
    return (f"Different wording, close in meaning: {basis} {similarity:.2f}. No shared key terms were found, "
            "so this link rests on the model's judgement. Read both passages to confirm it.")


def discover(engine: "PrivateSearchEngine", query: str, seed_k: int = 8, max_per_doc: int = 2,
             link_threshold: float | None = None, neighbors_per_seed: int = 3, max_related: int = 8,
             max_concepts: int = 8, max_edges_per_node: int = 5) -> DiscoveryResult:
    embedder = engine.embedder
    threshold = embedder.link_threshold if link_threshold is None else link_threshold
    result = DiscoveryResult(query=query, link_threshold=threshold, semantic=embedder.is_semantic)

    # 1. Seed passages: the best hybrid matches, at most `max_per_doc` per document for variety.
    candidates = engine.search(query, mode="hybrid", top_k=seed_k * 4, record_history=False)
    seeds, per_doc = [], {}
    for item in candidates:
        if per_doc.get(item.chunk.doc_id, 0) < max_per_doc:
            seeds.append(item)
            per_doc[item.chunk.doc_id] = per_doc.get(item.chunk.doc_id, 0) + 1
        if len(seeds) == seed_k:
            break
    if not seeds:
        result.message = "No passages matched this search. Try different words, or add more documents."
        return result

    seed_ids = [s.chunk_id for s in seeds]
    seed_vecs = dict(zip(*engine.get_vectors(seed_ids)))

    # 2. Expansion: nearest passages from OTHER documents, even when they use different words.
    related: dict[int, float] = {}
    for item in seeds:
        vec = seed_vecs.get(item.chunk_id)
        if vec is None:
            continue
        found = 0
        for chunk_id, sim in engine.index.search(vec, k=60):
            if chunk_id in seed_ids or chunk_id in related or sim < threshold:
                continue
            neighbor = engine.db.get_chunks([chunk_id]).get(chunk_id)
            if neighbor is None or neighbor.doc_id == item.chunk.doc_id:
                continue
            related[chunk_id] = sim
            found += 1
            if found >= neighbors_per_seed:
                break
    related_ids = [cid for cid, _ in sorted(related.items(), key=lambda kv: -kv[1])[:max_related]]

    node_ids = seed_ids + related_ids
    chunks = engine.db.get_chunks(node_ids)
    ids, matrix = engine.get_vectors(node_ids)
    vectors = dict(zip(ids, matrix))
    result.nodes.append(GraphNode("q", "query", query))
    seed_lookup = {s.chunk_id: s for s in seeds}
    for cid in node_ids:
        chunk = chunks.get(cid)
        if chunk is None or cid not in vectors:
            continue
        if cid in seed_lookup:
            s = seed_lookup[cid]
            cos = f"{s.cosine:.2f}" if s.cosine is not None else "n/a"
            reason = f"Matches your search (semantic similarity {cos})"
            if s.matched_terms:
                reason += f"; contains: {', '.join(s.matched_terms[:5])}"
            result.nodes.append(GraphNode(passage_id(cid), "passage", _short(chunk), chunk, "seed", reason, s.score))
            result.edges.append(GraphEdge("q", passage_id(cid), "query-match", max(s.score, 0.05), reason))
        else:
            result.nodes.append(GraphNode(passage_id(cid), "passage", _short(chunk), chunk, "related",
                                          "Found through its similarity to a matching passage", related[cid]))

    # 3. Connections between passages of different documents, each with literal shared terms.
    passages = result.passages
    term_sets = {n.id: extract_terms(n.chunk.text) for n in passages}
    degree: dict[str, int] = {}
    pairs = []
    for i in range(len(passages)):
        for j in range(i + 1, len(passages)):
            a, b = passages[i], passages[j]
            if a.chunk.doc_id == b.chunk.doc_id:
                continue
            sim = float(np.dot(vectors[a.chunk.id], vectors[b.chunk.id]))
            if sim >= threshold:
                pairs.append((sim, a, b))
    for sim, a, b in sorted(pairs, key=lambda p: -p[0]):
        if degree.get(a.id, 0) >= max_edges_per_node or degree.get(b.id, 0) >= max_edges_per_node:
            continue
        shared_keys = set(term_sets[a.id]) & set(term_sets[b.id])
        shared = sorted((term_sets[a.id][k] for k in shared_keys), key=lambda t: (-len(t.split()), t))[:6]
        result.edges.append(GraphEdge(a.id, b.id, "semantic-link", sim, explain_link(sim, shared, embedder.is_semantic), shared))
        degree[a.id] = degree.get(a.id, 0) + 1
        degree[b.id] = degree.get(b.id, 0) + 1

    # 4. Bridge concepts: terms that occur in passages from at least two different documents.
    query_stems = content_stems(query)
    total_chunks = max(engine.db.stats()["chunks"], 1)
    holders: dict[str, dict] = {}
    for node in passages:
        for key, surface in term_sets[node.id].items():
            entry = holders.setdefault(key, {"surface": surface, "nodes": []})
            entry["nodes"].append(node)
    scored: list[Concept] = []
    for key, entry in holders.items():
        parts = key.split()
        if all(p in query_stems for p in parts):
            continue
        docs = {n.chunk.doc_name for n in entry["nodes"]}
        if len(docs) < 2:
            continue
        df = max(engine.db.fts_count('"' + entry["surface"] + '"'), 1)
        if df / total_chunks > 0.4 and total_chunks > 10:
            continue
        score = len(docs) * math.log(1 + total_chunks / df) * (1.3 if len(parts) > 1 else 1.0)
        scored.append(Concept(entry["surface"], sorted(docs), [n.chunk.id for n in entry["nodes"]], score))
    result.concepts = sorted(scored, key=lambda c: -c.score)[:max_concepts]
    for concept in result.concepts:
        cid = f"t:{concept.term}"
        result.nodes.append(GraphNode(cid, "concept", concept.term, reason=f"Appears in {len(concept.doc_names)} documents"))
        for chunk_id in concept.chunk_ids:
            result.edges.append(GraphEdge(cid, passage_id(chunk_id), "concept-mention", 0.3,
                                          f"The passage contains the term '{concept.term}'.", [concept.term]))
    return result
