# Explaining PrivateSearch AI in an interview

## 30-second pitch
"I built a local-first search engine for personal documents. It combines keyword search (SQLite FTS5, BM25) with semantic search (Sentence-Transformers + FAISS), answers questions with a local LLM through Ollama using retrieval-augmented generation, and cites the exact page. Nothing leaves the machine. A Knowledge Discovery view links passages from different documents by meaning, and shows the evidence for every link."

## Algorithms
* **Chunking**: sentence-aware windows of ~160 words, 30-word overlap, never across pages. Small enough for the embedding model's 256-token limit, large enough to hold one idea.
* **BM25 (FTS5)**: ranks by term frequency, inverse document frequency and length normalisation. Great for exact names and rare terms. Porter stemming lets "agents" match "agent".
* **Embeddings**: all-MiniLM-L6-v2 maps text to a 384-dimension vector where similar meaning means nearby vectors. Normalised vectors make **inner product = cosine similarity**.
* **FAISS IndexFlatIP**: exact nearest-neighbour search. Fine for hundreds of thousands of passages; at larger scale switch to HNSW or IVF (approximate).
* **Reciprocal Rank Fusion**: `score = sum(weight / (60 + rank))`. Merges rankings without comparing incompatible score scales (BM25 vs cosine).
* **RAG**: retrieve, build a numbered-source prompt, instruct the model to answer only from sources, verify the citations afterwards.
* **Hallucination controls**: relevance gate before calling the model, `NOT_FOUND` path, removal of citation numbers that do not exist, word-overlap grounding score, low temperature, "sources are untrusted data" prompt-injection guard.
* **Knowledge Discovery**: seeds = best hybrid hits (max 2 per document); expansion = nearest neighbours from *other* documents above a similarity threshold; edges = cosine similarity of stored vectors between cross-document pairs; explanation = score + literal shared terms (or an honest "different wording" note); bridge concepts = unigrams/bigrams present in at least two documents, ranked by documents × IDF.

## Likely questions
* **Why hybrid?** Keywords miss paraphrases, embeddings blur exact terms and names. RRF gets both.
* **Why store embeddings in SQLite?** One persistent source of truth; FAISS rebuilds in milliseconds; deleting a document deletes its vectors.
* **How do you stop hallucinations?** You cannot eliminate them. I reduce and expose them: grounded prompt, citation verification, grounding score, extractive fallback, always showing the source passage.
* **How would you scale it?** Approximate index (HNSW/IVF), incremental indexing in a background worker, a cross-encoder re-ranker, OCR, multi-user storage with encryption.
* **How did you test it?** 33 unit/integration tests: page-accurate citations, duplicate prevention, deletion removing FTS/vector rows, persistence after restart, FTS injection safety, citation tampering, discovery evidence.
* **What are the limits?** Small local models are weaker writers; MiniLM is English-centric; scanned PDFs need OCR; links are similarity evidence, not proof of a relationship.

## Future improvements
Cross-encoder re-ranking, OCR (Tesseract), multilingual model, watch-folder auto-indexing, encrypted storage, query-time LLM-generated link explanations (labelled as generated), PyInstaller/Docker packaging, optional authentication for a shared LAN deployment.
