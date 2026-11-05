"""SQLite storage: documents, chunks, FTS5 keyword index, embeddings, history, prefs."""
from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator, Sequence

import numpy as np

from .models import ChunkRow, DocumentRow

SCHEMA = """
CREATE TABLE IF NOT EXISTS documents (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    sha256 TEXT NOT NULL UNIQUE,
    ext TEXT NOT NULL,
    stored_name TEXT NOT NULL,
    size_bytes INTEGER NOT NULL,
    n_pages INTEGER,
    n_chunks INTEGER NOT NULL DEFAULT 0,
    added_at TEXT NOT NULL DEFAULT (datetime('now'))
);
CREATE TABLE IF NOT EXISTS chunks (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    doc_id INTEGER NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    chunk_index INTEGER NOT NULL,
    page INTEGER,
    text TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_chunks_doc ON chunks(doc_id);
CREATE TABLE IF NOT EXISTS embeddings (
    chunk_id INTEGER PRIMARY KEY REFERENCES chunks(id) ON DELETE CASCADE,
    vec BLOB NOT NULL
);
CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
    text, content='chunks', content_rowid='id', tokenize='porter unicode61'
);
CREATE TRIGGER IF NOT EXISTS chunks_ai AFTER INSERT ON chunks BEGIN
    INSERT INTO chunks_fts(rowid, text) VALUES (new.id, new.text);
END;
CREATE TRIGGER IF NOT EXISTS chunks_ad AFTER DELETE ON chunks BEGIN
    INSERT INTO chunks_fts(chunks_fts, rowid, text) VALUES ('delete', old.id, old.text);
END;
CREATE TABLE IF NOT EXISTS history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    query TEXT NOT NULL,
    mode TEXT NOT NULL,
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);
CREATE TABLE IF NOT EXISTS meta (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
"""

_CHUNK_SELECT = (
    "SELECT c.id, c.doc_id, d.name AS doc_name, c.chunk_index, c.page, c.text "
    "FROM chunks c JOIN documents d ON d.id = c.doc_id"
)


def _chunk(row: sqlite3.Row) -> ChunkRow:
    return ChunkRow(row["id"], row["doc_id"], row["doc_name"], row["chunk_index"], row["page"], row["text"])


class Database:
    """Thin wrapper. A fresh connection is opened per call, which keeps it thread-safe."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.connect() as conn:
            conn.executescript(SCHEMA)
            conn.execute("PRAGMA journal_mode=WAL")

    @contextmanager
    def connect(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self.path, timeout=30)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys=ON")
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    # ---- documents -------------------------------------------------
    def find_by_hash(self, sha256: str) -> DocumentRow | None:
        with self.connect() as conn:
            row = conn.execute(self._doc_sql("WHERE d.sha256 = ?"), (sha256,)).fetchone()
        return self._doc(row) if row else None

    def get_document(self, doc_id: int) -> DocumentRow | None:
        with self.connect() as conn:
            row = conn.execute(self._doc_sql("WHERE d.id = ?"), (doc_id,)).fetchone()
        return self._doc(row) if row else None

    def list_documents(self) -> list[DocumentRow]:
        with self.connect() as conn:
            rows = conn.execute(self._doc_sql("ORDER BY d.added_at DESC, d.id DESC")).fetchall()
        return [self._doc(r) for r in rows]

    @staticmethod
    def _doc_sql(tail: str) -> str:
        return (
            "SELECT d.id, d.name, d.sha256, d.ext, d.stored_name, d.size_bytes, d.n_pages, "
            "(SELECT COUNT(*) FROM chunks c WHERE c.doc_id = d.id) AS n_chunks, d.added_at "
            f"FROM documents d {tail}"
        )

    @staticmethod
    def _doc(r: sqlite3.Row) -> DocumentRow:
        return DocumentRow(r["id"], r["name"], r["sha256"], r["ext"], r["stored_name"],
                           r["size_bytes"], r["n_pages"], r["n_chunks"], r["added_at"])

    def add_document(self, *, name: str, sha256: str, ext: str, stored_name: str, size_bytes: int,
                     n_pages: int | None, chunks: Sequence[tuple[int | None, str]],
                     vectors: np.ndarray) -> tuple[int, list[int]]:
        """Insert a document, its chunks and embeddings in ONE transaction."""
        if len(chunks) != len(vectors):
            raise ValueError("chunks and vectors must have the same length")
        with self.connect() as conn:
            cur = conn.execute(
                "INSERT INTO documents(name, sha256, ext, stored_name, size_bytes, n_pages, n_chunks) "
                "VALUES (?,?,?,?,?,?,?)",
                (name, sha256, ext, stored_name, size_bytes, n_pages, len(chunks)),
            )
            doc_id = int(cur.lastrowid)
            chunk_ids = self._insert_chunks(conn, doc_id, chunks, vectors)
        return doc_id, chunk_ids

    def _insert_chunks(self, conn: sqlite3.Connection, doc_id: int,
                       chunks: Sequence[tuple[int | None, str]], vectors: np.ndarray) -> list[int]:
        ids: list[int] = []
        for index, ((page, text), vec) in enumerate(zip(chunks, vectors)):
            cur = conn.execute(
                "INSERT INTO chunks(doc_id, chunk_index, page, text) VALUES (?,?,?,?)",
                (doc_id, index, page, text),
            )
            chunk_id = int(cur.lastrowid)
            conn.execute("INSERT INTO embeddings(chunk_id, vec) VALUES (?,?)",
                         (chunk_id, np.asarray(vec, dtype=np.float32).tobytes()))
            ids.append(chunk_id)
        return ids

    def replace_document_chunks(self, doc_id: int, chunks: Sequence[tuple[int | None, str]],
                                vectors: np.ndarray, n_pages: int | None) -> list[int]:
        with self.connect() as conn:
            conn.execute("DELETE FROM chunks WHERE doc_id = ?", (doc_id,))  # fires FTS trigger
            ids = self._insert_chunks(conn, doc_id, chunks, vectors)
            conn.execute("UPDATE documents SET n_chunks = ?, n_pages = ? WHERE id = ?",
                         (len(chunks), n_pages, doc_id))
        return ids

    def delete_document(self, doc_id: int) -> tuple[list[int], str | None]:
        """Remove a document and everything indexed from it. Returns (chunk ids, stored file name)."""
        with self.connect() as conn:
            row = conn.execute("SELECT stored_name FROM documents WHERE id = ?", (doc_id,)).fetchone()
            if row is None:
                return [], None
            ids = [r["id"] for r in conn.execute("SELECT id FROM chunks WHERE doc_id = ?", (doc_id,))]
            conn.execute("DELETE FROM chunks WHERE doc_id = ?", (doc_id,))
            conn.execute("DELETE FROM documents WHERE id = ?", (doc_id,))
        return ids, row["stored_name"]

    # ---- chunks & embeddings --------------------------------------
    def get_chunks(self, ids: Sequence[int]) -> dict[int, ChunkRow]:
        ids = list(dict.fromkeys(int(i) for i in ids))
        out: dict[int, ChunkRow] = {}
        with self.connect() as conn:
            for start in range(0, len(ids), 500):
                part = ids[start:start + 500]
                marks = ",".join("?" * len(part))
                for row in conn.execute(f"{_CHUNK_SELECT} WHERE c.id IN ({marks})", part):
                    out[row["id"]] = _chunk(row)
        return out

    def get_document_chunks(self, doc_id: int) -> list[ChunkRow]:
        with self.connect() as conn:
            rows = conn.execute(f"{_CHUNK_SELECT} WHERE c.doc_id = ? ORDER BY c.chunk_index", (doc_id,)).fetchall()
        return [_chunk(r) for r in rows]

    def all_chunk_texts(self) -> tuple[list[int], list[str]]:
        with self.connect() as conn:
            rows = conn.execute("SELECT id, text FROM chunks ORDER BY id").fetchall()
        return [r["id"] for r in rows], [r["text"] for r in rows]

    def set_embeddings(self, ids: Sequence[int], vectors: np.ndarray) -> None:
        with self.connect() as conn:
            conn.executemany(
                "INSERT OR REPLACE INTO embeddings(chunk_id, vec) VALUES (?,?)",
                [(int(i), np.asarray(v, dtype=np.float32).tobytes()) for i, v in zip(ids, vectors)],
            )

    def load_embeddings(self, ids: Sequence[int] | None = None) -> tuple[list[int], np.ndarray]:
        with self.connect() as conn:
            if ids is None:
                rows = conn.execute("SELECT chunk_id, vec FROM embeddings ORDER BY chunk_id").fetchall()
            else:
                rows = []
                ids = list(ids)
                for start in range(0, len(ids), 500):
                    part = ids[start:start + 500]
                    marks = ",".join("?" * len(part))
                    rows += conn.execute(
                        f"SELECT chunk_id, vec FROM embeddings WHERE chunk_id IN ({marks})", part).fetchall()
        if not rows:
            return [], np.empty((0, 0), dtype=np.float32)
        out_ids = [r["chunk_id"] for r in rows]
        matrix = np.vstack([np.frombuffer(r["vec"], dtype=np.float32) for r in rows])
        return out_ids, matrix

    # ---- keyword search -------------------------------------------
    def fts_search(self, match_query: str, limit: int) -> list[tuple[int, float]]:
        """BM25-ranked FTS5 search. Returns [(chunk_id, bm25_score)] best first (higher = better)."""
        if not match_query:
            return []
        with self.connect() as conn:
            rows = conn.execute(
                "SELECT rowid, bm25(chunks_fts) AS s FROM chunks_fts WHERE chunks_fts MATCH ? "
                "ORDER BY s LIMIT ?", (match_query, limit)).fetchall()
        return [(r["rowid"], -float(r["s"])) for r in rows]

    def fts_count(self, match_query: str) -> int:
        with self.connect() as conn:
            row = conn.execute("SELECT COUNT(*) AS n FROM chunks_fts WHERE chunks_fts MATCH ?",
                               (match_query,)).fetchone()
        return int(row["n"])

    # ---- history & prefs ------------------------------------------
    def add_history(self, query: str, mode: str, keep: int = 100) -> None:
        with self.connect() as conn:
            conn.execute("DELETE FROM history WHERE query = ? AND mode = ?", (query, mode))
            conn.execute("INSERT INTO history(query, mode) VALUES (?,?)", (query, mode))
            conn.execute("DELETE FROM history WHERE id NOT IN "
                         "(SELECT id FROM history ORDER BY id DESC LIMIT ?)", (keep,))

    def recent_history(self, limit: int = 8) -> list[tuple[str, str]]:
        with self.connect() as conn:
            rows = conn.execute("SELECT query, mode FROM history ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
        return [(r["query"], r["mode"]) for r in rows]

    def clear_history(self) -> None:
        with self.connect() as conn:
            conn.execute("DELETE FROM history")

    def get_meta(self, key: str, default: str | None = None) -> str | None:
        with self.connect() as conn:
            row = conn.execute("SELECT value FROM meta WHERE key = ?", (key,)).fetchone()
        return row["value"] if row else default

    def set_meta(self, key: str, value: str) -> None:
        with self.connect() as conn:
            conn.execute("INSERT OR REPLACE INTO meta(key, value) VALUES (?,?)", (key, str(value)))

    def stats(self) -> dict[str, int]:
        with self.connect() as conn:
            docs = conn.execute("SELECT COUNT(*) AS n FROM documents").fetchone()["n"]
            chunks = conn.execute("SELECT COUNT(*) AS n FROM chunks").fetchone()["n"]
            words = conn.execute("SELECT COALESCE(SUM(size_bytes),0) AS n FROM documents").fetchone()["n"]
        return {"documents": docs, "chunks": chunks, "bytes": words}

    def wipe(self) -> None:
        """Delete every document, chunk, embedding and history row."""
        with self.connect() as conn:
            conn.execute("DELETE FROM chunks")
            conn.execute("DELETE FROM documents")
            conn.execute("DELETE FROM history")
            conn.execute("INSERT INTO chunks_fts(chunks_fts) VALUES ('rebuild')")
