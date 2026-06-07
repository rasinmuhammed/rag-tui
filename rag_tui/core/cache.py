"""SQLite-backed persistent embedding cache for RAG-TUI.

Caches embeddings by (text_hash, provider, model) so large documents
do not need to be re-embedded on every session or parameter change.
"""

import hashlib
import json
import pickle
import sqlite3
import time
from pathlib import Path
from typing import Optional

import numpy as np

CACHE_DIR = Path.home() / ".rag-tui" / "cache"
_SCHEMA = """
CREATE TABLE IF NOT EXISTS embeddings (
    cache_key   TEXT PRIMARY KEY,
    embedding   BLOB NOT NULL,
    dim         INTEGER NOT NULL,
    created_at  REAL NOT NULL,
    hit_count   INTEGER NOT NULL DEFAULT 0
);
CREATE TABLE IF NOT EXISTS meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
"""


class EmbeddingCache:
    """Persistent SQLite embedding cache.

    Thread-safe via SQLite's WAL mode.  One DB file per (provider, model) pair
    lives in ~/.rag-tui/cache/ so caches are cleanly separated.
    """

    VERSION = "1"

    def __init__(self, provider_name: str, model_name: str) -> None:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        safe = lambda s: "".join(c if c.isalnum() else "_" for c in s)
        db_name = f"{safe(provider_name)}__{safe(model_name)}.db"
        self._path = CACHE_DIR / db_name
        self._conn = self._open()

    def _open(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self._path), check_same_thread=False)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.executescript(_SCHEMA)
        conn.execute(
            "INSERT OR IGNORE INTO meta VALUES ('version', ?)", (self.VERSION,)
        )
        conn.commit()
        return conn

    @staticmethod
    def _key(text: str) -> str:
        return hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()

    def get(self, text: str) -> Optional[np.ndarray]:
        """Return cached embedding or None if not found."""
        row = self._conn.execute(
            "SELECT embedding, dim FROM embeddings WHERE cache_key = ?",
            (self._key(text),),
        ).fetchone()
        if row is None:
            return None
        self._conn.execute(
            "UPDATE embeddings SET hit_count = hit_count + 1 WHERE cache_key = ?",
            (self._key(text),),
        )
        self._conn.commit()
        blob, dim = row
        arr = np.frombuffer(pickle.loads(blob), dtype=np.float32)
        return arr

    def put(self, text: str, embedding: np.ndarray) -> None:
        """Store an embedding in the cache."""
        key = self._key(text)
        blob = pickle.dumps(embedding.astype(np.float32).tobytes())
        self._conn.execute(
            """INSERT OR REPLACE INTO embeddings
               (cache_key, embedding, dim, created_at, hit_count)
               VALUES (?, ?, ?, ?, 0)""",
            (key, blob, len(embedding), time.time()),
        )
        self._conn.commit()

    def get_batch(self, texts: list) -> dict:
        """Return {text: embedding} for all texts found in cache."""
        results = {}
        for text in texts:
            vec = self.get(text)
            if vec is not None:
                results[text] = vec
        return results

    def put_batch(self, texts: list, embeddings: list) -> None:
        """Store multiple embeddings atomically."""
        with self._conn:
            for text, emb in zip(texts, embeddings):
                key = self._key(text)
                arr = np.array(emb, dtype=np.float32)
                blob = pickle.dumps(arr.tobytes())
                self._conn.execute(
                    """INSERT OR REPLACE INTO embeddings
                       (cache_key, embedding, dim, created_at, hit_count)
                       VALUES (?, ?, ?, ?, 0)""",
                    (key, blob, len(arr), time.time()),
                )

    @property
    def size(self) -> int:
        row = self._conn.execute("SELECT COUNT(*) FROM embeddings").fetchone()
        return row[0] if row else 0

    def clear(self) -> None:
        self._conn.execute("DELETE FROM embeddings")
        self._conn.commit()

    def stats(self) -> dict:
        row = self._conn.execute(
            "SELECT COUNT(*), SUM(hit_count) FROM embeddings"
        ).fetchone()
        count, hits = row if row else (0, 0)
        size_bytes = self._path.stat().st_size if self._path.exists() else 0
        return {
            "entries": count,
            "total_hits": hits or 0,
            "size_mb": round(size_bytes / (1024 * 1024), 2),
            "db_path": str(self._path),
        }

    def close(self) -> None:
        self._conn.close()
