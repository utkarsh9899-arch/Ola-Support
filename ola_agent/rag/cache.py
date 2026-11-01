"""In-memory response cache for the grounded-generation step (Task 16).

Keyed by normalised query text (lower-case, punctuation removed, whitespace collapsed) together with the
collection, k and threshold, so a changed retrieval setting never serves a stale answer. Only grounded
generation is cached; live ticket lookups are never cached. Callers pass already-masked text, so no raw
phone number is ever a cache key. Cleared whenever the knowledge base changes.
"""
from __future__ import annotations

import re
import threading
from typing import Any


def normalize_query(query: str) -> str:
    text = re.sub(r"[^\w\s]", " ", query.lower())
    return re.sub(r"\s+", " ", text).strip()


class GroundedCache:
    def __init__(self) -> None:
        self._data: dict[tuple, Any] = {}
        self._lock = threading.Lock()
        self.hits = 0
        self.misses = 0

    @staticmethod
    def key(query: str, collection: str, k: int, threshold: float) -> tuple:
        return (collection, k, threshold, normalize_query(query))

    def get(self, key: tuple) -> Any | None:
        with self._lock:
            value = self._data.get(key)
            if value is None:
                self.misses += 1
            else:
                self.hits += 1
            return value

    def put(self, key: tuple, value: Any) -> None:
        with self._lock:
            self._data[key] = value

    def clear(self) -> None:
        with self._lock:
            self._data.clear()

    def __len__(self) -> int:
        return len(self._data)


GROUNDED_CACHE = GroundedCache()
