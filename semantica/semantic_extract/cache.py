"""
Result Caching Module

This module provides caching mechanisms for extraction results to avoid redundant
computations and API calls. It implements an LRU (Least Recently Used) cache
with Time-To-Live (TTL) support.

Key Features:
    - LRU Caching: Evicts least recently used items when cache is full
    - TTL Support: Expires items after a configurable duration
    - Namespaced Caching: Separate caches for entities, relations, and triplets
    - Hash-based Keys: Uses stable hashing for text and parameters
    - Pluggable Backends: In-memory (default) or persistent (sqlite), selected
      without changing the ``ExtractionCache`` public API

Classes:
    - ExtractionCache: Main cache manager (owns key derivation + public API)
    - CacheBackend: Storage backend interface (owns persistence/expiry/eviction)
    - InMemoryBackend: Thread-safe in-memory LRU + TTL backend (historical default)
    - SqliteCacheBackend: Persistent backend backed by the stdlib ``sqlite3``
    - CacheItem: Container for cached data with metadata

Author: Semantica Contributors
License: MIT
"""

import time
import hashlib
import json
import pickle
import sqlite3
import threading
from abc import ABC, abstractmethod
from collections import OrderedDict
from pathlib import Path
from typing import Any, Callable, Dict, Optional
from threading import Lock

from ..utils.logging import get_logger

# The namespaces ExtractionCache manages. Kept as a module constant so backends
# and get_stats() agree on the set without hard-coding it in several places.
NAMESPACES = ("entities", "relations", "triplets")


class CacheItem:
    """Container for cached data."""
    def __init__(self, value: Any, ttl: Optional[int] = None):
        self.value = value
        self.timestamp = time.time()
        self.ttl = ttl

    def is_expired(self) -> bool:
        """Check if item has expired."""
        if self.ttl is None:
            return False
        return time.time() - self.timestamp > self.ttl


class CacheBackend(ABC):
    """Storage backend for :class:`ExtractionCache`.

    A backend owns persistence, expiry and eviction for opaque
    ``(namespace, key) -> value`` entries. ``ExtractionCache`` owns key
    derivation (the stable SHA-256 hash over text + params) and the public
    API, so backends never see raw text or generation parameters — only the
    already-hashed key. This keeps the key policy in one place and lets a
    backend be swapped (in-memory vs persistent) with no behavioral change to
    callers.
    """

    @abstractmethod
    def get(self, namespace: str, key: str) -> Optional[Any]:
        """Return the cached value or ``None`` if missing/expired."""

    @abstractmethod
    def set(self, namespace: str, key: str, value: Any, ttl: Optional[int]) -> None:
        """Store ``value`` under ``(namespace, key)`` with an optional TTL (seconds)."""

    @abstractmethod
    def clear(self, namespace: Optional[str] = None) -> None:
        """Clear one namespace, or all namespaces when ``namespace`` is ``None``."""

    @abstractmethod
    def size(self, namespace: str) -> int:
        """Return the number of stored entries in ``namespace``."""


class InMemoryBackend(CacheBackend):
    """Thread-safe in-memory LRU + TTL backend — the historical default.

    Preserves the original storage layout: one ``OrderedDict`` per namespace
    guarded by its own lock, LRU eviction at ``max_size``, and per-item TTL
    expiry checked lazily on read.
    """

    def __init__(self, max_size: int = 1000):
        self.max_size = max_size
        self._caches: Dict[str, "OrderedDict[str, CacheItem]"] = {
            ns: OrderedDict() for ns in NAMESPACES
        }
        self._locks: Dict[str, Lock] = {ns: Lock() for ns in NAMESPACES}

    def get(self, namespace: str, key: str) -> Optional[Any]:
        if namespace not in self._caches:
            return None
        with self._locks[namespace]:
            cache = self._caches[namespace]
            item = cache.get(key)
            if item is None:
                return None
            if item.is_expired():
                del cache[key]
                return None
            cache.move_to_end(key)  # mark as recently used
            return item.value

    def set(self, namespace: str, key: str, value: Any, ttl: Optional[int]) -> None:
        if namespace not in self._caches:
            return
        with self._locks[namespace]:
            cache = self._caches[namespace]
            if key in cache:
                cache.move_to_end(key)
            cache[key] = CacheItem(value, ttl)
            if len(cache) > self.max_size:
                cache.popitem(last=False)  # evict least recently used

    def clear(self, namespace: Optional[str] = None) -> None:
        if namespace is not None:
            if namespace in self._caches:
                with self._locks[namespace]:
                    self._caches[namespace].clear()
        else:
            for ns in self._caches:
                with self._locks[ns]:
                    self._caches[ns].clear()

    def size(self, namespace: str) -> int:
        cache = self._caches.get(namespace)
        return len(cache) if cache is not None else 0


class SqliteCacheBackend(CacheBackend):
    """Persistent cache backend backed by the standard-library ``sqlite3``.

    Unlike :class:`InMemoryBackend`, entries survive a process restart, so a
    fresh process (CI job, notebook kernel, batch worker, extraction
    subprocess) reuses previous results instead of re-paying every LLM call.
    Values are serialized with ``serializer`` (``pickle`` by default), so point
    ``db_path`` at a **trusted, local** location — a persistent cache file is
    deserialized back into the process on read. Provide ``serializer`` /
    ``deserializer`` (e.g. ``json.dumps`` over plain dicts) to avoid pickle.

    TTL and LRU (by last access) mirror :class:`InMemoryBackend`. All access is
    guarded by a lock and the connection uses WAL mode for concurrent readers.
    """

    def __init__(
        self,
        db_path: str,
        max_size: int = 1000,
        serializer: Callable[[Any], bytes] = pickle.dumps,
        deserializer: Callable[[bytes], Any] = pickle.loads,
    ):
        self.db_path = str(db_path)
        self.max_size = max_size
        self._serialize = serializer
        self._deserialize = deserializer
        self._lock = threading.Lock()
        self.logger = get_logger("extraction_cache.sqlite")

        if self.db_path != ":memory:":
            Path(self.db_path).expanduser().resolve().parent.mkdir(
                parents=True, exist_ok=True
            )
        self._conn = sqlite3.connect(self.db_path, check_same_thread=False)
        try:
            self._conn.execute("PRAGMA journal_mode=WAL")
        except sqlite3.DatabaseError:  # e.g. :memory: does not support WAL
            pass
        self._conn.execute(
            "CREATE TABLE IF NOT EXISTS cache ("
            "  namespace TEXT NOT NULL,"
            "  key TEXT NOT NULL,"
            "  value BLOB NOT NULL,"
            "  expires_at REAL,"
            "  last_access REAL NOT NULL,"
            "  PRIMARY KEY (namespace, key)"
            ")"
        )
        self._conn.commit()

    def get(self, namespace: str, key: str) -> Optional[Any]:
        now = time.time()
        with self._lock:
            row = self._conn.execute(
                "SELECT value, expires_at FROM cache WHERE namespace=? AND key=?",
                (namespace, key),
            ).fetchone()
            if row is None:
                return None
            value_blob, expires_at = row
            if expires_at is not None and expires_at < now:
                self._conn.execute(
                    "DELETE FROM cache WHERE namespace=? AND key=?", (namespace, key)
                )
                self._conn.commit()
                return None
            self._conn.execute(
                "UPDATE cache SET last_access=? WHERE namespace=? AND key=?",
                (now, namespace, key),
            )
            self._conn.commit()
        try:
            return self._deserialize(value_blob)
        except Exception as exc:  # corrupt/incompatible payload — treat as a miss
            self.logger.warning(f"Failed to deserialize cached value: {exc}")
            return None

    def set(self, namespace: str, key: str, value: Any, ttl: Optional[int]) -> None:
        now = time.time()
        expires_at = now + ttl if ttl else None
        try:
            blob = self._serialize(value)
        except Exception as exc:  # unserializable value — skip caching, never crash
            self.logger.warning(f"Failed to serialize value for caching: {exc}")
            return
        with self._lock:
            self._conn.execute(
                "INSERT INTO cache (namespace, key, value, expires_at, last_access) "
                "VALUES (?, ?, ?, ?, ?) "
                "ON CONFLICT(namespace, key) DO UPDATE SET "
                "  value=excluded.value,"
                "  expires_at=excluded.expires_at,"
                "  last_access=excluded.last_access",
                (namespace, key, blob, expires_at, now),
            )
            count = self._conn.execute(
                "SELECT COUNT(*) FROM cache WHERE namespace=?", (namespace,)
            ).fetchone()[0]
            if count > self.max_size:
                self._conn.execute(
                    "DELETE FROM cache WHERE rowid IN ("
                    "  SELECT rowid FROM cache WHERE namespace=? "
                    "  ORDER BY last_access ASC LIMIT ?"
                    ")",
                    (namespace, count - self.max_size),
                )
            self._conn.commit()

    def clear(self, namespace: Optional[str] = None) -> None:
        with self._lock:
            if namespace is None:
                self._conn.execute("DELETE FROM cache")
            else:
                self._conn.execute("DELETE FROM cache WHERE namespace=?", (namespace,))
            self._conn.commit()

    def size(self, namespace: str) -> int:
        with self._lock:
            row = self._conn.execute(
                "SELECT COUNT(*) FROM cache WHERE namespace=?", (namespace,)
            ).fetchone()
        return int(row[0]) if row else 0

    def close(self) -> None:
        """Close the underlying connection (safe to call more than once)."""
        with self._lock:
            try:
                self._conn.close()
            except sqlite3.Error:
                pass


class ExtractionCache:
    """
    LRU Cache for extraction results.

    Owns stable key derivation and the public ``get``/``set``/``clear`` API;
    delegates storage to a pluggable :class:`CacheBackend`. Defaults to a
    thread-safe in-memory backend, preserving the original behavior. Pass a
    persistent backend (e.g. :class:`SqliteCacheBackend`) to survive restarts.
    """
    def __init__(
        self,
        max_size: int = 1000,
        ttl: int = 3600,
        backend: Optional[CacheBackend] = None,
    ):
        """
        Initialize the cache.

        Args:
            max_size: Maximum number of items to store per namespace (used by
                the default in-memory backend)
            ttl: Time to live in seconds (default 1 hour)
            backend: Storage backend. Defaults to :class:`InMemoryBackend`.
        """
        self.max_size = max_size
        self.ttl = ttl
        self.logger = get_logger("extraction_cache")
        self.enabled = True
        self._backend: CacheBackend = (
            backend if backend is not None else InMemoryBackend(max_size=max_size)
        )

    # Backward-compat: some callers/tests reach into the historical internal
    # attributes. Proxy them to the backend when it exposes them (the default
    # in-memory backend does); otherwise present empty mappings.
    @property
    def _caches(self) -> Dict[str, Any]:
        return getattr(self._backend, "_caches", {})

    @property
    def _locks(self) -> Dict[str, Any]:
        return getattr(self._backend, "_locks", {})

    def _generate_key(self, text: str, **params) -> str:
        """
        Generate a stable cache key based on text and parameters.

        Note: Sensitive parameters like 'api_key' are excluded from the cache key
        to prevent security risks and ensure cache sharing where appropriate.
        """
        # Filter out sensitive keys
        sensitive_keys = {'api_key', 'token', 'password', 'secret', 'auth', 'authorization'}
        filtered_params = {k: v for k, v in params.items() if k.lower() not in sensitive_keys}

        # Create a stable string representation of params
        # Sort keys to ensure consistent ordering
        param_str = json.dumps(filtered_params, sort_keys=True, default=str)

        # Combine text and params
        content = f"{text}|{param_str}"

        # Return hash (SHA-256 for better security than MD5)
        return hashlib.sha256(content.encode('utf-8')).hexdigest()

    def get(self, namespace: str, text: str, **params) -> Optional[Any]:
        """
        Retrieve item from cache.

        Args:
            namespace: Cache namespace ("entities", "relations", "triplets")
            text: Input text used for extraction
            **params: Extraction parameters used

        Returns:
            Cached result or None if not found/expired
        """
        if not self.enabled:
            return None

        if namespace not in NAMESPACES:
            return None

        key = self._generate_key(text, **params)
        return self._backend.get(namespace, key)

    def set(self, namespace: str, text: str, value: Any, **params) -> None:
        """
        Add item to cache.

        Args:
            namespace: Cache namespace
            text: Input text
            value: Result to cache
            **params: Extraction parameters
        """
        if not self.enabled:
            return

        if namespace not in NAMESPACES:
            self.logger.warning(f"Unknown cache namespace: {namespace}")
            return

        key = self._generate_key(text, **params)
        self._backend.set(namespace, key, value, self.ttl)

    def clear(self, namespace: Optional[str] = None):
        """Clear cache(s)."""
        self._backend.clear(namespace)

    def get_stats(self) -> Dict[str, Dict[str, int]]:
        """Get cache statistics."""
        return {
            ns: {"size": self._backend.size(ns), "max_size": self.max_size}
            for ns in NAMESPACES
        }

# Global cache instance
extraction_cache = ExtractionCache()
