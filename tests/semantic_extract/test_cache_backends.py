"""Unit tests for ExtractionCache pluggable backends (in-memory + sqlite)."""

from __future__ import annotations

import json

from semantica.semantic_extract import (
    CacheBackend,
    ExtractionCache,
    InMemoryBackend,
    SqliteCacheBackend,
)


# ---------------------------------------------------------------------------
# ExtractionCache + default in-memory backend (behavior parity)
# ---------------------------------------------------------------------------

def test_default_backend_is_in_memory():
    cache = ExtractionCache()
    assert isinstance(cache._backend, InMemoryBackend)
    assert isinstance(cache._backend, CacheBackend)


def test_get_set_roundtrip():
    cache = ExtractionCache()
    assert cache.get("entities", "hello") is None
    cache.set("entities", "hello", [{"name": "X"}])
    assert cache.get("entities", "hello") == [{"name": "X"}]


def test_sensitive_params_excluded_from_key():
    cache = ExtractionCache()
    cache.set("entities", "hi", ["v"], provider="openai", api_key="secret-1")
    # A different api_key/token must not change the key -> still a hit.
    assert cache.get("entities", "hi", provider="openai", api_key="secret-2") == ["v"]


def test_different_params_produce_different_keys():
    cache = ExtractionCache()
    cache.set("entities", "hi", ["a"], model="gpt-4")
    assert cache.get("entities", "hi", model="gpt-4o") is None
    assert cache.get("entities", "hi", model="gpt-4") == ["a"]


def test_disabled_cache_is_noop():
    cache = ExtractionCache()
    cache.enabled = False
    cache.set("entities", "hi", ["a"])
    assert cache.get("entities", "hi") is None


def test_unknown_namespace_is_ignored():
    cache = ExtractionCache()
    cache.set("nope", "hi", ["a"])  # warns, stores nothing
    assert cache.get("nope", "hi") is None


def test_get_stats_structure():
    cache = ExtractionCache(max_size=42)
    cache.set("entities", "a", [1])
    stats = cache.get_stats()
    assert set(stats) == {"entities", "relations", "triplets"}
    assert stats["entities"] == {"size": 1, "max_size": 42}


def test_backward_compat_caches_proxy():
    # Existing callers/tests reach into ._caches / ._locks; keep that working.
    cache = ExtractionCache()
    cache.set("entities", "a", [1])
    assert "entities" in cache._caches
    assert len(cache._caches["entities"]) == 1
    assert "entities" in cache._locks
    cache._caches["entities"].clear()
    assert cache.get_stats()["entities"]["size"] == 0


# ---------------------------------------------------------------------------
# InMemoryBackend directly
# ---------------------------------------------------------------------------

def test_in_memory_ttl_expiry():
    backend = InMemoryBackend()
    backend.set("entities", "k", "v", ttl=-1)  # already expired
    assert backend.get("entities", "k") is None


def test_in_memory_ttl_none_never_expires():
    backend = InMemoryBackend()
    backend.set("entities", "k", "v", ttl=None)
    assert backend.get("entities", "k") == "v"


def test_in_memory_lru_eviction():
    backend = InMemoryBackend(max_size=2)
    backend.set("entities", "a", 1, ttl=None)
    backend.set("entities", "b", 2, ttl=None)
    backend.get("entities", "a")           # 'a' now most-recently-used
    backend.set("entities", "c", 3, ttl=None)  # evicts LRU -> 'b'
    assert backend.size("entities") == 2
    assert backend.get("entities", "b") is None
    assert backend.get("entities", "a") == 1
    assert backend.get("entities", "c") == 3


# ---------------------------------------------------------------------------
# SqliteCacheBackend
# ---------------------------------------------------------------------------

def _db(tmp_path):
    return str(tmp_path / "cache.sqlite3")


def test_sqlite_roundtrip_and_size(tmp_path):
    backend = SqliteCacheBackend(_db(tmp_path))
    assert backend.get("entities", "k") is None
    backend.set("entities", "k", {"x": 1}, ttl=None)
    assert backend.get("entities", "k") == {"x": 1}
    assert backend.size("entities") == 1
    backend.close()


def test_sqlite_persists_across_restart(tmp_path):
    path = _db(tmp_path)
    cache = ExtractionCache(backend=SqliteCacheBackend(path))
    cache.set("entities", "hello", [{"name": "X"}], provider="openai", model="gpt-4")
    cache._backend.close()

    # Simulate a fresh process: new backend + new cache over the same file.
    reopened = ExtractionCache(backend=SqliteCacheBackend(path))
    assert reopened.get("entities", "hello", provider="openai", model="gpt-4") == [
        {"name": "X"}
    ]
    reopened._backend.close()


def test_sqlite_ttl_expiry(tmp_path):
    backend = SqliteCacheBackend(_db(tmp_path))
    backend.set("entities", "k", "v", ttl=-1)  # already expired
    assert backend.get("entities", "k") is None
    backend.close()


def test_sqlite_lru_eviction_bounds_size(tmp_path):
    backend = SqliteCacheBackend(_db(tmp_path), max_size=2)
    for k in ("a", "b", "c"):
        backend.set("entities", k, k, ttl=None)
    assert backend.size("entities") == 2
    backend.close()


def test_sqlite_clear(tmp_path):
    backend = SqliteCacheBackend(_db(tmp_path))
    backend.set("entities", "a", 1, ttl=None)
    backend.set("relations", "b", 2, ttl=None)
    backend.clear("entities")
    assert backend.size("entities") == 0
    assert backend.size("relations") == 1
    backend.clear()
    assert backend.size("relations") == 0
    backend.close()


def test_sqlite_custom_json_serializer(tmp_path):
    backend = SqliteCacheBackend(
        _db(tmp_path),
        serializer=lambda v: json.dumps(v).encode("utf-8"),
        deserializer=lambda b: json.loads(b.decode("utf-8")),
    )
    backend.set("entities", "k", {"a": [1, 2, 3]}, ttl=None)
    assert backend.get("entities", "k") == {"a": [1, 2, 3]}
    backend.close()


def test_sqlite_unserializable_value_is_skipped(tmp_path):
    backend = SqliteCacheBackend(
        _db(tmp_path),
        serializer=lambda v: json.dumps(v).encode("utf-8"),  # can't encode a set
        deserializer=lambda b: json.loads(b.decode("utf-8")),
    )
    backend.set("entities", "k", {1, 2, 3}, ttl=None)  # not JSON-serializable
    assert backend.get("entities", "k") is None  # skipped, no crash
    backend.close()


def test_sqlite_corrupt_payload_returns_none(tmp_path):
    # serializer writes bytes the default pickle deserializer can't load.
    backend = SqliteCacheBackend(_db(tmp_path), serializer=lambda v: b"not-a-pickle")
    backend.set("entities", "k", "v", ttl=None)
    assert backend.get("entities", "k") is None
    backend.close()
