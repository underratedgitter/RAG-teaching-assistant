"""Tests for embedding caching and the parallel batch path.

Order is the dangerous part here: retrieval maps embeddings to chunks by
position, so a reordered result silently attaches every answer to the wrong
timestamp.
"""
import os
import sys
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).parent.parent


def load_module(monkeypatch, model="nomic-embed-text"):
    """Load preprocess_json's helpers without running its pipeline."""
    src = (ROOT / "preprocess_json.py").read_text()
    head = src[: src.index('print("="*50)')]
    ns = types.ModuleType("pp").__dict__
    monkeypatch.setenv("EMBED_MODEL", model)
    exec(compile(head, "preprocess_json.py", "exec"), ns)
    return ns


# ── cache key identity ──────────────────────────────────────────────────────

def test_same_text_gives_same_key(monkeypatch):
    m = load_module(monkeypatch)
    assert m["cache_key"](["a", "b"]) == m["cache_key"](["a", "b"])


def test_changed_text_invalidates(monkeypatch):
    m = load_module(monkeypatch)
    assert m["cache_key"](["a", "b"]) != m["cache_key"](["a", "b!"])


def test_reordered_text_invalidates(monkeypatch):
    m = load_module(monkeypatch)
    assert m["cache_key"](["a", "b"]) != m["cache_key"](["b", "a"])


def test_different_model_invalidates(monkeypatch):
    """Embeddings from one model must never be reused for another."""
    a = load_module(monkeypatch, model="nomic-embed-text")["cache_key"](["x"])
    b = load_module(monkeypatch, model="mxbai-embed-large")["cache_key"](["x"])
    assert a != b


def test_concatenation_cannot_collide(monkeypatch):
    """['ab','c'] and ['a','bc'] must not hash the same."""
    m = load_module(monkeypatch)
    assert m["cache_key"](["ab", "c"]) != m["cache_key"](["a", "bc"])


# ── cache round-trip ────────────────────────────────────────────────────────

def test_cache_round_trip(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    m = load_module(monkeypatch)
    texts = ["one", "two"]
    vecs = [[0.1, 0.2], [0.3, 0.4]]

    assert m["load_cached"]("f.json", texts) is None      # cold
    m["save_cached"]("f.json", texts, vecs)
    assert m["load_cached"]("f.json", texts) == vecs      # warm


def test_cache_misses_when_text_changes(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    m = load_module(monkeypatch)
    m["save_cached"]("f.json", ["one"], [[0.1]])
    assert m["load_cached"]("f.json", ["one edited"]) is None


def test_corrupt_cache_is_a_miss_not_a_crash(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    m = load_module(monkeypatch)
    os.makedirs(m["CACHE_DIR"], exist_ok=True)
    Path(m["CACHE_DIR"], "f.json.joblib").write_bytes(b"not a joblib file")
    assert m["load_cached"]("f.json", ["one"]) is None


def test_cache_write_is_atomic(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    m = load_module(monkeypatch)
    m["save_cached"]("f.json", ["one"], [[0.1]])
    leftovers = [p for p in Path(m["CACHE_DIR"]).iterdir() if p.name.endswith(".partial")]
    assert not leftovers, "a .partial file was left behind"


# ── parallel batching must preserve order ───────────────────────────────────

def test_parallel_batches_come_back_in_order(monkeypatch):
    m = load_module(monkeypatch)
    import time as _t

    def fake_embed(batch):
        # Finish out of order on purpose: later batches return first.
        _t.sleep(0.02 if batch[0].startswith("a") else 0.001)
        return [[float(ord(t[0])), len(t)] for t in batch]

    m["_embed_batch"] = fake_embed
    texts = [f"{c}{i}" for c in "abcde" for i in range(4)]   # 20 texts
    got = m["create_embedding"](texts, batch_size=4)

    assert len(got) == len(texts)
    expected = [[float(ord(t[0])), len(t)] for t in texts]
    assert got == expected, "batches were reassembled out of order"


def test_single_batch_path_still_works(monkeypatch):
    m = load_module(monkeypatch)
    m["_embed_batch"] = lambda b: [[1.0] for _ in b]
    assert m["create_embedding"](["x", "y"], batch_size=10) == [[1.0], [1.0]]


def test_empty_input_returns_empty(monkeypatch):
    m = load_module(monkeypatch)
    assert m["create_embedding"]([]) == []


def test_a_failing_batch_propagates(monkeypatch):
    """A silent partial result would misalign every later chunk."""
    m = load_module(monkeypatch)

    def boom(batch):
        if batch[0] == "c0":
            raise RuntimeError("ollama died")
        return [[0.0] for _ in batch]

    m["_embed_batch"] = boom
    with pytest.raises(RuntimeError):
        m["create_embedding"]([f"{c}0" for c in "abcde"], batch_size=1)
