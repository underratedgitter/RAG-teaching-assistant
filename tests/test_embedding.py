"""Tests for embedding caching and the parallel batch path.

Order is the dangerous part here: retrieval maps embeddings to chunks by
position, so a reordered result silently attaches every answer to the wrong
timestamp.
"""
import os
import re
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


def function_source(name):
    """The source of one top-level function, for asserting on its text."""
    src = (ROOT / "preprocess_json.py").read_text()
    start = src.index(f"def {name}")
    rest = src[start:]
    end = re.search(r"\n(?=[^\s#])", rest[len(f"def {name}"):])
    body = rest if end is None else rest[: len(f"def {name}") + end.start()]
    return body


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
    got = m["load_cached"]("f.json", texts)
    assert got is not None
    # The cache stores float32, so the vectors come back rounded.
    import numpy as np
    assert np.allclose(np.array(got), np.array(vecs, dtype=np.float32), atol=1e-6)


def test_cache_is_stored_as_npz(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    m = load_module(monkeypatch)
    m["save_cached"]("f.json", ["one"], [[0.1]])
    assert Path(m["CACHE_DIR"], "f.json.npz").exists()


def test_cache_misses_when_text_changes(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    m = load_module(monkeypatch)
    m["save_cached"]("f.json", ["one"], [[0.1]])
    assert m["load_cached"]("f.json", ["one edited"]) is None


def test_corrupt_cache_is_a_miss_not_a_crash(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    m = load_module(monkeypatch)
    os.makedirs(m["CACHE_DIR"], exist_ok=True)
    Path(m["CACHE_DIR"], "f.json.npz").write_bytes(b"not an npz file")
    assert m["load_cached"]("f.json", ["one"]) is None


def test_cache_write_is_atomic(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    m = load_module(monkeypatch)
    m["save_cached"]("f.json", ["one"], [[0.1]])
    leftovers = [p for p in Path(m["CACHE_DIR"]).iterdir() if p.name.endswith(".partial")]
    assert not leftovers, "a .partial file was left behind"


def test_empty_texts_do_not_crash_the_cache(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    m = load_module(monkeypatch)
    m["save_cached"]("f.json", [], [])
    assert m["load_cached"]("f.json", []) == []


# ── the cache is never unpickled ────────────────────────────────────────────

def test_cache_is_read_without_pickle():
    """A cached file must not be able to execute code when it is read."""
    assert "allow_pickle=False" in function_source("load_cached")


def test_no_joblib_left_in_the_pipeline():
    """joblib.load unpickles, so a tampered artifact would run code."""
    src = (ROOT / "preprocess_json.py").read_text()
    assert "joblib" not in src


# ── task prefixes ───────────────────────────────────────────────────────────

@pytest.mark.parametrize("model", ["nomic-embed-text", "nomic-embed-text:v1.5"])
def test_nomic_models_get_the_document_prefix(monkeypatch, model):
    m = load_module(monkeypatch, model=model)
    assert m["DOCUMENT_PREFIX"] == "search_document: "


def test_other_models_take_raw_text(monkeypatch):
    m = load_module(monkeypatch, model="mxbai-embed-large")
    assert m["DOCUMENT_PREFIX"] == ""


def test_the_prefix_changes_the_cache_key(monkeypatch):
    """Vectors embedded without the prefix must never be reused."""
    m = load_module(monkeypatch)
    assert m["cache_key"](["search_document: a"]) != m["cache_key"](["a"])


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


def test_failed_batch_falls_back_to_one_at_a_time(monkeypatch):
    """A rejected batch is retried per text, and the vectors keep their order."""
    m = load_module(monkeypatch)
    requests = m["requests"]

    class FakeResponse:
        def __init__(self, texts):
            self.texts = texts

        def raise_for_status(self):
            if len(self.texts) > 1:
                raise requests.exceptions.HTTPError("batch too large")

        def json(self):
            return {"embeddings": [[float(len(t))] for t in self.texts]}

    class FakeSession:
        def post(self, url, json, timeout):
            return FakeResponse(json["input"])

    m["session"] = FakeSession()
    assert m["_embed_batch"](["a", "bb", "ccc"]) == [[1.0], [2.0], [3.0]]


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


def test_failed_batch_with_failing_single_text_raises(monkeypatch):
    """When a batch fails and a text still fails alone, _embed_batch raises rather than shortening."""
    m = load_module(monkeypatch)
    requests = m["requests"]

    class FakeResponse:
        def __init__(self, texts):
            self.texts = texts

        def raise_for_status(self):
            if len(self.texts) > 1:
                raise requests.exceptions.HTTPError("batch too large")
            if "bad" in self.texts:
                raise requests.exceptions.HTTPError("invalid chunk")

        def json(self):
            return {"embeddings": [[float(len(t))] for t in self.texts]}

    class FakeSession:
        def post(self, url, json, timeout):
            return FakeResponse(json["input"])

    m["session"] = FakeSession()
    with pytest.raises(requests.exceptions.HTTPError):
        m["_embed_batch"](["a", "bad", "ccc"])

