"""A 50k-chunk library must load and answer without copying the corpus.

The matrix is normalised once, in place, at load, and a question is one
matrix-vector product: neither may allocate anything the size of the corpus.
Memory is what tracemalloc reports (numpy registers its buffers with it), so
the checks do not depend on how fast the machine is.
"""
import ast
import json
import os
import queue
import re
import secrets
import time
import tracemalloc
from pathlib import Path
from types import MethodType, SimpleNamespace

import numpy as np
import pandas as pd
import requests

import preflight

DASHBOARD = Path(__file__).parents[1] / "dashboard.py"

ROWS = 50_000
DIM = 256            # smaller than nomic-embed-text's 768 to keep the suite light
NEEDLE = 12_345

_CONSTANTS = {
    "TOP_K", "SIMILARITY_THRESHOLD", "FALLBACK_K", "OLLAMA_URL", "EMBED_MODEL",
    "ANSWER_MODEL", "OLLAMA_TIMEOUT", "_USES_TASK_PREFIX", "DOCUMENT_PREFIX", "QUERY_PREFIX",
    "SYSTEM_PROMPT", "GENERATION_OPTIONS", "NUM_PREDICT", "CHARS_PER_TOKEN", "CONTEXT_STEP",
    "EXCERPT_TAG_TOKENS", "MAX_CONTEXT_TOKENS", "PROMPT_OVERHEAD_TOKENS", "_TAG_RE",
}
_FUNCTIONS = {
    "_matrix_matches_dataframe", "_clean_field", "_neutralise_tags", "_time_parts",
    "build_prompt", "format_sources", "_estimate_tokens", "_excerpt_tokens", "_excerpt_budget",
    "_fit_rows", "_num_ctx",
}
_METHODS = {"load_embeddings", "_prepare_matrix", "_answer"}


def _namespace(monkeypatch):
    tree = ast.parse(DASHBOARD.read_text(encoding="utf-8"))
    body = []
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            getattr(t, "id", None) in _CONSTANTS for t in node.targets
        ):
            body.append(node)
        elif isinstance(node, ast.FunctionDef) and node.name in _FUNCTIONS:
            body.append(node)
        elif isinstance(node, ast.ClassDef) and node.name == "RAGDashboard":
            body.extend(c for c in node.body
                        if isinstance(c, ast.FunctionDef) and c.name in _METHODS)
    monkeypatch.setenv("EMBED_MODEL", "nomic-embed-text")
    ns = {"os": os, "json": json, "np": np, "pd": pd, "re": re,
          "secrets": secrets, "requests": requests, "queue": queue,
          "preflight": preflight}
    exec(compile(ast.Module(body=body, type_ignores=[]), str(DASHBOARD), "exec"), ns)
    return ns


def _corpus():
    rng = np.random.default_rng(7)
    matrix = rng.standard_normal((ROWS, DIM), dtype=np.float32)
    # Rows far from unit length, as an embedding model's output may be.
    matrix *= rng.uniform(0.5, 40.0, size=(ROWS, 1)).astype(np.float32)
    return matrix


def _write(tmp_path, matrix):
    np.save(tmp_path / "embedding_matrix.npy", matrix)
    chunks = [
        {"number": "1", "title": "Lecture", "start": i, "end": i + 30,
         "text": "the needle" if i == NEEDLE else f"filler {i}", "chunk_id": i}
        for i in range(ROWS)
    ]
    (tmp_path / "chunks.json").write_text(json.dumps({
        "corpus_id": "c", "model": "nomic-embed-text",
        "document_prefix": "search_document: ", "chunks": chunks,
    }), encoding="utf-8")
    (tmp_path / "embedding_matrix.meta.json").write_text(json.dumps(
        {"corpus_id": "c", "rows": ROWS, "model": "nomic-embed-text"}), encoding="utf-8")


def _peak_extra(fn):
    """Largest amount of memory fn allocated beyond what was live before it."""
    tracemalloc.start()
    try:
        tracemalloc.reset_peak()
        before = tracemalloc.get_traced_memory()[0]
        result = fn()
        return result, tracemalloc.get_traced_memory()[1] - before
    finally:
        tracemalloc.stop()


def _app(ns, tmp_path):
    app = SimpleNamespace(base_dir=str(tmp_path), df=None, embedding_matrix=None)
    app._prepare_matrix = MethodType(ns["_prepare_matrix"], app)
    return app


def test_normalising_a_float32_matrix_takes_no_second_copy(monkeypatch):
    ns = _namespace(monkeypatch)
    matrix = _corpus()
    original = matrix
    app = SimpleNamespace(embedding_matrix=matrix)
    app._prepare_matrix = MethodType(ns["_prepare_matrix"], app)

    _, extra = _peak_extra(app._prepare_matrix)

    assert extra < 0.05 * matrix.nbytes
    assert app.embedding_matrix is original
    assert app.embedding_matrix.dtype == np.float32
    norms = np.linalg.norm(app.embedding_matrix[::500], axis=1)
    np.testing.assert_allclose(norms, 1.0, atol=1e-5)


def test_a_float64_matrix_is_converted_once_and_normalised(monkeypatch):
    ns = _namespace(monkeypatch)
    matrix = np.random.default_rng(1).standard_normal((2_000, DIM)) * 5

    app = SimpleNamespace(embedding_matrix=matrix)
    app._prepare_matrix = MethodType(ns["_prepare_matrix"], app)
    app._prepare_matrix()

    assert app.embedding_matrix.dtype == np.float32
    assert app.embedding_matrix.flags.c_contiguous
    np.testing.assert_allclose(np.linalg.norm(app.embedding_matrix, axis=1), 1.0, atol=1e-5)
    assert matrix.dtype == np.float64        # the caller's array is left alone


def test_a_read_only_matrix_is_normalised_without_touching_the_original(monkeypatch):
    ns = _namespace(monkeypatch)
    matrix = np.random.default_rng(2).standard_normal((100, DIM)).astype(np.float32) * 3
    matrix.flags.writeable = False
    before = matrix.copy()

    app = SimpleNamespace(embedding_matrix=matrix)
    app._prepare_matrix = MethodType(ns["_prepare_matrix"], app)
    app._prepare_matrix()

    np.testing.assert_array_equal(matrix, before)
    np.testing.assert_allclose(np.linalg.norm(app.embedding_matrix, axis=1), 1.0, atol=1e-5)


def test_a_zero_row_stays_zero(monkeypatch):
    ns = _namespace(monkeypatch)
    matrix = np.zeros((3, 4), dtype=np.float32)
    matrix[1] = [3, 0, 4, 0]
    app = SimpleNamespace(embedding_matrix=matrix)
    app._prepare_matrix = MethodType(ns["_prepare_matrix"], app)

    app._prepare_matrix()

    np.testing.assert_allclose(app.embedding_matrix, [[0] * 4, [0.6, 0, 0.8, 0], [0] * 4])


def test_loading_fifty_thousand_chunks_holds_about_one_copy(monkeypatch, tmp_path):
    ns = _namespace(monkeypatch)
    matrix = _corpus()
    _write(tmp_path, matrix)
    size = matrix.nbytes
    del matrix
    app = _app(ns, tmp_path)

    started = time.perf_counter()
    _, extra = _peak_extra(lambda: ns["load_embeddings"](app))
    elapsed = time.perf_counter() - started

    assert app.df is not None and len(app.df) == ROWS
    assert app.embedding_matrix.shape == (ROWS, DIM)
    # The matrix itself, plus the chunk table; never the matrix twice.
    assert extra < 1.8 * size
    assert elapsed < 20


class _Response:
    def __init__(self, payload):
        self.payload = payload

    def raise_for_status(self):
        pass

    def json(self):
        return self.payload


class _Session:
    """Answers the embed call with a query next to the planted chunk."""

    def __init__(self, query):
        self.query = query
        self.prompts = []

    def post(self, url, json=None, timeout=None):
        if url.endswith("/api/embed"):
            return _Response({"embeddings": [self.query.tolist()]})
        self.prompts.append(json["prompt"])
        return _Response({"response": "an answer"})

    def close(self):
        pass


def test_a_question_finds_its_chunk_among_fifty_thousand_without_copying_the_corpus(
    monkeypatch, tmp_path
):
    ns = _namespace(monkeypatch)
    matrix = _corpus()
    _write(tmp_path, matrix)
    rng = np.random.default_rng(3)
    query = matrix[NEEDLE] + 0.05 * rng.standard_normal(DIM).astype(np.float32) * np.linalg.norm(matrix[NEEDLE])
    size = matrix.nbytes
    del matrix
    app = _app(ns, tmp_path)
    ns["load_embeddings"](app)
    session = _Session(query)
    app._new_session = lambda: session
    app._ui_queue = queue.Queue()
    logged = []
    app.log_safe = logged.append

    started = time.perf_counter()
    _, extra = _peak_extra(lambda: ns["_answer"](app, "where is the needle?"))
    elapsed = time.perf_counter() - started

    assert session.prompts, logged
    assert "the needle" in session.prompts[0]
    # One score per chunk and a prompt, nothing the size of the corpus.
    assert extra < 0.02 * size
    assert elapsed < 2
