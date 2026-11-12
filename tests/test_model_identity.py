"""Exercise the dashboard loader without starting Tk or Ollama.

The corpus on disk is plain JSON plus a .npy matrix: chunks.json carries the
text and the identity of the corpus, embedding_matrix.npy the vectors, and
embedding_matrix.meta.json the identity of the matrix. Nothing is pickled,
so a tampered artifact cannot execute code — and the loader must refuse
anything that does not belong to the corpus it is asked about.
"""
import ast
import builtins
import json
import os
from pathlib import Path
from types import MethodType, SimpleNamespace

import numpy as np
import pandas as pd

ROOT = Path(__file__).parents[1]

_CONSTANTS = {"EMBED_MODEL", "_USES_TASK_PREFIX", "DOCUMENT_PREFIX", "QUERY_PREFIX"}
_FUNCTIONS = {"_matrix_matches_dataframe"}
_METHODS = {"load_embeddings", "_prepare_matrix"}


def _namespace(monkeypatch, embed_model="current"):
    """Exec the loader and the constants it reads into a stub namespace."""
    path = ROOT / "dashboard.py"
    tree = ast.parse(path.read_text())
    body = []
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            getattr(target, "id", None) in _CONSTANTS for target in node.targets
        ):
            body.append(node)
        elif isinstance(node, ast.FunctionDef) and node.name in _FUNCTIONS:
            body.append(node)
    cls = next(
        node for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "RAGDashboard"
    )
    body += [
        node for node in cls.body
        if isinstance(node, ast.FunctionDef) and node.name in _METHODS
    ]

    monkeypatch.setenv("EMBED_MODEL", embed_model)
    ns = dict(os=os, json=json, np=np, pd=pd)
    exec(compile(ast.Module(body=body, type_ignores=[]), str(path), "exec"), ns)
    return ns


def _app(ns, tmp_path):
    """A stand-in for the dashboard, with the real matrix preparation."""
    app = SimpleNamespace(base_dir=str(tmp_path), df=None, embedding_matrix=None)
    app._prepare_matrix = MethodType(ns["_prepare_matrix"], app)
    return app


def _chunks():
    return [
        {"number": 1, "title": "Intro", "start": 0, "end": 30,
         "text": "hello", "chunk_id": 0},
        {"number": 2, "title": "Intro", "start": 30, "end": 60,
         "text": "world", "chunk_id": 1},
    ]


def _write_chunks(tmp_path, chunks=None, corpus_id="corpus", model="current",
                  document_prefix=""):
    payload = {
        "corpus_id": corpus_id,
        "model": model,
        "document_prefix": document_prefix,
        "chunks": _chunks() if chunks is None else chunks,
    }
    (tmp_path / "chunks.json").write_text(json.dumps(payload), encoding="utf-8")
    return payload


def _write_matrix(tmp_path, matrix=None, corpus_id="corpus", rows=2, model="current"):
    if matrix is None:
        matrix = np.array([[3.0, 4.0], [0.0, 1.0]], dtype=np.float32)
    np.save(tmp_path / "embedding_matrix.npy", matrix)
    (tmp_path / "embedding_matrix.meta.json").write_text(json.dumps(
        {"corpus_id": corpus_id, "rows": rows, "model": model}), encoding="utf-8")


def _assert_rejected(app):
    assert app.df is None
    assert app.embedding_matrix is None


# ── a consistent corpus loads ───────────────────────────────────────────────

def test_a_consistent_corpus_loads(monkeypatch, tmp_path):
    ns = _namespace(monkeypatch)
    _write_chunks(tmp_path)
    _write_matrix(tmp_path)
    app = _app(ns, tmp_path)

    ns["load_embeddings"](app)

    assert app.df is not None
    assert len(app.df) == 2
    assert list(app.df["text"]) == ["hello", "world"]
    assert app.df.attrs["corpus_id"] == "corpus"
    assert app.embedding_matrix is not None
    assert app.embedding_matrix.shape == (2, 2)


def test_prepare_matrix_normalises_loaded_matrix(monkeypatch, tmp_path):
    ns = _namespace(monkeypatch)
    _write_chunks(tmp_path)
    _write_matrix(tmp_path)
    app = _app(ns, tmp_path)

    ns["load_embeddings"](app)

    assert app.df is not None
    assert app.embedding_matrix is not None
    np.testing.assert_allclose(app.embedding_matrix[0], [0.6, 0.8], atol=1e-6)
    assert np.isclose(np.linalg.norm(app.embedding_matrix[0]), 1.0)


# ── a corpus that is not ours is refused ────────────────────────────────────

def test_a_different_embedding_model_is_rejected(monkeypatch, tmp_path):
    ns = _namespace(monkeypatch)
    _write_chunks(tmp_path, model="another-model")
    _write_matrix(tmp_path)
    app = _app(ns, tmp_path)

    ns["load_embeddings"](app)

    _assert_rejected(app)


def test_a_different_document_prefix_is_rejected(monkeypatch, tmp_path):
    ns = _namespace(monkeypatch)
    _write_chunks(tmp_path, document_prefix="search_document: ")
    _write_matrix(tmp_path)
    app = _app(ns, tmp_path)

    ns["load_embeddings"](app)

    _assert_rejected(app)


def test_a_matrix_from_another_corpus_is_rejected(monkeypatch, tmp_path):
    ns = _namespace(monkeypatch)
    _write_chunks(tmp_path)
    _write_matrix(tmp_path, corpus_id="another-corpus")
    app = _app(ns, tmp_path)

    ns["load_embeddings"](app)

    _assert_rejected(app)


def test_a_matrix_from_another_model_is_rejected(monkeypatch, tmp_path):
    ns = _namespace(monkeypatch)
    _write_chunks(tmp_path)
    _write_matrix(tmp_path, model="another-model")
    app = _app(ns, tmp_path)

    ns["load_embeddings"](app)

    _assert_rejected(app)


def test_a_matrix_with_the_wrong_row_count_is_rejected(monkeypatch, tmp_path):
    ns = _namespace(monkeypatch)
    _write_chunks(tmp_path)
    _write_matrix(tmp_path, rows=1)
    app = _app(ns, tmp_path)

    ns["load_embeddings"](app)

    _assert_rejected(app)


def test_a_missing_matrix_is_rejected(monkeypatch, tmp_path):
    ns = _namespace(monkeypatch)
    _write_chunks(tmp_path)
    (tmp_path / "embedding_matrix.meta.json").write_text(json.dumps(
        {"corpus_id": "corpus", "rows": 2, "model": "current"}), encoding="utf-8")
    app = _app(ns, tmp_path)

    ns["load_embeddings"](app)

    _assert_rejected(app)


def test_a_matrix_with_too_few_rows_is_rejected(monkeypatch, tmp_path):
    ns = _namespace(monkeypatch)
    _write_chunks(tmp_path)
    _write_matrix(tmp_path, matrix=np.array([[1.0, 0.0]], dtype=np.float32))
    app = _app(ns, tmp_path)

    ns["load_embeddings"](app)

    _assert_rejected(app)


# ── a malformed chunk file is refused ───────────────────────────────────────

def test_a_chunk_file_that_is_a_list_is_rejected(monkeypatch, tmp_path):
    ns = _namespace(monkeypatch)
    (tmp_path / "chunks.json").write_text(json.dumps([_chunks()]), encoding="utf-8")
    _write_matrix(tmp_path)
    app = _app(ns, tmp_path)

    ns["load_embeddings"](app)

    _assert_rejected(app)


def test_a_chunk_file_without_chunks_is_rejected(monkeypatch, tmp_path):
    ns = _namespace(monkeypatch)
    _write_chunks(tmp_path, chunks=[])
    _write_matrix(tmp_path)
    app = _app(ns, tmp_path)

    ns["load_embeddings"](app)

    _assert_rejected(app)


def test_a_chunk_file_without_a_corpus_id_is_rejected(monkeypatch, tmp_path):
    ns = _namespace(monkeypatch)
    _write_chunks(tmp_path, corpus_id="")
    _write_matrix(tmp_path)
    app = _app(ns, tmp_path)

    ns["load_embeddings"](app)

    _assert_rejected(app)


# ── the old pickle is ignored, never opened ─────────────────────────────────

def test_an_old_pickle_is_ignored_and_never_opened(monkeypatch, tmp_path):
    ns = _namespace(monkeypatch)
    legacy = tmp_path / "embeddings.joblib"
    legacy.write_bytes(b"this is not a pickle")
    app = _app(ns, tmp_path)

    opened = []
    real_open = builtins.open

    def tracking_open(file, *args, **kwargs):
        opened.append(str(file))
        return real_open(file, *args, **kwargs)

    monkeypatch.setattr(builtins, "open", tracking_open)
    ns["load_embeddings"](app)

    _assert_rejected(app)
    assert not any(path.endswith("embeddings.joblib") for path in opened), \
        "the loader must never open the legacy pickle"
    assert legacy.read_bytes() == b"this is not a pickle"


def test_a_missing_corpus_clears_previous_state(monkeypatch, tmp_path):
    ns = _namespace(monkeypatch)
    app = _app(ns, tmp_path)
    app.df = "stale dataframe"
    app.embedding_matrix = "stale matrix"

    ns["load_embeddings"](app)

    _assert_rejected(app)
