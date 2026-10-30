"""Exercise the dashboard loader without starting Tk or Ollama."""
import ast
import json
from pathlib import Path
from types import SimpleNamespace

import joblib
import numpy as np
import pandas as pd
import pytest


def loader():
    import os
    path = Path(__file__).parents[1] / "dashboard.py"
    tree = ast.parse(path.read_text())
    functions = [node for node in tree.body if isinstance(node, ast.FunctionDef)
                 and node.name in {"_recorded_models", "_dataframe_embeddings_match_model", "_matrix_matches_dataframe"}]
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "RAGDashboard")
    functions += [node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == "load_embeddings"]
    ns = dict(os=os, json=json, joblib=joblib, np=np, EMBED_MODEL="current")
    exec(compile(ast.Module(body=functions, type_ignores=[]), str(path), "exec"), ns)
    return ns["load_embeddings"]


@pytest.mark.parametrize("recorded", [None, "old", "current"])
def test_loader_rejects_unknown_or_different_model(tmp_path, recorded):
    df = pd.DataFrame([dict(text="hello", embedding=[1., 0.])])
    df.attrs["corpus_id"] = "corpus"
    if recorded:
        df.attrs["embedding_model"] = recorded
    joblib.dump(df, tmp_path / "embeddings.joblib")
    app = SimpleNamespace(base_dir=str(tmp_path), df=None, embedding_matrix=None,
                          _prepare_matrix=lambda: None)
    loader()(app)
    assert (app.df is not None) == (recorded == "current")
    assert (app.embedding_matrix is not None) == (recorded == "current")


def test_unrelated_model_sidecar_cannot_authorize_legacy_dataframe(tmp_path):
    df = pd.DataFrame([dict(text="hello", embedding=[1., 0.])])
    df.attrs["corpus_id"] = "old-corpus"
    joblib.dump(df, tmp_path / "embeddings.joblib")
    (tmp_path / "embedding_matrix.meta.json").write_text(json.dumps(
        dict(corpus_id="new-corpus", rows=1, model="current")))
    app = SimpleNamespace(base_dir=str(tmp_path), df=None, embedding_matrix=None,
                          _prepare_matrix=lambda: None)
    loader()(app)
    assert app.df is None
    assert app.embedding_matrix is None


def test_same_corpus_matrix_from_different_model_is_not_used(tmp_path):
    df = pd.DataFrame([dict(text="hello", embedding=[1., 0.])])
    df.attrs.update(corpus_id="same", embedding_model="current")
    joblib.dump(df, tmp_path / "embeddings.joblib")
    np.save(tmp_path / "embedding_matrix.npy", np.array([[0., 1.]]))
    (tmp_path / "embedding_matrix.meta.json").write_text(json.dumps(
        dict(corpus_id="same", rows=1, model="old")))
    app = SimpleNamespace(base_dir=str(tmp_path), df=None, embedding_matrix=None,
                          _prepare_matrix=lambda: None)
    loader()(app)
    assert app.df is not None
    np.testing.assert_array_equal(app.embedding_matrix, [[1., 0.]])
