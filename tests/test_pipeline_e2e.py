"""Run the four stages end to end on awkward file names.

The pipeline scripts are driven as real subprocesses, the way the dashboard
drives them, with a stand-in ffmpeg, a stand-in Whisper and a local fake
Ollama. That exercises what the unit tests cannot: names crossing process
boundaries, argument lists, JSON files, an ASCII-only console and the
machine-readable PROGRESS lines.
"""
import ast
import json
import os
import re
import stat
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import MethodType, SimpleNamespace

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).parents[1]

pytestmark = pytest.mark.skipif(
    sys.platform.startswith("win"), reason="the stand-in ffmpeg is a POSIX script"
)

ODD_NAMES = [
    "plain.mp4",
    "my lecture 1.mp4",
    "it's \"quoted\".mkv",
    "-leading dash.mov",
    "--help.mp4",
    "caf\u00e9 r\u00e9sum\u00e9.mp4",
    "\u65e5\u672c\u8a9e \u8b1b\u7fa9.mov",
    "100% done & more; $(touch pwned) `x`.avi",
    "UPPER.MP4",
]

FAKE_FFMPEG = """#!{python}
import sys
args = sys.argv[1:]
source = args[args.index("-i") + 1]
with open(source, "rb") as handle:
    handle.read(1)
with open(args[-1], "wb") as out:
    out.write(b"mp3 of " + source.encode("utf-8", "surrogateescape"))
"""

FAKE_CTRANSLATE2 = "def get_cuda_device_count():\n    return 0\n"

FAKE_WHISPER = """import os
from types import SimpleNamespace


class WhisperModel:
    def __init__(self, name, **kwargs):
        pass

    def transcribe(self, path, **kwargs):
        base = os.path.basename(path)
        segments = [
            SimpleNamespace(start=i * 5.0, end=i * 5.0 + 5.0,
                            text=f" Part {i} of {base}, na\u00efve caf\u00e9 \u65e5\u672c\u8a9e ")
            for i in range(60)
        ]
        return iter(segments), SimpleNamespace(duration=300.0, language="en")
"""


class _FakeOllama(BaseHTTPRequestHandler):
    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        vectors = [
            [float(b) + 1.0 for b in text.encode("utf-8")[:6].ljust(6, b"\x01")]
            for text in body["input"]
        ]
        payload = json.dumps({"embeddings": vectors}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, *args):
        pass


@pytest.fixture
def workspace(tmp_path):
    """A directory shaped like the project root, plus the env to run in."""
    bin_dir = tmp_path / "bin"
    stubs = tmp_path / "stubs"
    work = tmp_path / "work"
    for directory in (bin_dir, stubs, work / "videos"):
        directory.mkdir(parents=True)
    ffmpeg = bin_dir / "ffmpeg"
    ffmpeg.write_text(FAKE_FFMPEG.format(python=sys.executable))
    ffmpeg.chmod(ffmpeg.stat().st_mode | stat.S_IEXEC)
    (stubs / "ctranslate2.py").write_text(FAKE_CTRANSLATE2, encoding="utf-8")
    (stubs / "faster_whisper.py").write_text(FAKE_WHISPER, encoding="utf-8")

    server = ThreadingHTTPServer(("127.0.0.1", 0), _FakeOllama)
    threading.Thread(target=server.serve_forever, daemon=True).start()

    env = {
        key: value for key, value in os.environ.items()
        if key not in ("PYTHONUTF8", "PYTHONIOENCODING")
    }
    env.update(
        PATH=f"{bin_dir}{os.pathsep}{env.get('PATH', '')}",
        PYTHONPATH=str(stubs),
        OLLAMA_URL=f"http://127.0.0.1:{server.server_address[1]}",
        EMBED_MODEL="nomic-embed-text",
        WHISPER_DEVICE="cpu",
        # A console that cannot print the file names: the stages must still run.
        PYTHONIOENCODING="ascii",
    )
    try:
        yield SimpleNamespace(root=work, env=env)
    finally:
        server.shutdown()
        server.server_close()


def _run(workspace, script):
    return subprocess.run(
        [sys.executable, "-u", str(ROOT / script)],
        cwd=workspace.root, env=workspace.env,
        capture_output=True, text=True, encoding="ascii", errors="replace",
        timeout=120,
    )


def _progress(output):
    return [tuple(map(int, m.groups())) for m in re.finditer(r"^PROGRESS (\d+)/(\d+)$", output, re.M)]


def _dashboard_loader(workspace):
    """The dashboard's real loader and matrix preparation, without Tk."""
    tree = ast.parse((ROOT / "dashboard.py").read_text(encoding="utf-8"))
    wanted = {"EMBED_MODEL", "_USES_TASK_PREFIX", "DOCUMENT_PREFIX"}
    body = [
        node for node in tree.body
        if (isinstance(node, ast.Assign)
            and any(getattr(t, "id", None) in wanted for t in node.targets))
        or (isinstance(node, ast.FunctionDef) and node.name == "_matrix_matches_dataframe")
    ]
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "RAGDashboard")
    body += [n for n in cls.body
             if isinstance(n, ast.FunctionDef) and n.name in {"load_embeddings", "_prepare_matrix"}]
    ns = {"os": os, "json": json, "np": np, "pd": pd}
    os.environ["EMBED_MODEL"] = workspace.env["EMBED_MODEL"]
    exec(compile(ast.Module(body=body, type_ignores=[]), "dashboard.py", "exec"), ns)
    app = SimpleNamespace(base_dir=str(workspace.root), df=None, embedding_matrix=None)
    app._prepare_matrix = MethodType(ns["_prepare_matrix"], app)
    ns["load_embeddings"](app)
    return app


def test_odd_file_names_survive_all_four_stages(workspace):
    for name in ODD_NAMES:
        (workspace.root / "videos" / name).write_bytes(b"recording")

    converted = _run(workspace, "video_to_mp3.py")
    assert converted.returncode == 0, converted.stdout
    audios = sorted(p.name for p in (workspace.root / "audios").glob("*.mp3"))
    assert len(audios) == len(ODD_NAMES), audios
    assert not any(a.startswith(".") for a in audios)
    assert _progress(converted.stdout)[-1] == (len(ODD_NAMES), len(ODD_NAMES))

    transcribed = _run(workspace, "mp3_to_json.py")
    assert transcribed.returncode == 0, transcribed.stdout
    jsons = sorted(p.name for p in (workspace.root / "jsons").iterdir())
    assert jsons == [f"{a}.json" for a in audios]
    assert _progress(transcribed.stdout)[-1] == (len(ODD_NAMES), len(ODD_NAMES))

    embedded = _run(workspace, "preprocess_json.py")
    assert embedded.returncode == 0, embedded.stdout
    assert _progress(embedded.stdout)[-1] == (len(ODD_NAMES), len(ODD_NAMES))

    app = _dashboard_loader(workspace)
    assert app.df is not None and app.embedding_matrix is not None
    titles = set(app.df["title"])
    assert titles == {os.path.splitext(name)[0] for name in ODD_NAMES}
    assert not (workspace.root / "pwned").exists()
    assert not (workspace.root / "videos" / "pwned").exists()

    # Nothing half-written is left behind, anywhere.
    leftovers = [p for p in workspace.root.rglob("*") if p.name.endswith(".partial")]
    assert leftovers == []


def test_a_rerun_reports_progress_for_files_it_skips(workspace):
    for name in ODD_NAMES[:3]:
        (workspace.root / "videos" / name).write_bytes(b"recording")
    for script in ("video_to_mp3.py", "mp3_to_json.py", "preprocess_json.py"):
        assert _run(workspace, script).returncode == 0

    for script in ("video_to_mp3.py", "mp3_to_json.py", "preprocess_json.py"):
        rerun = _run(workspace, script)
        assert rerun.returncode == 0, rerun.stdout
        assert _progress(rerun.stdout) == [(0, 3), (1, 3), (2, 3), (3, 3)], script


def test_a_failed_file_still_counts_as_handled(workspace):
    (workspace.root / "videos" / "good.mp4").write_bytes(b"recording")
    (workspace.root / "videos" / "bad.mp4").write_bytes(b"recording")
    assert _run(workspace, "video_to_mp3.py").returncode == 0
    (workspace.root / "audios" / "1_bad.mp4.mp3").write_bytes(b"")

    transcribed = _run(workspace, "mp3_to_json.py")

    assert transcribed.returncode == 2, transcribed.stdout
    assert _progress(transcribed.stdout) == [(0, 2), (1, 2), (2, 2)]
