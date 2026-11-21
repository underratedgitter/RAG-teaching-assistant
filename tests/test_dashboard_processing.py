"""Cancelling a run, and the progress the status bar shows.

The pipeline stages are real child processes (small scripts standing in for
the real ones), so these tests cover what cancelling has to get right: the
stage's own children die with it, nothing half-written stays behind, and the
buttons come back. Tk is not involved; the dashboard methods are pulled out
of dashboard.py with ast, as in test_dashboard.py.
"""
import ast
import os
import queue
import re
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path
from types import MethodType, SimpleNamespace

import pytest

DASHBOARD = Path(__file__).parents[1] / "dashboard.py"

pytestmark = pytest.mark.skipif(
    sys.platform.startswith("win"), reason="the stand-in stages are POSIX process trees"
)

_FUNCTIONS = {"_parse_progress", "_stop_process_tree", "_wait_for_group_exit"}
_METHODS = {
    "run_script", "cancel_processing", "_remove_partial_files", "_cancelled",
    "_process", "_processing_finished", "_sync_buttons", "_status", "log_safe",
}
_CONSTANTS = {
    "PIPELINE_STEPS", "STAGE_COUNT", "_PROGRESS_RE", "CANCEL_GRACE_S",
    "ROOT_PARTIAL_FILES", "PARTIAL_FAILURE",
}


def _namespace():
    tree = ast.parse(DASHBOARD.read_text(encoding="utf-8"))
    body = []
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name in _FUNCTIONS:
            body.append(node)
        elif isinstance(node, ast.Assign) and any(
            getattr(target, "id", None) in _CONSTANTS for target in node.targets
        ):
            body.append(node)
        elif isinstance(node, ast.ClassDef) and node.name == "RAGDashboard":
            body.extend(c for c in node.body
                        if isinstance(c, ast.FunctionDef) and c.name in _METHODS)
    ns = {"os": os, "re": re, "signal": signal, "subprocess": subprocess,
          "sys": sys, "threading": threading, "time": time}
    exec(compile(ast.Module(body=body, type_ignores=[]), str(DASHBOARD), "exec"), ns)
    return ns


class _Button:
    def __init__(self):
        self.state = None

    def config(self, **kwargs):
        self.state = kwargs.get("state", self.state)


class _Label:
    def __init__(self):
        self.statuses = []

    def config(self, **kwargs):
        self.statuses.append(kwargs["text"])


def _app(ns, tmp_path, stages):
    """A stand-in dashboard whose stages are the given scripts in tmp_path."""
    for name, source in stages.items():
        (tmp_path / name).write_text(source, encoding="utf-8")
    for folder in ("audios", "jsons"):
        (tmp_path / folder).mkdir(exist_ok=True)
    ns["PIPELINE_STEPS"] = tuple(
        (name, f"Doing {name}...", f"Title {name}", f"noun {name}") for name in stages
    )
    ns["STAGE_COUNT"] = len(stages) + 1
    app = SimpleNamespace(
        base_dir=str(tmp_path),
        audios_dir=str(tmp_path / "audios"),
        jsons_dir=str(tmp_path / "jsons"),
        df=None,
        status_label=_Label(),
        process_btn=_Button(), ask_btn=_Button(), cancel_btn=_Button(),
        terminal=[],
        _processing=True,
        _answering=False,
        _cancel=threading.Event(),
        _process_lock=threading.Lock(),
        _current_process=None,
        _ui_queue=queue.Queue(),
        load_embeddings=lambda: setattr(app, "df", [0, 1, 2]),
    )
    app.log = app.terminal.append
    for name in _METHODS:
        setattr(app, name, MethodType(ns[name], app))
    return app


def _drain(app):
    """Apply what the worker queued, as the Tk thread would."""
    while not app._ui_queue.empty():
        action = app._ui_queue.get_nowait()
        if isinstance(action, str):
            app.terminal.append(action)
        else:
            action()


def _showing(app, fragment):
    """Whether the status bar has shown a status containing the fragment."""
    _drain(app)
    return any(fragment in status for status in app.status_label.statuses)


def _wait_for(condition, timeout=10):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if condition():
            return True
        time.sleep(0.02)
    return False


def _gone(pid, timeout=5):
    def dead():
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return True
        return False
    return _wait_for(dead, timeout)


# A stage that, like video_to_mp3.py, runs a worker of its own, has a temp
# file open, and reports progress.
HANGING_STAGE = """
import os, subprocess, sys, time
os.makedirs("audios", exist_ok=True)
worker = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(120)"])
with open("worker.pid", "w") as handle:
    handle.write(str(worker.pid))
with open("audios/.1_lecture.mp4.mp3.partial", "wb") as handle:
    handle.write(b"half an mp3")
with open("chunks.json.partial", "w") as handle:
    handle.write("{")
print("PROGRESS 1/3", flush=True)
time.sleep(120)
"""


def _start(app):
    thread = threading.Thread(target=app._process, daemon=True)
    thread.start()
    return thread


# ── progress ───────────────────────────────────────────────────────────────

@pytest.mark.parametrize("line, expected", [
    ("PROGRESS 3/10", (3, 10)),
    ("PROGRESS 0/0", (0, 0)),
    ("  PROGRESS 12/12  ", (12, 12)),
    ("PROGRESS 3/", None),
    ("PROGRESS -1/4", None),
    ("not PROGRESS 3/10", None),
    ("Created PROGRESS 3/10.mp3", None),
    ("", None),
])
def test_a_progress_line_is_recognised_only_when_it_is_the_whole_line(line, expected):
    assert _namespace()["_parse_progress"](line) == expected


def test_a_file_named_like_a_progress_line_is_not_taken_for_one(tmp_path):
    ns = _namespace()
    app = _app(ns, tmp_path, {"stage.py": "print('  Created PROGRESS 4/9')\nprint('PROGRESS 1/2')\n"})
    seen = []

    code = app.run_script("stage.py", "Stage", lambda done, total: seen.append((done, total)))
    _drain(app)

    assert code == 0
    assert seen == [(1, 2)]
    assert "  Created PROGRESS 4/9" in app.terminal
    assert "PROGRESS 1/2" not in app.terminal


def test_the_status_bar_shows_the_stage_and_the_file_count(tmp_path):
    ns = _namespace()
    stage = "print('PROGRESS 0/2', flush=True)\nprint('PROGRESS 1/2', flush=True)\nprint('PROGRESS 2/2', flush=True)\n"
    app = _app(ns, tmp_path, {"a.py": stage, "b.py": stage, "c.py": stage})

    app._process()
    _drain(app)

    statuses = app.status_label.statuses
    assert "Step 1/4: Doing a.py... 0/2 files" in statuses
    assert "Step 2/4: Doing b.py... 1/2 files" in statuses
    assert "Step 3/4: Doing c.py... 2/2 files" in statuses
    assert "Step 4/4: Loading the index..." in statuses
    assert statuses[-1] == "Done! 3 chunks indexed"
    assert statuses.index("Step 1/4: Doing a.py... 2/2 files") < statuses.index("Step 2/4: Doing b.py...")
    assert not any(line.startswith("PROGRESS") for line in app.terminal)


def test_a_script_that_never_reports_progress_still_shows_its_stage(tmp_path):
    ns = _namespace()
    app = _app(ns, tmp_path, {"a.py": "print('hello')\n"})

    app._process()
    _drain(app)

    assert "Step 1/2: Doing a.py..." in app.status_label.statuses


def test_scripts_print_file_names_the_console_cannot_encode(tmp_path, monkeypatch):
    """Both ends use UTF-8, whatever the console's code page is."""
    monkeypatch.setenv("PYTHONIOENCODING", "ascii")
    monkeypatch.setenv("PYTHONUTF8", "0")
    ns = _namespace()
    app = _app(ns, tmp_path, {"stage.py": "print('caf\\u00e9 \\u65e5\\u672c\\u8a9e.mp4')\n"})

    assert app.run_script("stage.py", "Stage") == 0
    _drain(app)

    assert "caf\u00e9 \u65e5\u672c\u8a9e.mp4" in app.terminal


# ── cancelling ─────────────────────────────────────────────────────────────

def test_cancel_stops_the_stage_and_its_workers_and_leaves_no_partial_files(tmp_path):
    ns = _namespace()
    app = _app(ns, tmp_path, {"hang.py": HANGING_STAGE, "next.py": "open('next-ran', 'w')\n"})
    # A finished file and an unrelated one must survive the sweep.
    (tmp_path / "audios" / "1_lecture.mp4.mp3").write_bytes(b"finished")
    (tmp_path / "notes.partial").write_text("mine")

    worker = _start(app)
    assert _wait_for(lambda: (tmp_path / "worker.pid").exists() and _showing(app, "1/3"))
    pid = int((tmp_path / "worker.pid").read_text())

    started = time.monotonic()
    app.cancel_processing()
    worker.join(timeout=15)
    _drain(app)

    assert not worker.is_alive()
    assert time.monotonic() - started < 5
    assert _gone(pid), "the stage's own worker kept running"
    assert not (tmp_path / "next-ran").exists(), "the next stage must not start"
    assert app.status_label.statuses[-1] == "Cancelled"
    assert not (tmp_path / "audios" / ".1_lecture.mp4.mp3.partial").exists()
    assert not (tmp_path / "chunks.json.partial").exists()
    assert (tmp_path / "audios" / "1_lecture.mp4.mp3").read_bytes() == b"finished"
    assert (tmp_path / "notes.partial").exists()


def test_a_worker_still_closing_its_file_is_waited_for_before_the_sweep(tmp_path):
    """ffmpeg finishes writing after the stage that started it has died."""
    slow_worker = (
        "import signal, sys, time\n"
        "def stop(*args):\n"
        "    time.sleep(0.6)\n"
        "    open('audios/.late.mp3.partial', 'w').write('half')\n"
        "    sys.exit(0)\n"
        "signal.signal(signal.SIGTERM, stop)\n"
        "import os\n"
        "open('worker.ready', 'w').write(str(os.getpid()))\n"
        "time.sleep(120)\n"
    )
    stage = (
        "import subprocess, sys, time\n"
        f"subprocess.Popen([sys.executable, '-c', {slow_worker!r}],\n"
        "                 stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)\n"
        "print('PROGRESS 1/3', flush=True)\n"
        "time.sleep(120)\n"
    )
    ns = _namespace()
    app = _app(ns, tmp_path, {"stage.py": stage})

    worker = _start(app)
    assert _wait_for(lambda: (tmp_path / "worker.ready").exists() and _showing(app, "1/3"))
    app.cancel_processing()
    worker.join(timeout=15)

    assert not worker.is_alive()
    assert _gone(int((tmp_path / "worker.ready").read_text()))
    assert not (tmp_path / "audios" / ".late.mp3.partial").exists()


def test_the_buttons_come_back_after_a_cancel(tmp_path):
    ns = _namespace()
    app = _app(ns, tmp_path, {"hang.py": HANGING_STAGE})
    app._sync_buttons()
    assert app.cancel_btn.state == "normal"
    assert app.process_btn.state == "disabled"

    worker = _start(app)
    assert _wait_for(lambda: (tmp_path / "worker.pid").exists())
    app.cancel_processing()
    assert app.cancel_btn.state == "disabled", "a second click has nothing left to cancel"
    worker.join(timeout=15)
    _drain(app)

    assert app._processing is False
    assert app.process_btn.state == "normal"
    assert app.ask_btn.state == "normal"
    assert app.cancel_btn.state == "disabled"


def test_a_stage_that_ignores_sigterm_is_killed_after_the_grace_period(tmp_path):
    ns = _namespace()
    ns["CANCEL_GRACE_S"] = 0.5
    stubborn = (
        "import os, signal, time\n"
        "signal.signal(signal.SIGTERM, signal.SIG_IGN)\n"
        "open('stubborn.pid', 'w').write(str(os.getpid()))\n"
        "print('PROGRESS 1/3', flush=True)\n"
        "time.sleep(120)\n"
    )
    app = _app(ns, tmp_path, {"stubborn.py": stubborn})

    worker = _start(app)
    assert _wait_for(lambda: (tmp_path / "stubborn.pid").exists() and _showing(app, "1/3"))
    pid = int((tmp_path / "stubborn.pid").read_text())
    app.cancel_processing()
    worker.join(timeout=15)

    assert not worker.is_alive()
    assert _gone(pid)
    _drain(app)
    assert app.status_label.statuses[-1] == "Cancelled"


def test_cancel_before_a_stage_starts_spawns_nothing(tmp_path):
    ns = _namespace()
    app = _app(ns, tmp_path, {"a.py": "open('ran', 'w')\n"})
    app._cancel.set()

    assert app.run_script("a.py", "A") is None
    assert not (tmp_path / "ran").exists()


def test_cancel_between_stages_skips_the_rest_and_the_load(tmp_path):
    ns = _namespace()
    loads = []
    first = "import time\ntime.sleep(0.3)\n"
    app = _app(ns, tmp_path, {"a.py": first, "b.py": "open('b-ran', 'w')\n"})
    app.load_embeddings = lambda: loads.append(True)

    worker = _start(app)
    assert _wait_for(lambda: app._current_process is not None)
    app.cancel_processing()
    worker.join(timeout=15)

    assert not (tmp_path / "b-ran").exists()
    assert loads == []


def test_cancel_with_nothing_running_does_nothing(tmp_path):
    ns = _namespace()
    app = _app(ns, tmp_path, {"a.py": ""})
    app._processing = False

    app.cancel_processing()

    assert not app._cancel.is_set()
    assert app.terminal == []


def test_a_finished_run_is_not_reported_as_cancelled(tmp_path):
    ns = _namespace()
    app = _app(ns, tmp_path, {"a.py": "print('ok')\n"})

    app._process()
    _drain(app)

    assert app.status_label.statuses[-1] == "Done! 3 chunks indexed"
    assert app._processing is False
