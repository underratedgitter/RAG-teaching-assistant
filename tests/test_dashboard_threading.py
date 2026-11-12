"""Worker threads must never touch Tk, and must not share an HTTP session.

Same approach as test_dashboard.py: Tkinter is unavailable here, so the
methods under test are pulled out of dashboard.py with ast and exec'd into a
namespace, then called on a stand-in for ``self``.
"""
import ast
import queue
import secrets
import threading
from pathlib import Path
from types import MethodType, SimpleNamespace

import numpy as np
import pandas as pd
import requests
from requests.adapters import HTTPAdapter

DASHBOARD = Path(__file__).parents[1] / "dashboard.py"

# The only methods allowed to call self.root.after*: both run on the Tk thread.
_POLLER = "_drain_ui_queue"
_POLL_STOPPER = "_stop_polling"

_FUNCTIONS = {"_clean_field", "_neutralise_tags", "_time_parts", "build_prompt"}
_METHODS = {
    "log",
    "log_safe",
    "_status",
    "_drain_ui_queue",
    "_flush_log_lines",
    "_apply_ui_action",
    "_new_session",
    "_answer",
}
_CONSTANTS = {
    "TOP_K",
    "SIMILARITY_THRESHOLD",
    "FALLBACK_K",
    "OLLAMA_URL",
    "EMBED_MODEL",
    "ANSWER_MODEL",
    "_USES_TASK_PREFIX",
    "QUERY_PREFIX",
    "SYSTEM_PROMPT",
    "GENERATION_OPTIONS",
    "_TAG_RE",
    "UI_POLL_MS",
    "TERMINAL_MAX_LINES",
}


def _class_methods():
    tree = ast.parse(DASHBOARD.read_text())
    cls = next(node for node in tree.body
               if isinstance(node, ast.ClassDef) and node.name == "RAGDashboard")
    return tree, [child for child in cls.body if isinstance(child, ast.FunctionDef)]


def _namespace():
    tree, methods = _class_methods()
    body = [
        node for node in tree.body
        if (isinstance(node, ast.FunctionDef) and node.name in _FUNCTIONS)
        or (isinstance(node, ast.Assign) and any(
            getattr(target, "id", None) in _CONSTANTS for target in node.targets))
    ]
    body.extend(method for method in methods if method.name in _METHODS)
    ns = {
        "os": __import__("os"),
        "re": __import__("re"),
        "queue": queue,
        "secrets": secrets,
        "np": np,
        "requests": requests,
        "HTTPAdapter": HTTPAdapter,
    }
    exec(compile(ast.Module(body=body, type_ignores=[]), str(DASHBOARD), "exec"), ns)
    return ns


# ── stubs ───────────────────────────────────────────────────────────────────

class _Root:
    """Records what was scheduled; nothing runs unless the test runs it."""

    def __init__(self):
        self.scheduled = []

    def after(self, delay, callback):
        self.scheduled.append((delay, callback))


class _Terminal:
    """A Text widget reduced to the calls log() makes, with Tk's line indices."""

    def __init__(self, events):
        self.events = events
        self.content = ""
        self.inserts = []
        self.sees = 0

    def insert(self, index, text):
        assert index == "end"
        self.inserts.append(text)
        self.content += text
        self.events.append(("insert", text))

    def index(self, index):
        assert index == "end-1c"
        line = self.content.count("\n") + 1
        column = len(self.content.rsplit("\n", 1)[-1])
        return f"{line}.{column}"

    def delete(self, start, end):
        assert start == "1.0" and end.endswith(".0")
        drop = int(end.split(".")[0]) - 1
        self.content = "".join(self.content.splitlines(keepends=True)[drop:])

    def see(self, index):
        self.sees += 1

    @property
    def lines(self):
        return self.content.splitlines()


class _Label:
    def __init__(self, events):
        self.events = events

    def config(self, **kwargs):
        self.events.append(("status", kwargs["text"]))


def _app(ns):
    events = []
    app = SimpleNamespace(
        root=_Root(),
        terminal=_Terminal(events),
        status_label=_Label(events),
        _ui_queue=queue.Queue(),
    )
    for name in ("log", "log_safe", "_status", "_drain_ui_queue",
                 "_flush_log_lines", "_apply_ui_action"):
        setattr(app, name, MethodType(ns[name], app))
    return app, events


def _on_thread(target, *args):
    thread = threading.Thread(target=target, args=args)
    thread.start()
    thread.join(timeout=10)
    assert not thread.is_alive()


# ── worker threads only queue ──────────────────────────────────────────────

def test_a_worker_thread_touches_no_widget_and_schedules_nothing():
    ns = _namespace()
    app, events = _app(ns)

    def worker():
        app.log_safe("a line")
        app._status("a status")

    _on_thread(worker)

    assert events == []
    assert app.terminal.inserts == []
    assert app.root.scheduled == []
    assert app._ui_queue.qsize() == 2

    app._drain_ui_queue()
    assert events == [("insert", "a line\n"), ("status", "a status")]


def test_a_drain_applies_queued_actions_in_order():
    ns = _namespace()
    app, events = _app(ns)

    app.log_safe("before")
    app._status("first")
    app._ui_queue.put(lambda: events.append(("action", "middle")))
    app.log_safe("after")
    app._status("second")
    app._drain_ui_queue()

    assert events == [
        ("insert", "before\n"),
        ("status", "first"),
        ("action", "middle"),
        ("insert", "after\n"),
        ("status", "second"),
    ]
    assert app._ui_queue.empty()


def test_queued_log_lines_are_inserted_in_one_call():
    ns = _namespace()
    app, events = _app(ns)

    for n in range(500):
        app.log_safe(f"line {n}")
    app._drain_ui_queue()

    assert len(app.terminal.inserts) == 1
    assert app.terminal.sees == 1
    assert app.terminal.lines == [f"line {n}" for n in range(500)]


# ── the terminal is capped ─────────────────────────────────────────────────

def test_the_terminal_is_trimmed_to_the_cap_after_a_drain():
    ns = _namespace()
    cap = ns["TERMINAL_MAX_LINES"]
    app, events = _app(ns)

    for n in range(cap + 250):
        app.log_safe(f"line {n}")
    app._drain_ui_queue()

    assert len(app.terminal.lines) == cap
    assert app.terminal.lines[0] == "line 250"
    assert app.terminal.lines[-1] == f"line {cap + 249}"


def test_a_direct_log_call_respects_the_cap():
    ns = _namespace()
    cap = ns["TERMINAL_MAX_LINES"]
    app, events = _app(ns)

    for n in range(cap + 3):
        app.log(f"line {n}")

    assert len(app.terminal.lines) == cap
    assert app.terminal.lines[0] == "line 3"
    assert app.terminal.lines[-1] == f"line {cap + 2}"


def test_a_line_with_embedded_newlines_counts_every_line():
    ns = _namespace()
    cap = ns["TERMINAL_MAX_LINES"]
    app, events = _app(ns)

    for n in range(cap):
        app.log(f"line {n}")
    app.log("\nbanner")

    assert len(app.terminal.lines) == cap
    assert app.terminal.lines[0] == "line 2"
    assert app.terminal.lines[-2:] == ["", "banner"]


# ── the poller keeps running ───────────────────────────────────────────────

def test_the_drain_reschedules_itself():
    ns = _namespace()
    app, events = _app(ns)

    app._drain_ui_queue()

    assert len(app.root.scheduled) == 1
    delay, callback = app.root.scheduled[0]
    assert delay == ns["UI_POLL_MS"] > 0
    assert callback == app._drain_ui_queue

    callback()
    assert len(app.root.scheduled) == 2


def test_a_failing_action_is_reported_and_does_not_stop_the_drain(capsys):
    ns = _namespace()
    app, events = _app(ns)

    def broken():
        raise RuntimeError("widget is gone")

    app.log_safe("before")
    app._ui_queue.put(broken)
    app.log_safe("after")
    app._status("still applied")
    app._drain_ui_queue()

    assert events == [
        ("insert", "before\n"),
        ("insert", "after\n"),
        ("status", "still applied"),
    ]
    assert "widget is gone" in capsys.readouterr().out
    assert len(app.root.scheduled) == 1


def test_a_failing_terminal_insert_is_reported_and_does_not_stop_the_drain(capsys):
    ns = _namespace()
    app, events = _app(ns)

    def broken_insert(index, text):
        raise RuntimeError("terminal is gone")

    app.terminal.insert = broken_insert
    app.log_safe("lost")
    app._status("still applied")
    app._drain_ui_queue()

    assert events == [("status", "still applied")]
    assert "terminal is gone" in capsys.readouterr().out
    assert len(app.root.scheduled) == 1


# ── each worker gets its own HTTP session, and closes it ───────────────────

class _Response:
    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self._payload


class _Session:
    def __init__(self, fail=None, gate=None):
        self.fail = fail
        self.gate = gate
        self.closed = False
        self.threads = set()

    def post(self, url, json=None, timeout=None):
        assert not self.closed
        self.threads.add(threading.get_ident())
        if self.fail is not None:
            raise self.fail
        if url.endswith("/api/embed"):
            if self.gate is not None:
                self.gate.wait(timeout=10)
            return _Response({"embeddings": [[1.0, 0.0]]})
        return _Response({"response": "an answer"})

    def close(self):
        self.closed = True


def _answer_app(ns, fail=None, gate=None):
    sessions = []

    def new_session():
        sessions.append(_Session(fail, gate))
        return sessions[-1]

    app, events = _app(ns)
    app.df = pd.DataFrame.from_records([{
        "number": 1, "title": "Intro", "start": 0, "end": 30, "text": "hello",
    }])
    app.embedding_matrix = np.array([[1.0, 0.0]], dtype=np.float32)
    app._new_session = new_session
    app._show_response = lambda text: events.append(("response", text))
    app._answer = MethodType(ns["_answer"], app)
    return app, events, sessions


def test_each_answer_thread_uses_its_own_session_and_closes_it():
    ns = _namespace()
    # Both threads are held inside their first request until the other has
    # arrived, so they are alive together and cannot reuse a thread id.
    app, events, sessions = _answer_app(ns, gate=threading.Barrier(2))

    threads = [threading.Thread(target=app._answer, args=(question,))
               for question in ("What is a monoid?", "What is a functor?")]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=10)
        assert not thread.is_alive()

    first, second = sessions
    assert first is not second
    assert len(first.threads) == 1 and len(second.threads) == 1
    assert first.threads.isdisjoint(second.threads)
    assert first.closed and second.closed


def test_an_answer_thread_only_queues_its_result():
    ns = _namespace()
    app, events, sessions = _answer_app(ns)

    _on_thread(app._answer, "What is a monoid?")

    assert events == []
    assert app.root.scheduled == []

    app._drain_ui_queue()
    assert events[-1] == ("response", "an answer")
    assert events[-2][0] == "insert" and events[-2][1].endswith("Done!\n")


def test_the_session_is_closed_when_the_request_fails():
    ns = _namespace()
    app, events, sessions = _answer_app(
        ns, fail=requests.exceptions.ConnectionError("refused"))

    _on_thread(app._answer, "What is a monoid?")
    app._drain_ui_queue()

    assert sessions[0].closed
    assert events[-1] == (
        "response", "Error: Cannot connect to Ollama. Start it with: ollama serve")


def test_a_new_session_keeps_the_pooled_adapter():
    ns = _namespace()
    first = ns["_new_session"](SimpleNamespace())
    second = ns["_new_session"](SimpleNamespace())
    try:
        assert first is not second
        adapter = first.get_adapter("http://localhost:11434")
        assert adapter is not second.get_adapter("http://localhost:11434")
        assert adapter._pool_connections == 2
        assert adapter._pool_maxsize == 2
    finally:
        first.close()
        second.close()


# ── source-level guards ────────────────────────────────────────────────────

def _is_self_attribute(node, name):
    return (isinstance(node, ast.Attribute) and node.attr == name
            and isinstance(node.value, ast.Name) and node.value.id == "self")


def test_only_the_poller_schedules_on_the_tk_root():
    tree, methods = _class_methods()
    callers = {
        method.name
        for method in methods
        for node in ast.walk(method)
        if isinstance(node, ast.Attribute)
        and node.attr.startswith("after")
        and _is_self_attribute(node.value, "root")
    }
    assert callers == {_POLLER, _POLL_STOPPER}


def test_the_poller_is_started_from_init():
    tree, methods = _class_methods()
    init = next(method for method in methods if method.name == "__init__")
    # At the top level of __init__, after the try/except, so the fallback UI
    # path starts it too.
    last = init.body[-1]
    assert isinstance(last, ast.Expr) and isinstance(last.value, ast.Call)
    assert _is_self_attribute(last.value.func, _POLLER)
    assert any(isinstance(node, ast.Try) for node in init.body[:-1])


def test_no_session_is_shared_on_the_instance():
    tree, methods = _class_methods()
    shared = [
        method.name
        for method in methods
        for node in ast.walk(method)
        if _is_self_attribute(node, "session")
    ]
    assert shared == []


def test_destroying_the_window_cancels_the_scheduled_poll():
    tree, methods = _class_methods()
    stop = next(method for method in methods if method.name == _POLL_STOPPER)
    ns = {}
    exec(compile(ast.Module(body=[stop], type_ignores=[]), str(DASHBOARD), "exec"), ns)
    cancelled = []
    root = SimpleNamespace(after_cancel=cancelled.append)
    app = SimpleNamespace(root=root, _poll_id="after#7")

    ns[_POLL_STOPPER](app, SimpleNamespace(widget=object()))   # a child widget
    assert cancelled == [] and app._poll_id == "after#7"

    ns[_POLL_STOPPER](app, SimpleNamespace(widget=root))
    assert cancelled == ["after#7"] and app._poll_id is None
