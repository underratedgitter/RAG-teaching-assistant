"""Startup checks: what is reported, what it blocks, and what the scripts do about it."""
import os
import subprocess
import sys
from pathlib import Path

import pytest
import requests

import preflight
from preflight import ASK, PROCESS

ROOT = Path(__file__).parents[1]


class _Reply:
    def __init__(self, payload=None, status=200):
        self._payload = payload
        self.status_code = status

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(f"{self.status_code} error", response=self)

    def json(self):
        if isinstance(self._payload, Exception):
            raise self._payload
        return self._payload


def _tags(*names):
    return _Reply({"models": [{"name": name} for name in names]})


def _ollama(monkeypatch, reply):
    """Answer GET /api/tags with `reply`, or raise it if it is an exception."""
    def get(url, timeout=None):
        assert url.endswith("/api/tags")
        if isinstance(reply, Exception):
            raise reply
        return reply
    monkeypatch.setattr(preflight.requests, "get", get)


# ── settings ────────────────────────────────────────────────────────────────

def test_the_defaults_are_valid():
    assert preflight.check_config({}) == []


@pytest.mark.parametrize("name, value", [
    ("OLLAMA_URL", "localhost:11434"),
    ("OLLAMA_URL", "ftp://localhost"),
    ("EMBED_MODEL", "  "),
    ("EMBED_CONCURRENCY", "0"),
    ("EMBED_CONCURRENCY", "three"),
    ("SIMILARITY_THRESHOLD", "high"),
    ("SIMILARITY_THRESHOLD", "nan"),
    ("SIMILARITY_THRESHOLD", "1.5"),
    ("TOP_K", "0"),
    ("FALLBACK_K", "-1"),
    ("OLLAMA_TIMEOUT", "0"),
    ("WHISPER_MODEL", "smal"),
    ("WHISPER_DEVICE", "gpu"),
    ("WHISPER_COMPUTE_TYPE", "fp16"),
    ("WHISPER_BEAM_SIZE", "0"),
    ("WHISPER_VAD", "yes"),
    ("WHISPER_CPU_THREADS", "-2"),
    ("CLEAR_ON_START", "yes"),
])
def test_a_bad_value_names_the_variable_and_the_rule(name, value):
    [error] = preflight.check_config({name: value})
    assert error.startswith(f"{name}={value!r} must")


@pytest.mark.parametrize("name, value", [
    ("OLLAMA_URL", "https://ollama.example.com:8443"),
    ("EMBED_MODEL", "mxbai-embed-large"),
    ("SIMILARITY_THRESHOLD", "0.45"),
    ("TOP_K", "12"),
    ("OLLAMA_TIMEOUT", "300"),
    ("WHISPER_MODEL", "large-v3"),
    ("WHISPER_MODEL", "distil-large-v3"),
    ("WHISPER_MODEL", "Systran/faster-whisper-small"),
    ("WHISPER_DEVICE", "cuda"),
    ("WHISPER_DEVICE", ""),
    ("WHISPER_COMPUTE_TYPE", "int8_float16"),
    ("WHISPER_VAD", "0"),
    ("WHISPER_CPU_THREADS", "0"),
    ("CLEAR_ON_START", "1"),
])
def test_a_good_value_is_accepted(name, value):
    assert preflight.check_config({name: value}) == []


def test_a_local_whisper_model_directory_is_accepted(tmp_path):
    assert preflight.check_config({"WHISPER_MODEL": str(tmp_path)}) == []


def test_every_bad_value_is_reported_at_once():
    errors = preflight.check_config({"TOP_K": "x", "WHISPER_VAD": "maybe"})
    assert len(errors) == 2


def test_the_defaults_match_the_ones_the_scripts_use(monkeypatch):
    for name in preflight.DEFAULTS:
        monkeypatch.delenv(name, raising=False)
    import ast
    for script, names in (("dashboard.py", ("OLLAMA_URL", "EMBED_MODEL", "ANSWER_MODEL")),
                          ("preprocess_json.py", ("OLLAMA_URL", "EMBED_MODEL"))):
        for node in ast.parse((ROOT / script).read_text()).body:
            if isinstance(node, ast.Assign) and getattr(node.targets[0], "id", None) in names:
                assert node.value.args[1].value == preflight.DEFAULTS[node.targets[0].id], script


# ── ffmpeg and faster-whisper ───────────────────────────────────────────────

def test_ffmpeg_on_path_is_fine(monkeypatch):
    monkeypatch.setattr(preflight.shutil, "which", lambda name: "/usr/bin/ffmpeg")
    assert preflight.check_ffmpeg() == []


def test_missing_ffmpeg_blocks_only_processing(monkeypatch):
    monkeypatch.setattr(preflight.shutil, "which", lambda name: None)
    [problem] = preflight.check_ffmpeg()
    assert "ffmpeg" in problem.message
    assert "install ffmpeg" in problem.fix
    assert problem.blocks == (PROCESS,)


def test_faster_whisper_that_imports_is_fine(monkeypatch):
    monkeypatch.setattr(preflight.importlib, "import_module", lambda name: object())
    assert preflight.check_whisper() == []


def test_faster_whisper_that_does_not_import_says_how_to_install_it(monkeypatch):
    def fail(name):
        raise ImportError("No module named 'faster_whisper'")
    monkeypatch.setattr(preflight.importlib, "import_module", fail)
    [problem] = preflight.check_whisper()
    assert "No module named 'faster_whisper'" in problem.message
    assert problem.fix.endswith("-m pip install -r requirements.txt")
    assert problem.blocks == (PROCESS,)


def test_an_isolated_check_reads_the_child_process_error(monkeypatch):
    def run(command, **kwargs):
        return subprocess.CompletedProcess(
            command, 1, "", "Traceback...\nModuleNotFoundError: No module named 'av'\n")
    monkeypatch.setattr(preflight.subprocess, "run", run)
    [problem] = preflight.check_whisper(isolated=True)
    assert "No module named 'av'" in problem.message


def test_an_isolated_check_that_succeeds_reports_nothing(monkeypatch):
    monkeypatch.setattr(
        preflight.subprocess, "run",
        lambda command, **kwargs: subprocess.CompletedProcess(command, 0, "", ""))
    assert preflight.check_whisper(isolated=True) == []


# ── ollama and its models ───────────────────────────────────────────────────

def test_ollama_with_both_models_is_fine(monkeypatch):
    _ollama(monkeypatch, _tags("nomic-embed-text:latest", "qwen2.5:1.5b"))
    assert preflight.check_ollama("http://x", "nomic-embed-text", "qwen2.5:1.5b") == []


def test_a_name_without_a_tag_means_latest(monkeypatch):
    _ollama(monkeypatch, _tags("nomic-embed-text:latest"))
    assert preflight.check_ollama("http://x", "nomic-embed-text") == []


def test_a_different_tag_is_not_the_same_model(monkeypatch):
    _ollama(monkeypatch, _tags("qwen2.5:7b", "nomic-embed-text:latest"))
    [problem] = preflight.check_ollama("http://x", "nomic-embed-text", "qwen2.5:1.5b")
    assert problem.fix == "ollama pull qwen2.5:1.5b"


def test_a_missing_answer_model_blocks_only_asking(monkeypatch):
    _ollama(monkeypatch, _tags("nomic-embed-text:latest"))
    [problem] = preflight.check_ollama("http://x", "nomic-embed-text", "qwen2.5:1.5b")
    assert problem.fix == "ollama pull qwen2.5:1.5b"
    assert problem.blocks == (ASK,)


def test_a_missing_embedding_model_blocks_both(monkeypatch):
    _ollama(monkeypatch, _tags("qwen2.5:1.5b"))
    [problem] = preflight.check_ollama("http://x", "nomic-embed-text", "qwen2.5:1.5b")
    assert problem.fix == "ollama pull nomic-embed-text"
    assert set(problem.blocks) == {PROCESS, ASK}


@pytest.mark.parametrize("failure", [
    requests.ConnectionError("refused"), requests.ConnectTimeout("slow"),
])
def test_an_unreachable_ollama_says_how_to_start_it(monkeypatch, failure):
    _ollama(monkeypatch, failure)
    [problem] = preflight.check_ollama("http://x:1", "nomic-embed-text", "qwen2.5:1.5b")
    assert "http://x:1" in problem.message
    assert problem.fix == "ollama serve"
    assert set(problem.blocks) == {PROCESS, ASK}


@pytest.mark.parametrize("reply", [
    _Reply(status=500), _Reply(ValueError("not json")), _Reply({"unexpected": 1}),
    _Reply({"models": None}),
])
def test_a_server_that_is_not_ollama_is_reported_not_raised(monkeypatch, reply):
    _ollama(monkeypatch, reply)
    [problem] = preflight.check_ollama("http://x", "nomic-embed-text")
    assert "did not return an Ollama model list" in problem.message
    assert set(problem.blocks) == {PROCESS, ASK}


def test_the_blocked_actions_are_the_union(monkeypatch):
    _ollama(monkeypatch, _tags("nomic-embed-text"))
    monkeypatch.setattr(preflight.shutil, "which", lambda name: None)
    problems = preflight.check_ffmpeg() + preflight.check_ollama(
        "http://x", "nomic-embed-text", "qwen2.5:1.5b")
    assert preflight.blocked_actions(problems) == {PROCESS, ASK}
    assert preflight.blocked_actions(problems[1:]) == {ASK}
    assert preflight.blocked_actions([]) == frozenset()


def test_format_lists_each_problem_with_its_fix():
    problems = [
        preflight.Problem("a", "A is missing.", "install a", (PROCESS,)),
        preflight.Problem("b", "B is missing.", "install b", (ASK,)),
    ]
    assert preflight.format_problems(problems) == (
        "- A is missing.\n    Fix: install a\n- B is missing.\n    Fix: install b")


# ── query-time errors ───────────────────────────────────────────────────────

def _http_error(status, body):
    return requests.HTTPError("boom", response=_Reply(body, status))


def test_a_read_timeout_is_readable():
    text = preflight.describe_ollama_error(requests.ReadTimeout("slow"), "qwen2.5:1.5b")
    assert "did not answer in time" in text
    assert "qwen2.5:1.5b" in text
    assert "OLLAMA_TIMEOUT" in text


def test_a_refused_connection_says_how_to_start_ollama():
    text = preflight.describe_ollama_error(requests.ConnectionError("refused"), "m")
    assert text == "Cannot connect to Ollama. Start it with: ollama serve"


def test_a_connect_timeout_is_a_connection_problem_not_a_slow_model():
    text = preflight.describe_ollama_error(requests.ConnectTimeout("no route"), "m")
    assert text.startswith("Cannot connect to Ollama")


def test_a_missing_model_says_how_to_pull_it():
    error = _http_error(404, {"error": 'model "qwen2.5:1.5b" not found, try pulling it first'})
    text = preflight.describe_ollama_error(error, "qwen2.5:1.5b")
    assert text == "The model qwen2.5:1.5b is not pulled. Pull it with: ollama pull qwen2.5:1.5b"


def test_another_http_error_carries_ollamas_own_explanation():
    text = preflight.describe_ollama_error(
        _http_error(500, {"error": "model requires more system memory"}), "m")
    assert text == "Ollama returned an error (HTTP 500): model requires more system memory"


def test_an_http_error_with_no_body_still_reads_well():
    text = preflight.describe_ollama_error(_http_error(502, ValueError("html")), "m")
    assert text == "Ollama returned an error (HTTP 502)."


def test_a_404_that_is_not_about_a_model_is_not_blamed_on_the_model():
    text = preflight.describe_ollama_error(_http_error(404, ValueError("html")), "m")
    assert "pull" not in text


def test_an_unreadable_reply_is_named():
    text = preflight.describe_ollama_error(KeyError("embeddings"), "m")
    assert "cannot read" in text and "KeyError" in text


def test_anything_else_keeps_its_type_and_message():
    text = preflight.describe_ollama_error(RuntimeError("odd"), "m")
    assert text == "Something went wrong (RuntimeError: odd)."


# ── the scripts refuse to start ─────────────────────────────────────────────

def _run(script, tmp_path, **env):
    """Run a pipeline script in an empty directory with a controlled environment."""
    clean = {k: v for k, v in os.environ.items() if k not in preflight.DEFAULTS}
    clean.update(env)
    return subprocess.run(
        [sys.executable, str(ROOT / script)], cwd=tmp_path, env=clean,
        capture_output=True, text=True, timeout=60,
    )


def test_require_returns_quietly_when_nothing_is_wrong(monkeypatch, capsys):
    monkeypatch.setattr(preflight.shutil, "which", lambda name: "/usr/bin/ffmpeg")
    preflight.require(ffmpeg=True)
    assert capsys.readouterr().err == ""


def test_require_exits_1_and_prints_the_fix(monkeypatch, capsys):
    monkeypatch.setattr(preflight.shutil, "which", lambda name: None)
    with pytest.raises(SystemExit) as stop:
        preflight.require(ffmpeg=True)
    assert stop.value.code == 1
    err = capsys.readouterr().err
    assert "Cannot start" in err and "Fix:" in err


def test_require_reports_a_bad_setting_before_checking_anything(monkeypatch, capsys):
    monkeypatch.setenv("OLLAMA_URL", "not a url")
    monkeypatch.setattr(
        preflight.requests, "get", lambda *a, **k: pytest.fail("Ollama was contacted"))
    with pytest.raises(SystemExit):
        preflight.require(embeddings=True)
    assert "OLLAMA_URL='not a url'" in capsys.readouterr().err


def test_video_to_mp3_stops_without_ffmpeg(tmp_path):
    done = _run("video_to_mp3.py", tmp_path, PATH=str(tmp_path))
    assert done.returncode == 1
    assert "ffmpeg was not found" in done.stderr
    assert "Traceback" not in done.stderr


def test_mp3_to_json_stops_without_faster_whisper(tmp_path):
    fake = tmp_path / "fake"
    fake.mkdir()
    (fake / "faster_whisper.py").write_text("raise ImportError('simulated')\n")
    done = _run("mp3_to_json.py", tmp_path, PYTHONPATH=str(fake))
    assert done.returncode == 1
    assert "faster-whisper could not be imported" in done.stderr
    assert "simulated" in done.stderr
    assert "Traceback" not in done.stderr


def test_preprocess_json_stops_when_ollama_is_down(tmp_path):
    done = _run("preprocess_json.py", tmp_path, OLLAMA_URL="http://127.0.0.1:9")
    assert done.returncode == 1
    assert "Ollama is not reachable at http://127.0.0.1:9" in done.stderr
    assert "ollama serve" in done.stderr
    assert "Traceback" not in done.stderr


def test_a_script_reports_a_bad_setting_instead_of_a_traceback(tmp_path):
    done = _run("preprocess_json.py", tmp_path, EMBED_CONCURRENCY="lots")
    assert done.returncode == 1
    assert "EMBED_CONCURRENCY='lots' must be a whole number" in done.stderr
    assert "Traceback" not in done.stderr
