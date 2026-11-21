"""Startup checks: what is reported, what it blocks, and what the scripts do about it."""
import json
import os
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
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


# ── python preflight.py: the launcher's entry point ─────────────────────────
#
# These run main() against a throwaway Ollama bound to 127.0.0.1 and a PATH the
# test controls. Nothing here needs Ollama, ffmpeg or the network.

@pytest.fixture
def fake_ollama():
    """A fake Ollama whose installed models the test chooses."""
    installed = []

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path != "/api/tags":
                self.send_error(404)
                return
            body = json.dumps({"models": [{"name": n} for n in installed]}).encode()
            self.send_response(200)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):
            pass

    server = HTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{server.server_port}", installed
    server.shutdown()
    server.server_close()


@pytest.fixture
def tools(tmp_path, monkeypatch):
    """A PATH holding only the fake executables a test installs.

    faster-whisper is reported as importable: these tests are about ffmpeg,
    Ollama and the pulls, and the real import is slow.
    """
    monkeypatch.setenv("PATH", str(tmp_path))
    for var in ("OLLAMA_URL", "EMBED_MODEL", "ANSWER_MODEL", "RAG_AUTO_PULL"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setattr(preflight, "check_whisper", lambda isolated=False: [])

    def install(name, script="exit 0"):
        path = tmp_path / name
        path.write_text(f"#!/bin/sh\n{script}\n")
        path.chmod(0o755)
        return path

    return install


def test_everything_present_passes(fake_ollama, tools, monkeypatch, capsys):
    url, installed = fake_ollama
    installed += ["nomic-embed-text:latest", "qwen2.5:1.5b"]
    tools("ffmpeg")
    monkeypatch.setenv("OLLAMA_URL", url)

    assert preflight.main() == 0
    assert "missing" not in capsys.readouterr().out


def test_missing_model_is_named_with_its_pull_command(fake_ollama, tools, monkeypatch, capsys):
    url, installed = fake_ollama
    installed.append("nomic-embed-text:latest")
    tools("ffmpeg")
    monkeypatch.setenv("OLLAMA_URL", url)

    assert preflight.main() == 1
    assert "ollama pull qwen2.5:1.5b" in capsys.readouterr().out


def test_configured_models_are_the_ones_checked(fake_ollama, tools, monkeypatch, capsys):
    url, installed = fake_ollama
    installed += ["nomic-embed-text:latest", "qwen2.5:1.5b"]
    tools("ffmpeg")
    monkeypatch.setenv("OLLAMA_URL", url)
    monkeypatch.setenv("ANSWER_MODEL", "llama3.2:3b")

    assert preflight.main() == 1
    assert "ollama pull llama3.2:3b" in capsys.readouterr().out


def test_missing_ffmpeg_gets_an_install_hint(fake_ollama, tools, monkeypatch, capsys):
    url, installed = fake_ollama
    installed += ["nomic-embed-text", "qwen2.5:1.5b"]
    monkeypatch.setenv("OLLAMA_URL", url)

    assert preflight.main() == 1
    assert "ffmpeg" in capsys.readouterr().out.lower()


def test_unreachable_ollama_lists_the_pulls_instead_of_guessing(tools, monkeypatch, capsys):
    tools("ffmpeg")
    monkeypatch.setenv("OLLAMA_URL", "http://127.0.0.1:1")

    assert preflight.main() == 1
    out = capsys.readouterr().out
    assert "ollama pull nomic-embed-text" in out
    assert "ollama pull qwen2.5:1.5b" in out


def test_auto_pull_pulls_each_missing_model(fake_ollama, tools, monkeypatch, tmp_path):
    url, installed = fake_ollama
    installed.append("nomic-embed-text:latest")
    tools("ffmpeg")
    tools("ollama", f'echo "$@" >> {tmp_path}/pulls')
    monkeypatch.setenv("OLLAMA_URL", url)
    monkeypatch.setenv("RAG_AUTO_PULL", "1")

    preflight.main()

    assert (tmp_path / "pulls").read_text().split("\n")[:-1] == ["pull qwen2.5:1.5b"]


def test_without_auto_pull_or_a_terminal_nothing_is_pulled(fake_ollama, tools, monkeypatch, tmp_path):
    url, _ = fake_ollama
    tools("ffmpeg")
    tools("ollama", f'echo "$@" >> {tmp_path}/pulls')
    monkeypatch.setenv("OLLAMA_URL", url)

    assert preflight.main() == 1
    assert not (tmp_path / "pulls").exists()


def test_a_failed_pull_leaves_the_model_reported_missing(fake_ollama, tools, monkeypatch):
    url, installed = fake_ollama
    installed.append("nomic-embed-text:latest")
    tools("ffmpeg")
    tools("ollama", "exit 1")
    monkeypatch.setenv("OLLAMA_URL", url)
    monkeypatch.setenv("RAG_AUTO_PULL", "1")

    assert preflight.main() == 1


@pytest.mark.parametrize("requested, installed, present", [
    ("nomic-embed-text", ["nomic-embed-text:latest"], True),
    ("qwen2.5:1.5b", ["qwen2.5:1.5b"], True),
    ("qwen2.5:1.5b", ["qwen2.5:7b"], False),
    ("qwen2.5", ["qwen2.5:7b"], False),
])
def test_model_matching_follows_ollama_tags(monkeypatch, requested, installed, present):
    _ollama(monkeypatch, _tags(*installed))
    problems = preflight.check_ollama("http://x", requested)
    assert (problems == []) is present
    assert present or problems[0].model == requested


def test_the_command_line_pulls_only_what_a_check_says_is_pullable(
    fake_ollama, tools, monkeypatch, tmp_path
):
    url, installed = fake_ollama
    installed += ["nomic-embed-text:latest", "qwen2.5:1.5b"]
    tools("ffmpeg")
    tools("ollama", f'echo "$@" >> {tmp_path}/pulls')
    monkeypatch.setenv("OLLAMA_URL", url)
    monkeypatch.setenv("RAG_AUTO_PULL", "1")

    assert preflight.main() == 0
    assert not (tmp_path / "pulls").exists()


def test_a_pull_that_works_makes_the_next_check_pass(fake_ollama, tools, monkeypatch, capsys):
    url, installed = fake_ollama
    installed.append("nomic-embed-text:latest")
    tools("ffmpeg")
    tools("ollama", "true")
    monkeypatch.setenv("OLLAMA_URL", url)
    monkeypatch.setenv("RAG_AUTO_PULL", "1")
    monkeypatch.setattr(preflight, "_pull", lambda models, u: installed.extend(models))

    assert preflight.main() == 0
    assert "missing" not in capsys.readouterr().out


def test_the_command_line_checks_faster_whisper_in_a_child_process(
    fake_ollama, tools, monkeypatch, capsys
):
    url, installed = fake_ollama
    installed += ["nomic-embed-text:latest", "qwen2.5:1.5b"]
    tools("ffmpeg")
    monkeypatch.setenv("OLLAMA_URL", url)
    asked = []
    problem = preflight.Problem("faster-whisper", "faster-whisper could not be imported (x).",
                                "pip install -r requirements.txt", (PROCESS,))
    monkeypatch.setattr(preflight, "check_whisper",
                        lambda isolated=False: asked.append(isolated) or [problem])

    assert preflight.main() == 1
    assert asked == [True]
    assert "pip install -r requirements.txt" in capsys.readouterr().out


def test_the_command_line_reports_a_bad_setting_by_name(tools, monkeypatch, capsys):
    monkeypatch.setenv("TOP_K", "many")
    assert preflight.main() == 1
    assert "TOP_K" in capsys.readouterr().out


def test_ollama_is_checked_without_requests_installed(fake_ollama, monkeypatch):
    url, installed = fake_ollama
    installed.append("nomic-embed-text:latest")
    monkeypatch.setattr(preflight, "requests", None)

    assert preflight.check_ollama(url, "nomic-embed-text") == []
    [problem] = preflight.check_ollama(url, "nomic-embed-text", "qwen2.5:1.5b")
    assert problem.model == "qwen2.5:1.5b"
    [problem] = preflight.check_ollama("http://127.0.0.1:1", "nomic-embed-text")
    assert problem.fix == "ollama serve"


def test_the_script_runs_with_only_the_standard_library(fake_ollama, tmp_path):
    """The launcher runs it before requirements are installed."""
    url, installed = fake_ollama
    installed += ["nomic-embed-text:latest", "qwen2.5:1.5b"]
    ffmpeg = tmp_path / "ffmpeg"
    ffmpeg.write_text("#!/bin/sh\nexit 0\n")
    ffmpeg.chmod(0o755)
    blocked = (
        "import sys, runpy\n"
        "sys.modules['requests'] = None\n"
        "sys.modules['faster_whisper'] = None\n"
        f"sys.exit(runpy.run_path({str(ROOT / 'preflight.py')!r}, run_name='__main__') or 0)\n"
    )
    done = subprocess.run(
        [sys.executable, "-c", blocked], capture_output=True, text=True, timeout=120,
        env={"PATH": f"{tmp_path}{os.pathsep}{os.environ['PATH']}", "OLLAMA_URL": url},
    )
    assert "Traceback" not in done.stderr, done.stderr
    assert "ok       model nomic-embed-text" in done.stdout
    assert "ok       model qwen2.5:1.5b" in done.stdout
