"""The launcher's preflight: what it reports, and the pulls it offers.

preflight.py talks to Ollama over HTTP and to the shell for ffmpeg, so the
tests run it against a throwaway server bound to 127.0.0.1 and a PATH they
control. Nothing here needs Ollama, ffmpeg or the network.
"""
import json
import sys
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parents[1]))
import preflight  # noqa: E402


@pytest.fixture
def ollama():
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
    """A PATH holding only the fake executables a test installs."""
    monkeypatch.setenv("PATH", str(tmp_path))
    for var in ("OLLAMA_URL", "EMBED_MODEL", "ANSWER_MODEL", "RAG_AUTO_PULL"):
        monkeypatch.delenv(var, raising=False)

    def install(name, script="exit 0"):
        path = tmp_path / name
        path.write_text(f"#!/bin/sh\n{script}\n")
        path.chmod(0o755)
        return path

    return install


def test_everything_present_passes(ollama, tools, monkeypatch, capsys):
    url, installed = ollama
    installed += ["nomic-embed-text:latest", "qwen2.5:1.5b"]
    tools("ffmpeg")
    monkeypatch.setenv("OLLAMA_URL", url)

    assert preflight.main() == 0
    assert "missing" not in capsys.readouterr().out


def test_missing_model_is_named_with_its_pull_command(ollama, tools, monkeypatch, capsys):
    url, installed = ollama
    installed.append("nomic-embed-text:latest")
    tools("ffmpeg")
    monkeypatch.setenv("OLLAMA_URL", url)

    assert preflight.main() == 1
    assert "ollama pull qwen2.5:1.5b" in capsys.readouterr().out


def test_configured_models_are_the_ones_checked(ollama, tools, monkeypatch, capsys):
    url, installed = ollama
    installed += ["nomic-embed-text:latest", "qwen2.5:1.5b"]
    tools("ffmpeg")
    monkeypatch.setenv("OLLAMA_URL", url)
    monkeypatch.setenv("ANSWER_MODEL", "llama3.2:3b")

    assert preflight.main() == 1
    assert "ollama pull llama3.2:3b" in capsys.readouterr().out


def test_missing_ffmpeg_gets_an_install_hint(ollama, tools, monkeypatch, capsys):
    url, installed = ollama
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


def test_auto_pull_pulls_each_missing_model(ollama, tools, monkeypatch, tmp_path):
    url, installed = ollama
    installed.append("nomic-embed-text:latest")
    tools("ffmpeg")
    tools("ollama", f'echo "$@" >> {tmp_path}/pulls')
    monkeypatch.setenv("OLLAMA_URL", url)
    monkeypatch.setenv("RAG_AUTO_PULL", "1")

    preflight.main()

    assert (tmp_path / "pulls").read_text().split("\n")[:-1] == ["pull qwen2.5:1.5b"]


def test_without_auto_pull_or_a_terminal_nothing_is_pulled(ollama, tools, monkeypatch, tmp_path):
    url, _ = ollama
    tools("ffmpeg")
    tools("ollama", f'echo "$@" >> {tmp_path}/pulls')
    monkeypatch.setenv("OLLAMA_URL", url)

    assert preflight.main() == 1
    assert not (tmp_path / "pulls").exists()


def test_a_failed_pull_leaves_the_model_reported_missing(ollama, tools, monkeypatch):
    url, installed = ollama
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
def test_model_matching_follows_ollama_tags(requested, installed, present):
    assert (requested in preflight.missing_models([requested], installed)) is (not present)
