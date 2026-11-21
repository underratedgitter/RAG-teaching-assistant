"""Check what the dashboard needs before it opens, and offer to fetch the rest.

Standard library only, so the launcher can run it before anything is installed.
Exits 0 when ffmpeg, Ollama and both models are ready and 1 otherwise, naming
the exact command that fixes each gap. Models are pulled only when asked for:
RAG_AUTO_PULL=1, or a yes at the prompt on an interactive terminal.
"""
import json
import os
import platform
import shutil
import subprocess
import sys
import urllib.error
import urllib.request

FFMPEG_INSTALL = {
    "Darwin": "brew install ffmpeg",
    "Linux": "sudo apt install ffmpeg   (or your distribution's equivalent)",
    "Windows": "winget install ffmpeg",
}


def installed_models(url):
    """Names of the models Ollama has, or None when it cannot be reached."""
    try:
        with urllib.request.urlopen(f"{url.rstrip('/')}/api/tags", timeout=3) as r:
            return [m["name"] for m in json.load(r)["models"]]
    except (urllib.error.URLError, OSError, ValueError, KeyError, TypeError):
        return None


def missing_models(required, installed):
    """The required models Ollama does not have; a bare name means :latest."""
    have = set(installed)
    return [m for m in required if (m if ":" in m else f"{m}:latest") not in have]


def pull(models, url):
    """Pull each model with the ollama CLI; a failed pull shows as still missing."""
    env = {**os.environ, "OLLAMA_HOST": url}
    for m in models:
        subprocess.run(["ollama", "pull", m], env=env)


def wants_pull():
    if os.environ.get("RAG_AUTO_PULL") == "1":
        return True
    if not (sys.stdin.isatty() and sys.stdout.isatty()):
        return False
    return input("Pull the missing models now? [Y/n] ").strip().lower() in ("", "y", "yes")


def main():
    url = os.environ.get("OLLAMA_URL", "http://localhost:11434")
    models = list(dict.fromkeys([
        os.environ.get("EMBED_MODEL", "nomic-embed-text"),
        os.environ.get("ANSWER_MODEL", "qwen2.5:1.5b"),
    ]))
    ready = True

    if shutil.which("ffmpeg"):
        print("ok       ffmpeg")
    else:
        hint = FFMPEG_INSTALL.get(platform.system(), "see https://ffmpeg.org/download.html")
        print(f"missing  ffmpeg (needed to process videos): {hint}")
        ready = False

    installed = installed_models(url)
    if installed is None:
        print(f"missing  Ollama is not answering at {url}")
        if shutil.which("ollama"):
            print("         the dashboard will try to start it; or run: ollama serve")
        else:
            print("         install it from https://ollama.com/download")
        for m in models:
            print(f"         then: ollama pull {m}")
        return 1

    print(f"ok       Ollama at {url}")
    absent = missing_models(models, installed)
    for m in models:
        print(f"missing  model {m}: ollama pull {m}" if m in absent else f"ok       model {m}")

    if absent and shutil.which("ollama") and wants_pull():
        pull(absent, url)
        absent = missing_models(models, installed_models(url) or [])
    return 0 if ready and not absent else 1


if __name__ == "__main__":
    sys.exit(main())
