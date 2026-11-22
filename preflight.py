"""Startup checks and settings validation, shared by the dashboard, the scripts
and the launcher.

Everything the pipeline needs from outside Python (ffmpeg, Ollama and its
models, faster-whisper) is checked here, and each missing piece becomes a
Problem that says what is wrong, the exact command that fixes it, and which
actions cannot work until it is fixed. A check never raises: a missing tool
is an expected state on a fresh machine, not an error in the program.

Run as `python preflight.py` it prints one line per piece with the fix for each
gap, offers to pull missing Ollama models (RAG_AUTO_PULL=1, or a yes at the
prompt on a terminal) and exits 0 when everything is ready, 1 otherwise. That
entry point needs only the standard library, so the launcher can run it before
requirements are installed: requests is optional, and faster-whisper is only
ever imported in a child process.
"""
import importlib
import json
import math
import os
import shutil
import socket
import subprocess
import sys
import urllib.error
import urllib.request
from dataclasses import dataclass
from urllib.parse import urlparse

try:
    import requests
except ImportError:     # the launcher runs this file before requirements are installed
    requests = None

# The two things a user can do in the dashboard. A Problem lists the ones it
# blocks, so only those are disabled.
PROCESS = "process"
ASK = "ask"

# Where the scripts look when the environment does not say. Kept in step with
# the scripts' own defaults by a test.
DEFAULTS = {
    "OLLAMA_URL": "http://localhost:11434",
    "EMBED_MODEL": "nomic-embed-text",
    "ANSWER_MODEL": "qwen2.5:1.5b",
}

# Seconds to wait for Ollama's model list. It is a local call, so a host that
# does not answer in this long is not going to.
CHECK_TIMEOUT = 3
# Importing faster-whisper loads CTranslate2, which can take a while on a cold disk.
IMPORT_TIMEOUT = 60

# Exit status of the child that looks for the Whisper weights in the cache,
# when they are not there.
NOT_CACHED = 3

_FFMPEG_INSTALL = {"darwin": "brew install ffmpeg", "win32": "winget install ffmpeg"}


@dataclass(frozen=True)
class Problem:
    """One thing that is missing, how to fix it, and what it stops working."""

    piece: str
    message: str
    fix: str
    blocks: tuple[str, ...]
    # The Ollama model that `ollama pull` would supply, when that is the fix.
    model: str | None = None


# ── settings ────────────────────────────────────────────────────────────────

def _integer(low, high):
    def check(value):
        try:
            number = int(value)
        except ValueError:
            raise ValueError("must be a whole number") from None
        if not low <= number <= high:
            raise ValueError(f"must be between {low} and {high}")
    return check


def _number(low, high):
    def check(value):
        try:
            number = float(value)
        except ValueError:
            raise ValueError("must be a number") from None
        if not math.isfinite(number) or not low <= number <= high:
            raise ValueError(f"must be a number between {low} and {high}")
    return check


def _choice(*options):
    def check(value):
        if value not in options:
            raise ValueError("must be one of: " + ", ".join(options))
    return check


def _text(value):
    if not value.strip():
        raise ValueError("must not be empty")


def _url(value):
    parts = urlparse(value)
    if parts.scheme not in ("http", "https") or not parts.netloc:
        raise ValueError("must be an http(s) address such as " + DEFAULTS["OLLAMA_URL"])


_WHISPER_SIZES = frozenset({
    "tiny", "tiny.en", "base", "base.en", "small", "small.en", "medium", "medium.en",
    "large", "large-v1", "large-v2", "large-v3", "turbo", "large-v3-turbo",
})


def _whisper_model(value):
    # A size name, a distil-* name, a Hugging Face repo id or a local directory
    # all load; anything else is a typo that would fail after a long download.
    if (value in _WHISPER_SIZES or value.startswith("distil-")
            or "/" in value or os.path.isdir(value)):
        return
    raise ValueError("must be a Whisper size such as tiny, base, small, medium or large-v3")


# (name, validator, whether an empty value counts as unset). Empty means unset
# only where the code that reads the variable already treats it that way.
SETTINGS = (
    ("OLLAMA_URL", _url, False),
    ("EMBED_MODEL", _text, False),
    ("ANSWER_MODEL", _text, False),
    ("EMBED_CONCURRENCY", _integer(1, 32), False),
    ("SIMILARITY_THRESHOLD", _number(-1.0, 1.0), False),
    ("TOP_K", _integer(1, 100), False),
    ("FALLBACK_K", _integer(1, 100), False),
    ("OLLAMA_TIMEOUT", _integer(1, 3600), False),
    ("NUM_PREDICT", _integer(16, 4096), False),
    ("WHISPER_MODEL", _whisper_model, False),
    ("WHISPER_DEVICE", _choice("cpu", "cuda"), True),
    ("WHISPER_COMPUTE_TYPE", _choice(
        "int8", "int8_float16", "int8_float32", "int16", "float16", "bfloat16", "float32",
    ), False),
    ("WHISPER_BEAM_SIZE", _integer(1, 20), False),
    ("WHISPER_VAD", _choice("0", "1"), False),
    ("WHISPER_CPU_THREADS", _integer(0, 512), False),
    ("CLEAR_ON_START", _choice("0", "1"), True),
    ("RAG_NO_GUI", _choice("0", "1"), True),
    ("RAG_AUTO_PULL", _choice("0", "1"), True),
)


def check_config(environ=None):
    """One message per environment variable that holds an unusable value."""
    environ = os.environ if environ is None else environ
    errors = []
    for name, validator, blank_is_unset in SETTINGS:
        value = environ.get(name)
        if value is None or (blank_is_unset and not value.strip()):
            continue
        try:
            validator(value)
        except ValueError as exc:
            errors.append(f"{name}={value!r} {exc}. Set a valid value or unset it.")
    return errors


# ── checks ──────────────────────────────────────────────────────────────────

def check_ffmpeg():
    if shutil.which("ffmpeg"):
        return []
    return [Problem(
        "ffmpeg", "ffmpeg was not found on PATH, so videos cannot be converted to audio.",
        _FFMPEG_INSTALL.get(sys.platform, "sudo apt install ffmpeg"), (PROCESS,),
    )]


def check_whisper(isolated=False):
    """Whether faster-whisper imports.

    The dashboard asks with isolated=True, so the check runs in a child
    process and CTranslate2 is never loaded into the interface itself.
    """
    error = None
    if isolated:
        try:
            done = subprocess.run(
                [sys.executable, "-c", "import faster_whisper"],
                capture_output=True, text=True, errors="replace", timeout=IMPORT_TIMEOUT,
            )
            if done.returncode != 0:
                lines = done.stderr.strip().splitlines()
                error = lines[-1] if lines else f"exit {done.returncode}"
        except (OSError, subprocess.TimeoutExpired) as exc:
            error = str(exc) or type(exc).__name__
    else:
        try:
            importlib.import_module("faster_whisper")
        except Exception as exc:    # a broken native library raises more than ImportError
            error = f"{type(exc).__name__}: {exc}"
    if error is None:
        return []
    return [Problem(
        "faster-whisper", f"faster-whisper could not be imported ({error}).",
        f"{sys.executable} -m pip install -r requirements.txt", (PROCESS,),
    )]


# Finds the weights mp3_to_json.py will load, in a child process so that
# CTranslate2 is never loaded into the caller. Prints the model's name; exits
# NOT_CACHED when its files are not in the Hugging Face cache. The default
# size follows mp3_to_json.py: small on CUDA, otherwise base.
_WHISPER_PROBE = f"""
import os, sys
import ctranslate2
from faster_whisper.utils import download_model
name = os.environ.get("WHISPER_MODEL")
if not name:
    forced = os.environ.get("WHISPER_DEVICE")
    cuda = forced == "cuda" if forced else ctranslate2.get_cuda_device_count() > 0
    name = "small" if cuda else "base"
print(name)
if not os.path.isdir(name):
    try:
        download_model(name, local_files_only=True)
    except Exception:
        sys.exit({NOT_CACHED})
"""


def _hub_reachable():
    """Whether the Hugging Face Hub, where Whisper weights come from, answers."""
    if os.environ.get("HF_HUB_OFFLINE") == "1":
        return False
    parts = urlparse(os.environ.get("HF_ENDPOINT") or "https://huggingface.co")
    try:
        with socket.create_connection(
            (parts.hostname or "huggingface.co", parts.port or 443), timeout=CHECK_TIMEOUT
        ):
            return True
    except OSError:
        return False


def check_whisper_model():
    """Whether the Whisper weights are on disk, or can still be downloaded.

    The first transcription downloads them, so a missing model is only a
    problem when the Hub cannot be reached either. Anything that stops the
    probe running (a broken install) is check_whisper's to report.
    """
    try:
        done = subprocess.run(
            [sys.executable, "-c", _WHISPER_PROBE],
            capture_output=True, text=True, errors="replace", timeout=IMPORT_TIMEOUT,
        )
    except (OSError, subprocess.TimeoutExpired):
        return []
    if done.returncode != NOT_CACHED or _hub_reachable():
        return []
    name = done.stdout.strip().splitlines()[-1] if done.stdout.strip() else "Whisper"
    return [Problem(
        "whisper model",
        f"The Whisper model '{name}' is not downloaded and the Hugging Face Hub "
        "cannot be reached, so the first transcription cannot fetch it.",
        "connect to the internet and run Process Videos once (the weights are then "
        f"kept and need no network), or download '{name}' elsewhere and set "
        "WHISPER_MODEL to its folder",
        (PROCESS,),
    )]


def _is_installed(model, installed):
    # "name" means "name:latest" to Ollama.
    return model in installed or (":" not in model and f"{model}:latest" in installed)


def _get_tags(url):
    """Ollama's model list, as the decoded JSON.

    Uses requests when it is installed and the standard library when it is
    not. Either way an unreachable host raises ConnectionError or Timeout
    (requests' or the builtin ones) and a bad reply raises ValueError.
    """
    endpoint = f"{url.rstrip('/')}/api/tags"
    if requests is not None:
        reply = requests.get(endpoint, timeout=CHECK_TIMEOUT)
        reply.raise_for_status()
        return reply.json()
    try:
        with urllib.request.urlopen(endpoint, timeout=CHECK_TIMEOUT) as reply:
            return json.load(reply)
    except urllib.error.HTTPError as exc:
        raise ValueError(f"HTTP {exc.code}") from exc
    except urllib.error.URLError as exc:
        raise ConnectionError(str(exc.reason)) from exc


# What "nobody is listening" looks like, from either HTTP client.
_UNREACHABLE = (ConnectionError, TimeoutError) + (
    (requests.ConnectionError, requests.Timeout) if requests is not None else ())
_BAD_REPLY = (ValueError, KeyError, TypeError, AttributeError) + (
    (requests.RequestException,) if requests is not None else ())


def check_ollama(url, embed_model, answer_model=None):
    """Whether Ollama answers at `url` and has the models that were asked for."""
    try:
        installed = {
            entry.get("name") or entry.get("model") for entry in _get_tags(url)["models"]
        }
    except _UNREACHABLE:
        return [Problem(
            "ollama", f"Ollama is not reachable at {url}. Is it running, and is OLLAMA_URL right?",
            "ollama serve", (PROCESS, ASK),
        )]
    except _BAD_REPLY as exc:
        return [Problem(
            "ollama", f"{url} did not return an Ollama model list ({exc}).",
            f"curl {url.rstrip('/')}/api/tags", (PROCESS, ASK),
        )]

    problems = []
    if not _is_installed(embed_model, installed):
        problems.append(Problem(
            f"model {embed_model}", f"The embedding model {embed_model} is not pulled.",
            f"ollama pull {embed_model}", (PROCESS, ASK), embed_model,
        ))
    if answer_model and not _is_installed(answer_model, installed):
        problems.append(Problem(
            f"model {answer_model}", f"The answer model {answer_model} is not pulled.",
            f"ollama pull {answer_model}", (ASK,), answer_model,
        ))
    return problems


def run_checks(url, embed_model, answer_model):
    """Every check the dashboard needs, in the order a user would fix them."""
    whisper = check_whisper(isolated=True)
    return (check_ffmpeg() + whisper + ([] if whisper else check_whisper_model())
            + check_ollama(url, embed_model, answer_model))


def blocked_actions(problems):
    return frozenset(action for problem in problems for action in problem.blocks)


def format_problems(problems):
    return "\n".join(f"- {p.message}\n    Fix: {p.fix}" for p in problems)


def require(*, ffmpeg=False, whisper=False, embeddings=False):
    """Stop a pipeline script, with the reason, when it cannot do its work.

    Settings are validated first and on their own: a bad OLLAMA_URL would
    make the Ollama check meaningless. Exits 1, which the dashboard reports
    as that stage failing.
    """
    errors = check_config()
    problems = []
    if not errors:
        if ffmpeg:
            problems += check_ffmpeg()
        if whisper:
            problems += check_whisper()
        if embeddings:
            problems += check_ollama(
                os.environ.get("OLLAMA_URL", DEFAULTS["OLLAMA_URL"]),
                os.environ.get("EMBED_MODEL", DEFAULTS["EMBED_MODEL"]),
            )
    if not errors and not problems:
        return
    lines = ["[ERROR] Cannot start:"]
    lines += [f"- {error}" for error in errors]
    if problems:
        lines.append(format_problems(problems))
    print("\n".join(lines), file=sys.stderr, flush=True)
    sys.exit(1)


# ── query-time errors ───────────────────────────────────────────────────────

def describe_ollama_error(exc, model):
    """A sentence a student can act on, for a failed Ollama request."""
    # ConnectTimeout is both a timeout and a connection error: it means nothing listened.
    if isinstance(exc, requests.ConnectionError):
        return "Cannot connect to Ollama. Start it with: ollama serve"
    if isinstance(exc, requests.Timeout):
        return (f"Ollama did not answer in time. It may still be loading {model} or be busy: "
                "wait a moment and ask again, or raise OLLAMA_TIMEOUT.")
    if isinstance(exc, requests.HTTPError):
        status = exc.response.status_code if exc.response is not None else None
        detail = _ollama_error_text(exc.response)
        if status == 404 and "not found" in detail.lower():
            return f"The model {model} is not pulled. Pull it with: ollama pull {model}"
        return f"Ollama returned an error (HTTP {status}): {detail}" if detail else (
            f"Ollama returned an error (HTTP {status}).")
    if isinstance(exc, (ValueError, KeyError, IndexError, TypeError)):
        return f"Ollama sent a reply this program cannot read ({type(exc).__name__}: {exc})."
    return f"Something went wrong ({type(exc).__name__}: {exc})."


def _ollama_error_text(response):
    """Ollama's own explanation, which it sends as {"error": "..."}."""
    try:
        text = response.json().get("error")
    except (AttributeError, ValueError):
        return ""
    return text if isinstance(text, str) else ""


# ── command line ────────────────────────────────────────────────────────────

def _wants_pull():
    if os.environ.get("RAG_AUTO_PULL") == "1":
        return True
    if not (sys.stdin.isatty() and sys.stdout.isatty()):
        return False
    return input("Pull the missing models now? [Y/n] ").strip().lower() in ("", "y", "yes")


def _pull(models, url):
    """Pull each model with the ollama CLI; one that fails stays reported as missing."""
    env = {**os.environ, "OLLAMA_HOST": url}
    for model in models:
        subprocess.run(["ollama", "pull", model], env=env)


def main():
    """Print what is ready and what is not, offer to pull models, return the exit code."""
    errors = check_config()
    if errors:
        for error in errors:
            print(f"invalid  {error}")
        return 1

    url = os.environ.get("OLLAMA_URL", DEFAULTS["OLLAMA_URL"])
    embed_model = os.environ.get("EMBED_MODEL", DEFAULTS["EMBED_MODEL"])
    answer_model = os.environ.get("ANSWER_MODEL", DEFAULTS["ANSWER_MODEL"])
    models = list(dict.fromkeys([embed_model, answer_model]))
    have_ollama_cli = shutil.which("ollama") is not None

    whisper = check_whisper(isolated=True)
    local = check_ffmpeg() + whisper + ([] if whisper else check_whisper_model())
    remote = check_ollama(url, embed_model, answer_model)
    pullable = [problem.model for problem in remote if problem.model]
    if pullable and have_ollama_cli and _wants_pull():
        _pull(pullable, url)
        remote = check_ollama(url, embed_model, answer_model)

    problems = local + remote
    by_piece = {problem.piece: problem for problem in problems}
    for piece in (["ffmpeg", "faster-whisper", "whisper model", "ollama"]
                  + [f"model {m}" for m in models]):
        problem = by_piece.get(piece)
        if problem is None:
            print(f"ok       {'Ollama at ' + url if piece == 'ollama' else piece}")
            continue
        print(f"missing  {problem.message}\n         Fix: {problem.fix}")
        if piece == "ollama":
            if urlparse(url).hostname not in ("localhost", "127.0.0.1", "::1"):
                print(f"         {url} is not on this machine, so the dashboard will not start it")
            elif have_ollama_cli:
                print("         the dashboard will try to start it")
            else:
                print("         install it from https://ollama.com/download")
            for model in models:
                print(f"         then: ollama pull {model}")
            break
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
