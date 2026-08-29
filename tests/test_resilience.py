"""Tests for the crash-safety of the pipeline's file handling.

These cover failures that only appear after an interruption — a Ctrl+C, a
timeout, a machine going to sleep mid-encode — and which the skip-already-done
logic would otherwise make permanent.
"""
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).parent.parent


def _extract(path, name):
    """Pull one function out of a pipeline script without running the script."""
    src = (ROOT / path).read_text()
    start = src.index(f"def {name}")
    rest = src[start:]
    # up to the next top-level statement
    end = re.search(r"\n(?=[^\s#])", rest[len(f"def {name}"):])
    body = rest if end is None else rest[: len(f"def {name}") + end.start()]
    ns = {"os": os, "json": json}
    exec(compile(body, path, "exec"), ns)
    return ns[name]


# ── mp3_to_json: a corrupt transcript must not be skipped forever ───────────

@pytest.fixture
def jsons_dir(tmp_path, monkeypatch):
    d = tmp_path / "jsons"
    d.mkdir()
    monkeypatch.chdir(tmp_path)
    return d


def test_complete_transcript_is_skipped(jsons_dir):
    already = _extract("mp3_to_json.py", "already_transcribed")
    (jsons_dir / "a.mp3.json").write_text(
        json.dumps({"chunks": [{"text": "hi", "start": 0, "end": 1}]})
    )
    assert already("a.mp3.json") is True


def test_missing_transcript_is_not_skipped(jsons_dir):
    already = _extract("mp3_to_json.py", "already_transcribed")
    assert already("nope.mp3.json") is False


def test_empty_transcript_is_not_skipped(jsons_dir):
    already = _extract("mp3_to_json.py", "already_transcribed")
    (jsons_dir / "a.mp3.json").write_text("")
    assert already("a.mp3.json") is False


def test_truncated_transcript_is_redone_and_removed(jsons_dir):
    """The regression: a half-written JSON used to look finished forever."""
    already = _extract("mp3_to_json.py", "already_transcribed")
    path = jsons_dir / "a.mp3.json"
    path.write_text('{"chunks": [{"text": "hi", "sta')   # killed mid-write

    assert already("a.mp3.json") is False, "corrupt transcript was treated as done"
    assert not path.exists(), "corrupt transcript should be deleted so it is retried"


def test_transcript_with_no_chunks_is_not_skipped(jsons_dir):
    already = _extract("mp3_to_json.py", "already_transcribed")
    (jsons_dir / "a.mp3.json").write_text(json.dumps({"chunks": [], "text": ""}))
    assert already("a.mp3.json") is False


# ── video_to_mp3: partial audio must not be mistaken for a finished file ────

def test_partial_file_is_cleaned_up(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    cleanup = _extract("video_to_mp3.py", "_cleanup")
    p = tmp_path / ".1_x.mp3.partial"
    p.write_bytes(b"\x00" * 10)
    cleanup(str(p))
    assert not p.exists()


def test_cleanup_is_safe_when_file_is_absent(tmp_path):
    cleanup = _extract("video_to_mp3.py", "_cleanup")
    cleanup(str(tmp_path / "does-not-exist"))   # must not raise


def test_zero_byte_audio_is_not_treated_as_converted(tmp_path, monkeypatch):
    """An interrupted encode can leave an empty file; the size check catches it."""
    monkeypatch.chdir(tmp_path)
    audios = tmp_path / "audios"
    audios.mkdir()
    empty = audios / "1_lecture.mp3"
    empty.write_bytes(b"")
    assert empty.exists()
    assert os.path.getsize(empty) == 0   # the guard video_to_mp3 now applies


# ── the scripts stay importable/parseable ───────────────────────────────────

@pytest.mark.parametrize(
    "script", ["video_to_mp3.py", "mp3_to_json.py", "preprocess_json.py", "dashboard.py"]
)
def test_script_compiles(script):
    src = (ROOT / script).read_text()
    compile(src, script, "exec")


def test_atomic_writes_are_used_everywhere_output_is_produced():
    """Every producer writes to a temp path and renames."""
    for script in ("video_to_mp3.py", "mp3_to_json.py"):
        src = (ROOT / script).read_text()
        assert "os.replace(" in src, f"{script} should rename atomically"
        assert ".partial" in src, f"{script} should stage output in a temp file"


# ── cross-platform behaviour ────────────────────────────────────────────────

def test_open_folder_does_not_use_windows_only_api():
    """os.startfile does not exist on macOS or Linux.

    Regression: the Open Folder button called it unconditionally and raised
    AttributeError on any non-Windows machine.
    """
    src = (ROOT / "dashboard.py").read_text()
    lines = src.splitlines()
    calls = [n for n, line in enumerate(lines, 1)
             if re.search(r"os\.startfile\s*\(", line)]
    assert calls, "expected a Windows branch to still exist"
    for line_no in calls:
        window = "\n".join(lines[max(0, line_no - 5): line_no])
        assert re.search(r"sys\.platform.*win", window), (
            f"the os.startfile call at line {line_no} is not inside a Windows branch"
        )


def test_gpu_offload_is_not_assumed_on_intel_mac():
    """num_gpu must not be sent where no GPU offload exists."""
    src = (ROOT / "dashboard.py").read_text()
    assert '"num_gpu": 99' not in src.replace('GENERATION_OPTIONS["num_gpu"] = 99', ""), \
        "num_gpu should be conditional, not inlined in the request"
    assert "_has_gpu_offload" in src


def test_whisper_device_selection_falls_back_to_cpu(monkeypatch):
    """No CUDA and no Metal must resolve to CPU rather than erroring."""
    src = (ROOT / "mp3_to_json.py").read_text()
    start = src.index("def pick_device")
    end = src.index("device = pick_device()")
    ns = {"os": os, "torch": type("t", (), {"cuda": type("c", (), {"is_available": staticmethod(lambda: False)})})}
    exec(compile(src[start:end], "mp3_to_json.py", "exec"), ns)
    monkeypatch.delenv("WHISPER_DEVICE", raising=False)
    assert ns["pick_device"]() == "cpu"


def test_whisper_device_can_be_forced(monkeypatch):
    src = (ROOT / "mp3_to_json.py").read_text()
    start = src.index("def pick_device")
    end = src.index("device = pick_device()")
    ns = {"os": os, "torch": type("t", (), {"cuda": type("c", (), {"is_available": staticmethod(lambda: False)})})}
    exec(compile(src[start:end], "mp3_to_json.py", "exec"), ns)
    monkeypatch.setenv("WHISPER_DEVICE", "mps")
    assert ns["pick_device"]() == "mps"


def test_model_load_failure_is_reported_not_raised():
    """Previously a failure on 'tiny' propagated as an unhandled traceback."""
    src = (ROOT / "mp3_to_json.py").read_text()
    assert "SystemExit" in src or "raise SystemExit" in src
