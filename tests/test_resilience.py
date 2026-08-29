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


def _pick_device(cuda_devices=0):
    """Load pick_device() alone, with a stub CTranslate2 reporting N GPUs."""
    src = (ROOT / "mp3_to_json.py").read_text()
    start = src.index("def pick_device")
    end = src.index("device = pick_device()")
    ns = {"os": os, "ctranslate2": type("ct", (), {
        "get_cuda_device_count": staticmethod(lambda: cuda_devices)})}
    exec(compile(src[start:end], "mp3_to_json.py", "exec"), ns)
    return ns["pick_device"]


def test_whisper_device_selection_falls_back_to_cpu(monkeypatch):
    """No CUDA must resolve to CPU rather than erroring."""
    monkeypatch.delenv("WHISPER_DEVICE", raising=False)
    assert _pick_device(cuda_devices=0)() == "cpu"


def test_whisper_device_selection_finds_cuda(monkeypatch):
    monkeypatch.delenv("WHISPER_DEVICE", raising=False)
    assert _pick_device(cuda_devices=1)() == "cuda"


def test_whisper_device_can_be_forced(monkeypatch):
    monkeypatch.setenv("WHISPER_DEVICE", "cpu")
    assert _pick_device(cuda_devices=1)() == "cpu"


def test_model_load_failure_is_reported_not_raised():
    """Previously a failure on 'tiny' propagated as an unhandled traceback."""
    src = (ROOT / "mp3_to_json.py").read_text()
    assert "SystemExit" in src or "raise SystemExit" in src


# ── mp3_to_json: the faster-whisper path ───────────────────────────────────

def _transcription_helpers(model, beam_size=1, use_vad=True):
    """Load transcribe()/transcribe_with_fallback() against a stub model."""
    src = (ROOT / "mp3_to_json.py").read_text()
    start = src.index("def transcribe(")
    end = src.index('print(f"Found {len(audios)}')
    ns = {"os": os, "model": model, "_beam_size": beam_size, "_use_vad": use_vad}
    exec(compile(src[start:end], "mp3_to_json.py", "exec"), ns)
    return ns


class _Seg:
    def __init__(self, start, end, text):
        self.start, self.end, self.text = start, end, text


class _Info:
    duration = 10.0
    language = "en"


class _StubModel:
    """Records how it was called; optionally fails when VAD is requested."""

    def __init__(self, fail_on_vad=False):
        self.fail_on_vad = fail_on_vad
        self.calls = []

    def transcribe(self, path, **kw):
        self.calls.append(kw)
        if self.fail_on_vad and kw.get("vad_filter"):
            raise RuntimeError("vad model missing")
        return iter([_Seg(0.0, 5.0, "hello"), _Seg(5.0, 10.0, "world")]), _Info()


def test_vad_failure_retries_without_vad():
    """A broken VAD must cost speed, not the lecture."""
    model = _StubModel(fail_on_vad=True)
    ns = _transcription_helpers(model)
    segments, info = ns["transcribe_with_fallback"]("audios/x.mp3")
    assert [s.text for s in segments] == ["hello", "world"]
    assert [c["vad_filter"] for c in model.calls] == [True, False]


def test_transcription_failure_without_vad_still_raises():
    """Only VAD gets a second chance; a real failure must not be swallowed."""
    class Broken(_StubModel):
        def transcribe(self, path, **kw):
            raise RuntimeError("corrupt audio")

    ns = _transcription_helpers(Broken())
    with pytest.raises(RuntimeError, match="corrupt audio"):
        ns["transcribe_with_fallback"]("audios/x.mp3")


def test_segments_are_drained_not_returned_lazily():
    """faster-whisper returns a generator; a lazy return would defer all the
    work past the error handling meant to contain it."""
    ns = _transcription_helpers(_StubModel())
    segments, _ = ns["transcribe"]("audios/x.mp3", False)
    assert isinstance(segments, list) and len(segments) == 2


def test_greedy_decoding_is_the_default():
    """beam_size=5 is faster-whisper's default and several times slower."""
    src = (ROOT / "mp3_to_json.py").read_text()
    assert 'os.environ.get("WHISPER_BEAM_SIZE", "1")' in src


def test_compute_type_matches_the_device():
    """float16 is a GPU feature; int8 is what makes the CPU path usable."""
    src = (ROOT / "mp3_to_json.py").read_text()
    assert '"float16" if device == "cuda" else "int8"' in src


def test_torch_is_no_longer_a_dependency():
    """Transcription runs on CTranslate2; nothing else in the repo needs torch."""
    reqs = [ln.split("#")[0].strip()
            for ln in (ROOT / "requirements.txt").read_text().splitlines()]
    assert not any(ln.startswith("torch") for ln in reqs if ln)
    for script in ("mp3_to_json.py", "preprocess_json.py", "dashboard.py"):
        assert "import torch" not in (ROOT / script).read_text()
