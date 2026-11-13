"""Tests for audio conversion, naming, and transcription pipeline stages."""
import json
import os
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import pytest

from _pipeline import WHISPER_CONFIG, load

ROOT = Path(__file__).parent.parent


def _video_helpers(tmp_path, monkeypatch):
    """Load video_to_mp3's helper functions without running the pipeline."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / "audios").mkdir(exist_ok=True)
    (tmp_path / "videos").mkdir(exist_ok=True)
    return load(
        "video_to_mp3.py",
        functions="all",
        constants={"PARTIAL_FAILURE", "VIDEO_EXTENSIONS"},
        namespace={
            "os": os, "json": json, "re": re, "tempfile": tempfile,
            "subprocess": subprocess, "sys": sys, "time": time,
        },
    )


def _extract(path, name):
    """Pull one function out of a pipeline script without running the script."""
    return load(path, functions=[name], namespace={"os": os, "json": json})[name]


def test_zero_byte_audio_keeps_same_output_name(tmp_path, monkeypatch):
    """Fix 3: zero-byte mp3 retains mapping so _plan_conversions reuses the name."""
    h = _video_helpers(tmp_path, monkeypatch)
    (tmp_path / "videos" / "lecture.mp4").write_bytes(b"video data")
    empty_audio = tmp_path / "audios" / "1_lecture.mp4.mp3"
    empty_audio.write_bytes(b"")

    jobs = h["_plan_conversions"](["lecture.mp4"], {"1_lecture.mp4.mp3"})
    assert jobs == [("1_lecture.mp4.mp3", "lecture.mp4")]


def test_non_dict_json_transcript_treated_as_corrupt(tmp_path, monkeypatch):
    """Fix 5: a [] transcript returns False, is removed, and does not raise."""
    monkeypatch.chdir(tmp_path)
    jsons_dir = tmp_path / "jsons"
    jsons_dir.mkdir(exist_ok=True)
    bad_json = jsons_dir / "lecture.mp3.json"
    bad_json.write_text("[]")

    already = _extract("mp3_to_json.py", "already_transcribed")
    assert already("lecture.mp3.json") is False
    assert not bad_json.exists()


def test_split_audio_name():
    """Fix 7: title metadata derives cleanly and strips video extensions."""
    split = _extract("mp3_to_json.py", "split_audio_name")
    assert split("1_lecture.mp4.mp3") == ("1", "lecture")
    assert split("3_My_Talk.MOV.mp3") == ("3", "My_Talk")
    assert split("intro.mp3") == ("1", "intro")
    assert split("2_notes.mp3") == ("2", "notes")


def test_the_pinned_av_still_takes_faster_whisper_arguments(tmp_path):
    """The av<19 cap is load-bearing, and a comment alone does not hold it.

    faster-whisper 1.2.1 opens audio with av.open(..., metadata_errors=...),
    which PyAV 19 removed. If the cap were ever lifted, every transcription
    would fail at runtime — so check that the installed pair still fits.
    """
    try:
        import av
        import faster_whisper
    except ImportError as missing:
        if os.environ.get("CI"):
            pytest.fail(f"the av/faster-whisper pair is not installed: {missing}")
        pytest.skip(f"the av/faster-whisper pair is not installed: {missing}")
    import wave

    audio = Path(faster_whisper.__file__).parent / "audio.py"
    # What faster-whisper actually asks of av, so the check below is exactly
    # the call it will make.
    kwargs = {}
    if "metadata_errors" in audio.read_text(encoding="utf-8"):
        kwargs["metadata_errors"] = "ignore"

    path = tmp_path / "silence.wav"
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(16000)
        handle.writeframes(b"\0" * 3200)

    with av.open(str(path), mode="r", **kwargs) as container:
        assert container.duration is not None


def test_cuda_exhausted_falls_back_to_cpu(monkeypatch):
    """Fix 10: CUDA failure retries on CPU and returns the CPU model."""
    monkeypatch.setenv("WHISPER_DEVICE", "cuda")

    class StubCudaFailsCpuSucceeds:
        def __init__(self, name, device, compute_type, cpu_threads=0):
            if device == "cuda":
                raise RuntimeError("CUDA out of memory")
            self.device = device
            self.name = name
            self.compute_type = compute_type
            self.cpu_threads = cpu_threads

    ns = load(
        "mp3_to_json.py",
        functions=["pick_device", "load_model"],
        constants=WHISPER_CONFIG,
        namespace={
            "os": os,
            "json": json,
            "ctranslate2": type("ct", (), {
                "get_cuda_device_count": staticmethod(lambda: 1)}),
            "WhisperModel": StubCudaFailsCpuSucceeds,
        },
    )
    model = ns["load_model"]()
    assert model.device == "cpu"


def test_cuda_success_makes_no_cpu_attempt(monkeypatch):
    """Fix 10: Successful CUDA load never attempts CPU."""
    monkeypatch.setenv("WHISPER_DEVICE", "cuda")

    class StubCudaSucceeds:
        devices_attempted = []

        def __init__(self, name, device, compute_type, cpu_threads=0):
            StubCudaSucceeds.devices_attempted.append(device)
            self.device = device
            self.name = name
            self.compute_type = compute_type
            self.cpu_threads = cpu_threads

    ns = load(
        "mp3_to_json.py",
        functions=["pick_device", "load_model"],
        constants=WHISPER_CONFIG,
        namespace={
            "os": os,
            "json": json,
            "ctranslate2": type("ct", (), {
                "get_cuda_device_count": staticmethod(lambda: 1)}),
            "WhisperModel": StubCudaSucceeds,
        },
    )
    model = ns["load_model"]()
    assert model.device == "cuda"
    assert "cpu" not in StubCudaSucceeds.devices_attempted
