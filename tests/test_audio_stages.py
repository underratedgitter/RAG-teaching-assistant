"""Tests for audio conversion, naming, and transcription pipeline stages."""
import json
import os
import re
from pathlib import Path

ROOT = Path(__file__).parent.parent


def _video_helpers(tmp_path, monkeypatch):
    """Load video_to_mp3's helper functions without running the pipeline."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / "audios").mkdir(exist_ok=True)
    (tmp_path / "videos").mkdir(exist_ok=True)
    src = (ROOT / "video_to_mp3.py").read_text()
    head = src[: src.index("existing_audios =")]
    ns = {"__name__": "video_to_mp3_helpers"}
    exec(compile(head, "video_to_mp3.py", "exec"), ns)
    return ns


def _extract(path, name):
    """Pull one function out of a pipeline script without running the script."""
    src = (ROOT / path).read_text()
    start = src.index(f"def {name}")
    rest = src[start:]
    end = re.search(r"\n(?=[^\s#])", rest[len(f"def {name}"):])
    body = rest if end is None else rest[: len(f"def {name}") + end.start()]
    ns = {"os": os, "json": json}
    exec(compile(body, path, "exec"), ns)
    return ns[name]


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


def test_cuda_exhausted_falls_back_to_cpu(monkeypatch):
    """Fix 10: CUDA failure retries on CPU and returns the CPU model."""
    monkeypatch.setenv("WHISPER_DEVICE", "cuda")
    src = (ROOT / "mp3_to_json.py").read_text()
    start = src.index("def pick_device")
    end = src.index("def split_audio_name")

    class StubCudaFailsCpuSucceeds:
        def __init__(self, name, device, compute_type, cpu_threads=0):
            if device == "cuda":
                raise RuntimeError("CUDA out of memory")
            self.device = device
            self.name = name
            self.compute_type = compute_type
            self.cpu_threads = cpu_threads

    ns = {
        "os": os,
        "json": json,
        "ctranslate2": type("ct", (), {"get_cuda_device_count": staticmethod(lambda: 1)}),
        "WhisperModel": StubCudaFailsCpuSucceeds,
    }
    exec(compile(src[start:end], "mp3_to_json.py", "exec"), ns)
    model = ns["load_model"]()
    assert model.device == "cpu"


def test_cuda_success_makes_no_cpu_attempt(monkeypatch):
    """Fix 10: Successful CUDA load never attempts CPU."""
    monkeypatch.setenv("WHISPER_DEVICE", "cuda")
    src = (ROOT / "mp3_to_json.py").read_text()
    start = src.index("def pick_device")
    end = src.index("def split_audio_name")

    class StubCudaSucceeds:
        devices_attempted = []

        def __init__(self, name, device, compute_type, cpu_threads=0):
            StubCudaSucceeds.devices_attempted.append(device)
            self.device = device
            self.name = name
            self.compute_type = compute_type
            self.cpu_threads = cpu_threads

    ns = {
        "os": os,
        "json": json,
        "ctranslate2": type("ct", (), {"get_cuda_device_count": staticmethod(lambda: 1)}),
        "WhisperModel": StubCudaSucceeds,
    }
    exec(compile(src[start:end], "mp3_to_json.py", "exec"), ns)
    model = ns["load_model"]()
    assert model.device == "cuda"
    assert "cpu" not in StubCudaSucceeds.devices_attempted
