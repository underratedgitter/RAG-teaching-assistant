import json
import os
import time

import ctranslate2
from faster_whisper import WhisperModel

# Setup for performance and reliability
os.makedirs("jsons", exist_ok=True)


def pick_device():
    """CUDA where present, otherwise CPU.

    Transcription runs on CTranslate2 now, which reports its own GPUs — so
    this no longer needs torch loaded just to answer one question. Metal is
    not a CTranslate2 backend, so an Apple machine lands on CPU like any
    other host without CUDA; WHISPER_DEVICE still forces the choice.
    """
    forced = os.environ.get("WHISPER_DEVICE")
    if forced:
        return forced
    try:
        if ctranslate2.get_cuda_device_count() > 0:
            return "cuda"
    except Exception:
        pass
    return "cpu"


device = pick_device()
print(f"Using: {device.upper()}")

# Weight precision is the second-biggest lever after model size. float16
# halves the work a GPU does per layer; int8 does the same for a CPU, where
# integer kernels are what the vector units are actually good at. Neither is
# a meaningful accuracy loss for speech. Override with WHISPER_COMPUTE_TYPE.
_default_compute = "float16" if device == "cuda" else "int8"
_compute = os.environ.get("WHISPER_COMPUTE_TYPE", _default_compute)

# Model size is the biggest lever on transcription time, and the right
# default depends on the hardware. Override with WHISPER_MODEL=small|base|tiny.
_default_model = "small" if device == "cuda" else "base"
_preferred = os.environ.get("WHISPER_MODEL", _default_model)

_model_ladder = [_preferred] + [m for m in ("small", "base", "tiny") if m != _preferred]

# Precision fallbacks, tried per model size. A GPU too old for float16, or one
# short on VRAM, still gets a transcript rather than an error.
if device == "cuda":
    _compute_ladder = [_compute] + [c for c in ("float16", "int8_float16", "float32") if c != _compute]
else:
    _compute_ladder = [_compute] + [c for c in ("int8", "float32") if c != _compute]

# Whisper decoding is sequential, so extra threads only help the encoder — but
# that is where a CPU run spends most of its time.
_cpu_threads = int(os.environ.get("WHISPER_CPU_THREADS", "0")) or (os.cpu_count() or 4)

model = None
for _name in _model_ladder:
    for _ct in _compute_ladder:
        try:
            model = WhisperModel(
                _name,
                device=device,
                compute_type=_ct,
                cpu_threads=_cpu_threads if device == "cpu" else 0,
            )
            print(f"Loaded '{_name}' model ({_ct})")
            break
        except Exception as e:
            print(f"[WARNING] Failed to load '{_name}' model as {_ct}: {e}")
    if model is not None:
        break

if model is None:
    # Previously a failure on 'tiny' propagated as an unhandled exception.
    raise SystemExit(
        "Could not load any Whisper model. Check the install and free disk space."
    )

# Greedy decoding. Beam search explores five candidate transcripts per window
# and picks the best; on lecture speech it changes almost nothing and costs
# several times the decode time. Raise with WHISPER_BEAM_SIZE if a recording
# is genuinely hard to hear.
_beam_size = int(os.environ.get("WHISPER_BEAM_SIZE", "1"))

# Voice-activity detection drops silence before it reaches the model. Lecture
# recordings are full of pauses, and every skipped second is a second the
# model never has to decode. Set WHISPER_VAD=0 to disable.
_use_vad = os.environ.get("WHISPER_VAD", "1") != "0"

audios = [f for f in os.listdir("audios") if f.endswith('.mp3')]


def already_transcribed(json_filename):
    """True only if the existing JSON is complete and parseable.

    Checking the filename alone meant a transcript truncated by a crash or a
    Ctrl+C was skipped on every later run — permanently, and silently, since
    the file looked done. Anything unreadable is deleted so it gets redone.
    """
    path = os.path.join("jsons", json_filename)
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        return False
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        return bool(data.get("chunks"))
    except (json.JSONDecodeError, OSError, UnicodeDecodeError):
        print(f"  [!] {json_filename} is corrupt — re-transcribing")
        try:
            os.remove(path)
        except OSError:
            pass
        return False


def transcribe(path, use_vad):
    """Run one file and drain the segment generator.

    faster-whisper returns segments lazily, so the work happens here, not in
    the transcribe() call — which is also why progress can be reported at all.
    """
    segments, info = model.transcribe(
        path,
        language="en",
        task="transcribe",
        beam_size=_beam_size,
        vad_filter=use_vad,
        vad_parameters={"min_silence_duration_ms": 500} if use_vad else None,
        condition_on_previous_text=True,
    )
    collected = []
    total = getattr(info, "duration", 0) or 0
    next_mark = 0.1
    for seg in segments:
        collected.append(seg)
        if total:
            done = seg.end / total
            if done >= next_mark:
                print(f"      {min(done, 1.0) * 100:.0f}%", flush=True)
                next_mark = done + 0.1
    return collected, info


def transcribe_with_fallback(path, use_vad=None):
    """Transcribe, dropping VAD if VAD is what broke.

    VAD is a second model in front of the real one, with its own weights to
    load. When it fails the recording is still perfectly transcribable — just
    without the silence-skipping — so a failure here costs speed, not the
    lecture.
    """
    use_vad = _use_vad if use_vad is None else use_vad
    try:
        return transcribe(path, use_vad)
    except Exception as e:
        if not use_vad:
            raise
        print(f"  [WARNING] VAD failed ({e}) — retrying without it", flush=True)
        return transcribe(path, False)


print(f"Found {len(audios)} audio files", flush=True)

start_total = time.time()
results = {"success": 0, "skipped": 0, "failed": 0}
audio_seconds = 0.0

for idx, audio in enumerate(audios, 1):
    json_filename = f"{audio}.json"
    if already_transcribed(json_filename):
        print(f"[{idx}/{len(audios)}] Skipping {audio} (exists)", flush=True)
        results["skipped"] += 1
        continue

    # Extract number and title from filename: "1_VideoName.mp3"
    name = os.path.splitext(audio)[0]  # Remove .mp3
    if "_" in name:
        parts = name.split("_", 1)
        number = parts[0]
        title = parts[1] if len(parts) > 1 else name
    else:
        number = "1"
        title = name

    print(f"[{idx}/{len(audios)}] Transcribing: {audio}", flush=True)
    start = time.time()

    try:
        # Validate file exists and is readable
        audio_path = f"audios/{audio}"
        if not os.path.exists(audio_path) or os.path.getsize(audio_path) == 0:
            print(f"  [!] Invalid audio file (empty or missing)")
            results["failed"] += 1
            continue

        segments, info = transcribe_with_fallback(audio_path)

        chunks = []
        text_parts = []
        for seg in segments:
            try:
                chunk = {
                    "number": number,
                    "title": title,
                    "start": float(seg.start),
                    "end": float(seg.end),
                    "text": str(seg.text).strip()
                }
                if chunk["text"]:  # Only add non-empty chunks
                    chunks.append(chunk)
                    text_parts.append(chunk["text"])
            except (ValueError, AttributeError, TypeError) as e:
                print(f"  [WARNING] Skipped malformed segment: {e}")
                continue

        if not chunks:
            print(f"  [!] No valid chunks extracted")
            results["failed"] += 1
            continue

        # Save with error handling
        try:
            # Write to a temp file and rename, so an interrupted save cannot
            # leave a half-written transcript that later runs treat as done.
            tmp_path = f"jsons/.{json_filename}.partial"
            with open(tmp_path, "w", encoding="utf-8") as f:
                json.dump({
                    "chunks": chunks,
                    "text": " ".join(text_parts),
                    "language": getattr(info, "language", "en") or "en"
                }, f, ensure_ascii=False)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp_path, f"jsons/{json_filename}")

            elapsed = time.time() - start
            spoken = getattr(info, "duration", 0) or 0
            audio_seconds += spoken
            speed = f", {spoken / elapsed:.0f}x realtime" if elapsed > 0 and spoken else ""
            print(f"  [+] Created {len(chunks)} chunks ({elapsed:.1f}s{speed})", flush=True)
            results["success"] += 1
        except Exception as e:
            print(f"  [!] Failed to save JSON: {e}")
            try:
                if os.path.exists(f"jsons/.{json_filename}.partial"):
                    os.remove(f"jsons/.{json_filename}.partial")
            except OSError:
                pass
            results["failed"] += 1

        # Deliberately nothing freed between files. CTranslate2 keeps the
        # model resident and reuses its own buffers, so the next lecture
        # starts decoding immediately instead of re-allocating.

    except Exception as e:
        print(f"  [!] Transcription failed: {e}", flush=True)
        results["failed"] += 1
        # Continue with next file instead of crashing.

elapsed_total = time.time() - start_total
overall = f" — {audio_seconds / elapsed_total:.0f}x realtime" if elapsed_total > 0 and audio_seconds else ""
print(f"\nDone! ({elapsed_total:.1f}s total{overall})")
print(f"  Success: {results['success']}, Skipped: {results['skipped']}, Failed: {results['failed']}")

if results["success"] > 0 or results["skipped"] > 0:
    print("[STATUS] Ready for next step")
