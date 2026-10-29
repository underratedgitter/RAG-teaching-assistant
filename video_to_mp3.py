# Converts videos to mp3 - Optimized for speed
import json
import os
import re
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

os.makedirs("audios", exist_ok=True)


def _source_signature(source_path):
    """Identity of the video a converted audio came from."""
    stat = os.stat(source_path)
    return {
        "source": os.path.basename(source_path),
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }


def _sidecar_path(audio_path):
    return f"{audio_path}.source.json"


def _read_source_sidecar(audio_path):
    try:
        with open(_sidecar_path(audio_path), encoding="utf-8") as handle:
            value = json.load(handle)
        return value if isinstance(value, dict) else None
    except (OSError, ValueError, UnicodeDecodeError):
        return None


def _write_source_sidecar(audio_path, source_path):
    """Record the source signature atomically after a successful encode."""
    target = _sidecar_path(audio_path)
    tmp = f"{target}.partial"
    try:
        with open(tmp, "w", encoding="utf-8") as handle:
            json.dump(_source_signature(source_path), handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, target)
    finally:
        try:
            if os.path.exists(tmp):
                os.remove(tmp)
        except OSError:
            pass


def _audio_is_current(audio_path, source_path):
    """Whether the existing mp3 was produced from the current video file.

    New conversions carry a sidecar with the source size and mtime. Audio made
    before sidecars existed falls back to an mtime comparison, so replacing a
    video with a newer file triggers a re-encode instead of reusing old audio.
    """
    try:
        if not os.path.exists(audio_path) or os.path.getsize(audio_path) == 0:
            return False
        sidecar = _read_source_sidecar(audio_path)
        if sidecar is not None:
            return sidecar == _source_signature(source_path)
        return os.path.getmtime(audio_path) >= os.path.getmtime(source_path)
    except OSError:
        return False


def _remove_transcript(audio_name):
    """Drop the old transcript when its audio is replaced."""
    for path in (
        os.path.join("jsons", f"{audio_name}.json"),
        os.path.join("jsons", f".{audio_name}.json.partial"),
    ):
        try:
            os.remove(path)
        except OSError:
            pass


existing_audios = set(os.listdir("audios"))

# Audio names carry a position prefix ("3_Lecture.mp3") taken from the sorted
# video list. Uploading a video that sorts earlier shifts every later index, so
# matching on the full name re-encoded already-converted lectures under new
# numbers and indexed them twice. Match on the video name instead.
converted_by_name = {}
for existing in existing_audios:
    m = re.match(r"^\d+_(.+)\.mp3$", existing)
    if m and os.path.getsize(os.path.join("audios", existing)) > 0:
        converted_by_name.setdefault(m.group(1), existing)

# New videos take numbers after the highest one already used, so a late upload
# never reuses a lecture number ("Video 1") that an earlier lecture already has.
used_numbers = [int(n.split("_", 1)[0]) for n in converted_by_name.values()]
next_number = max(used_numbers, default=0) + 1

files = [f for f in os.listdir("videos") if f.endswith(('.mp4', '.avi', '.mkv', '.mov'))]
print(f"Found {len(files)} video files")

def convert_video(args):
    """Convert a single video file, replacing stale audio when needed."""
    output_name, file = args
    name = os.path.splitext(file)[0]

    final_path = os.path.join("audios", output_name)
    source_path = os.path.join("videos", file)
    if _audio_is_current(final_path, source_path):
        return True, f"Skipping {file} (up to date as {output_name})"

    # Encode to a temp file and rename only on success. Writing straight to
    # the final name meant an interrupted ffmpeg left a truncated mp3 that
    # the check above then skipped forever — Whisper would transcribe half a
    # lecture and nothing would ever say so.
    tmp_path = os.path.join("audios", f".{output_name}.partial")

    try:
        result = subprocess.run(
            ["ffmpeg", "-y", "-i", f"videos/{file}",
             "-vn",
             # Whisper works at 16 kHz mono and resamples anything else, so
             # producing that directly is cheaper to encode, smaller on disk,
             # and faster to load than 165 kbps stereo.
             "-ar", "16000", "-ac", "1",
             "-acodec", "libmp3lame", "-q:a", "6", "-threads", "0",
             tmp_path],
            capture_output=True,
            timeout=3600
        )

        if result.returncode == 0 and os.path.getsize(tmp_path) > 0:
            if os.path.exists(final_path):
                # The video changed; the transcript from the old audio is stale.
                _remove_transcript(output_name)
            os.replace(tmp_path, final_path)   # atomic on the same filesystem
            try:
                _write_source_sidecar(final_path, source_path)
            except OSError as exc:
                print(f"   [WARNING] Could not record source signature for {output_name}: {exc}")
            return True, f"Created {output_name}"

        _cleanup(tmp_path)
        err = (result.stderr or b"").decode("utf-8", "replace").strip().splitlines()
        detail = err[-1] if err else f"exit {result.returncode}"
        return False, f"Error converting {file}: {detail}"
    except subprocess.TimeoutExpired:
        _cleanup(tmp_path)
        return False, f"Timeout converting {file}"
    except Exception as e:
        _cleanup(tmp_path)
        return False, f"Error converting {file}: {e}"


def _cleanup(path):
    """Remove a partial file so the next run retries instead of skipping it."""
    try:
        if os.path.exists(path):
            os.remove(path)
    except OSError:
        pass

start = time.time()
max_workers = min(4, os.cpu_count() or 2)

failures = []
with ThreadPoolExecutor(max_workers=max_workers) as executor:
    jobs = []
    for f in sorted(files):
        name = os.path.splitext(f)[0]
        existing = converted_by_name.get(name)
        if existing:
            # Reuse the existing lecture number; convert_video decides whether
            # the audio is current or needs to be replaced.
            jobs.append((existing, f))
        else:
            output_name = f"{next_number}_{name}.mp3"
            jobs.append((output_name, f))
            next_number += 1
    futures = [executor.submit(convert_video, job) for job in jobs]
    for future in as_completed(futures):
        ok, message = future.result()
        print(f"  {message}")
        if not ok:
            failures.append(message)

elapsed = time.time() - start
print(f"Done! ({elapsed:.1f}s)")

if failures:
    # Exit non-zero so a caller driving the pipeline can tell that some
    # lectures are missing rather than assuming a clean run.
    print(f"\n[WARNING] {len(failures)} file(s) failed to convert:")
    for message in failures:
        print(f"    {message}")
    sys.exit(1)
