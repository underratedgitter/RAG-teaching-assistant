# Converts videos to mp3 - Optimized for speed
import os
import re
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

os.makedirs("audios", exist_ok=True)
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
    """Convert a single video file"""
    i, file = args
    name = os.path.splitext(file)[0]
    output_name = f"{i}_{name}.mp3"

    final_path = os.path.join("audios", output_name)
    if name in converted_by_name:
        return True, f"Skipping {file} (exists as {converted_by_name[name]})"

    # Encode to a temp file and rename only on success. Writing straight to
    # the final name meant an interrupted ffmpeg left a truncated mp3 tha
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
            os.replace(tmp_path, final_path)   # atomic on the same filesystem
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
        if os.path.splitext(f)[0] in converted_by_name:
            jobs.append((0, f))            # skipped inside convert_video
        else:
            jobs.append((next_number, f))
            next_number += 1
    futures = [executor.submit(convert_video, job) for job in jobs]
    for future in as_completed(futures):
        ok, message = future.result()
        print(f"  {message}")
        if not ok:
            failures.append(message)

elapsed = time.time() - star
print(f"Done! ({elapsed:.1f}s)")

if failures:
    # Exit non-zero so a caller driving the pipeline can tell that some
    # lectures are missing rather than assuming a clean run.
    print(f"\n[WARNING] {len(failures)} file(s) failed to convert:")
    for message in failures:
        print(f"    {message}")
    sys.exit(1)
