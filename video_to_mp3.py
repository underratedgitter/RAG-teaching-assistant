# Converts videos to mp3 - Optimized for speed
import os 
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

os.makedirs("audios", exist_ok=True)
existing_audios = set(os.listdir("audios"))

files = [f for f in os.listdir("videos") if f.endswith(('.mp4', '.avi', '.mkv', '.mov'))]
print(f"Found {len(files)} video files")

def convert_video(args):
    """Convert a single video file"""
    i, file = args
    name = os.path.splitext(file)[0]
    output_name = f"{i}_{name}.mp3"
    
    final_path = os.path.join("audios", output_name)
    if output_name in existing_audios and os.path.getsize(final_path) > 0:
        return True, f"Skipping {file} (exists)"

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
    futures = [executor.submit(convert_video, (i, f)) for i, f in enumerate(sorted(files), 1)]
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
