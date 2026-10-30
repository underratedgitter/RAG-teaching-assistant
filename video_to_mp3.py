# Converts videos to mp3 - Optimized for speed
import json
import os
import re
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

os.makedirs("audios", exist_ok=True)

VIDEO_EXTENSIONS = ('.mp4', '.avi', '.mkv', '.mov')


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


def _split_audio_name(audio_name):
    """Return ``(number, label)`` for ``12_name.mp3``, else ``(None, None)``."""
    match = re.match(r"^(\d+)_(.+)\.mp3$", audio_name)
    if not match:
        return None, None
    return int(match.group(1)), match.group(2)


def _label_includes_video_extension(label):
    """Whether an audio label is a full source basename, not just its stem."""
    return os.path.splitext(label)[1].lower() in VIDEO_EXTENSIONS


def _existing_audio_index(existing_audios):
    """Index converted audio by source identity.

    New audio names include the source's full basename
    (``1_lecture.mp4.mp3``), so the label alone identifies the video. Audio
    made before that scheme is named by stem (``1_lecture.mp3``) and carries
    no extension; it can only be matched by a stem, and the caller must reject
    that match when more than one current video shares the stem.

    Returns ``(by_source, legacy_by_stem, used_numbers)``. Sidecars are
    authoritative because they pin an audio file to one exact source even
    when its filename predates the extension-bearing scheme.
    """
    by_source = {}
    legacy_by_stem = {}
    used_numbers = set()

    for audio_name in sorted(existing_audios):
        number, label = _split_audio_name(audio_name)
        if number is None:
            continue
        used_numbers.add(number)

        try:
            if os.path.getsize(os.path.join("audios", audio_name)) <= 0:
                continue
        except OSError:
            continue

        sidecar = _read_source_sidecar(os.path.join("audios", audio_name))
        source_name = sidecar.get("source") if isinstance(sidecar, dict) else None
        if isinstance(source_name, str) and source_name:
            by_source.setdefault(source_name, audio_name)
        elif _label_includes_video_extension(label):
            by_source.setdefault(label, audio_name)
        else:
            legacy_by_stem.setdefault(label, []).append(audio_name)

    return by_source, legacy_by_stem, used_numbers


def _plan_conversions(files, existing_audios):
    """Pair every video with exactly one output mp3.

    Identity is the full basename, not the stem: ``lecture.mp4`` and
    ``lecture.mov`` are different lectures and must not share audio, partial
    files, or transcripts. A legacy stem-named mp3 is reused only when one
    current video has that stem and only one legacy candidate exists; with two
    same-stem videos there is no way to know which one it came from, so both
    get fresh, distinct names instead.
    """
    converted_by_source, legacy_by_stem, used_numbers = _existing_audio_index(existing_audios)

    stem_counts = {}
    for filename in files:
        stem = os.path.splitext(filename)[0]
        stem_counts[stem] = stem_counts.get(stem, 0) + 1

    next_number = max(used_numbers, default=0) + 1
    jobs = []
    for filename in sorted(files):
        output_name = converted_by_source.get(filename)

        if output_name is None:
            stem = os.path.splitext(filename)[0]
            legacy = legacy_by_stem.get(stem, [])
            if stem_counts[stem] == 1 and len(legacy) == 1:
                output_name = legacy[0]

        if output_name is None:
            # The extension is part of the identity, so a same-stem video
            # with another container can never collide with this path.
            output_name = f"{next_number}_{filename}.mp3"
            next_number += 1

        jobs.append((output_name, filename))

    return jobs


def _ensure_source_sidecar(audio_path, source_path):
    """Record provenance the first time an older audio is matched to a video.

    Without this, a legacy stem-named mp3 stays ambiguous: adding a second
    video with the same stem would make the next run discard a perfectly good
    conversion. Writing the sidecar when the pair is first accepted pins it to
    this exact source for later runs.
    """
    if _read_source_sidecar(audio_path) is not None:
        return
    try:
        _write_source_sidecar(audio_path, source_path)
    except OSError as exc:
        print(f"   [WARNING] Could not record source signature for {audio_path}: {exc}")


def _quarantine_ambiguous_audio(files, existing_audios):
    """Preserve unidentifiable legacy caches outside the active corpus.

    Only called after all replacement conversions succeeded. Leaving a third,
    stem-only audio/transcript active would index stale lecture text alongside
    the two correctly identified new sources.
    """
    _, legacy, _ = _existing_audio_index(existing_audios)
    for stem, names in legacy.items():
        if sum(os.path.splitext(source)[0] == stem for source in files) < 2:
            continue
        for name in names:
            directory = os.path.join('audios', '.ambiguous')
            os.makedirs(directory, exist_ok=True)
            backup = tempfile.mkdtemp(prefix='legacy-', dir=directory)
            for source in (os.path.join('audios', name),
                           os.path.join('jsons', name + '.json')):
                if os.path.exists(source):
                    os.replace(source, os.path.join(backup, os.path.basename(source)))
            print(f"  Preserved ambiguous legacy cache {name} in {backup}")


existing_audios = set(os.listdir("audios"))

files = [f for f in os.listdir("videos") if f.endswith(VIDEO_EXTENSIONS)]
print(f"Found {len(files)} video files")


def convert_video(args):
    """Convert a single video file, replacing stale audio when needed."""
    output_name, file = args

    final_path = os.path.join("audios", output_name)
    source_path = os.path.join("videos", file)
    if _audio_is_current(final_path, source_path):
        _ensure_source_sidecar(final_path, source_path)
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
             "-acodec", "libmp3lame", "-q:a", "6", "-threads", "0", "-f", "mp3",
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
    jobs = _plan_conversions(files, existing_audios)
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

_quarantine_ambiguous_audio(files, existing_audios)
