"""Tests for merge_segments — the chunking that retrieval quality depends on."""
import sys, types
from pathlib import Path

# Import merge_segments without running the module's top-level pipeline,
# which reads jsons/ and calls Ollama.
_src = (Path(__file__).parent.parent / "preprocess_json.py").read_text()
_fn = _src[_src.index("def merge_segments"): _src.index('print("="*50)')]
_mod = types.ModuleType("_merge")
exec(compile(_fn, "preprocess_json.py", "exec"), _mod.__dict__)
merge_segments = _mod.merge_segments


def seg(text, start, end, number="1", title="L1"):
    return {"number": number, "title": title, "start": start, "end": end, "text": text}


def build(n=200, words_per=8, seconds_per=3.0):
    return [
        seg(" ".join(f"w{i}_{j}" for j in range(words_per)), i * seconds_per, (i + 1) * seconds_per)
        for i in range(n)
    ]


def test_empty_input_returns_empty():
    assert merge_segments([]) == []


def test_single_segment_survives():
    out = merge_segments([seg("hello world", 0.0, 1.0)])
    assert len(out) == 1
    assert out[0]["text"] == "hello world"
    assert out[0]["start"] == 0.0 and out[0]["end"] == 1.0


def test_chunks_reach_the_target_size():
    out = merge_segments(build(), target_words=150, overlap_words=30)
    # every chunk but the last should be at or past the target
    for c in out[:-1]:
        assert len(c["text"].split()) >= 150


def test_start_covers_the_first_words_of_the_chunk():
    """The timestamp must point at the words the chunk actually opens with.

    Regression: `start` used to be reset to the segment that triggered the
    flush, ignoring the overlap carried over from the previous chunk — so
    every overlapped chunk cited a time later than the words it quoted.
    """
    segs = build()
    by_first_word = {s["text"].split()[0]: s["start"] for s in segs}

    for chunk in merge_segments(segs):
        first_word = chunk["text"].split()[0]
        assert chunk["start"] == by_first_word[first_word], (
            f"chunk starts at {chunk['start']} but its first word "
            f"{first_word!r} occurs at {by_first_word[first_word]}"
        )


def test_end_covers_the_last_words():
    segs = build()
    by_last_word = {s["text"].split()[-1]: s["end"] for s in segs}
    for chunk in merge_segments(segs):
        assert chunk["end"] == by_last_word[chunk["text"].split()[-1]]


def test_time_ranges_are_ordered():
    for c in merge_segments(build()):
        assert c["start"] <= c["end"]


def test_consecutive_chunks_overlap_in_time():
    out = merge_segments(build(), target_words=150, overlap_words=30)
    assert len(out) > 2
    for a, b in zip(out, out[1:]):
        # overlap means the next chunk reaches back before the previous ended
        assert b["start"] < a["end"]


def test_no_content_is_lost():
    segs = build(n=60)
    joined = " ".join(c["text"] for c in merge_segments(segs))
    for s in segs:
        assert s["text"] in joined


def test_metadata_is_carried():
    segs = [seg("a b c", 0, 1, number="7", title="Lecture 7")]
    out = merge_segments(segs)
    assert out[0]["number"] == "7"
    assert out[0]["title"] == "Lecture 7"


def test_scales_linearly_not_quadratically():
    """Guards the running word count. The old version re-split the whole
    buffer on every segment, so 4x the input cost far more than 4x."""
    import time

    def timed(n):
        segs = build(n=n)
        t0 = time.perf_counter()
        merge_segments(segs)
        return time.perf_counter() - t0

    timed(500)  # warm
    small, large = timed(500), timed(4000)
    # 8x the input should be well under 30x the time if it is linear
    assert large < small * 30, f"{small:.4f}s -> {large:.4f}s looks superlinear"
