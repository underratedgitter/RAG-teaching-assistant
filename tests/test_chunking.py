"""Tests for merge_segments — the chunking that retrieval quality depends on."""
from _pipeline import load

# Import merge_segments without running the module's top-level pipeline,
# which reads jsons/ and calls Ollama.
merge_segments = load("preprocess_json.py", functions=["merge_segments"])["merge_segments"]


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
        t0 = time.process_time()
        merge_segments(segs)
        return time.process_time() - t0

    timed(500)  # warm
    # CPU time, not wall-clock: being descheduled makes a run look slow on
    # the clock even though it did no more work, and that is what made this
    # test fail under load on CI. The best of three is the run that was
    # disturbed least.
    small = min(timed(500) for _ in range(3))
    large = min(timed(4000) for _ in range(3))
    # 8x the input should be well under 30x the time if it is linear
    assert large < small * 30, f"{small:.4f}s -> {large:.4f}s looks superlinear"


def test_oversized_segment_is_not_repeated():
    segs = []
    for i in range(3):
        text = " ".join(f"pre_{i}_{w}" for w in range(8))
        segs.append(seg(text, float(i), float(i + 1)))

    big_text = " ".join(f"big_{w}" for w in range(160))
    segs.append(seg(big_text, 3.0, 23.0))

    for i in range(10):
        text = " ".join(f"post_{i}_{w}" for w in range(8))
        segs.append(seg(text, 23.0 + i, 24.0 + i))

    merged = merge_segments(segs, target_words=150, overlap_words=30)

    # (a) Oversized segment appears in at most 2 chunks; every segment in at least one chunk
    big_count = sum(1 for c in merged if big_text in c["text"])
    assert big_count <= 2
    for s in segs:
        assert any(s["text"] in c["text"] for c in merged)

    # (c) No merged chunk is empty and chunk start times are non-decreasing
    assert all(len(c["text"].strip()) > 0 for c in merged)
    start_times = [c["start"] for c in merged]
    assert all(t1 <= t2 for t1, t2 in zip(start_times, start_times[1:]))


def test_overlap_zero_produces_no_overlap():
    # (b) overlap_words=0 produces no overlap: with existing build() helper,
    # each original segment text appears in exactly one merged chunk.
    segs = build()
    merged = merge_segments(segs, target_words=150, overlap_words=0)
    for s in segs:
        count = sum(1 for c in merged if s["text"] in c["text"])
        assert count == 1

