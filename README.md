# RAG Teaching Assistant

Drop lecture videos in. Ask a question. Get an answer grounded in what was actually said, with the timestamp where it was said.

Runs entirely on your machine — Whisper for transcription, Ollama for embeddings and generation. No API key, no upload, nothing leaves the laptop.

Transcription is the slow stage, so it runs on [faster-whisper](https://github.com/SYSTRAN/faster-whisper) (CTranslate2) rather than the reference PyTorch implementation: same weights, roughly **18–27× realtime on a plain CPU**, and no 2 GB torch install to sit through.

---

## How it works

Four stages, each a script you can run on its own.

```
videos/  ──video_to_mp3──>  audios/  ──mp3_to_json──>  jsons/  ──preprocess_json──>  embeddings.joblib
                                                                                            │
                                                                                      dashboard.py
```

**1. `video_to_mp3.py`** — extracts audio with ffmpeg. Runs conversions in a thread pool, and skips anything already converted, so re-running after adding one video costs one video.

**2. `mp3_to_json.py`** — transcribes with faster-whisper, keeping each segment's `start` and `end` timestamps. Picks CUDA when it's available and falls back to CPU, and picks its precision to match: `float16` on a GPU, `int8` on a CPU. Voice-activity detection drops silence before it reaches the model, so the pauses in a lecture cost nothing. Decoding is greedy rather than beam search — on lecture speech the transcript is the same and the decode is several times quicker.

If the `small` model won't load it degrades to `base`, then `tiny`, and each size falls back through lower precisions first, rather than failing outright — a machine with less VRAM gets a worse transcript instead of no transcript.

**3. `preprocess_json.py`** — Whisper's segments are too short to retrieve against on their own, so `merge_segments()` combines them into ~150-word chunks with a 30-word overlap. Overlap matters: it stops an answer being cut in half at a chunk boundary. Chunks are embedded with `nomic-embed-text` in batches of 128, several batches in flight at once, with a retrying HTTP session and a per-item fallback so one bad chunk doesn't lose the batch.

Embeddings are cached per file under `.embed_cache/`, keyed by a hash of that file's chunk text and the model name. Adding one lecture to a library of twenty re-embeds one lecture, not twenty. Editing a transcript or switching embedding model invalidates only what changed.

**4. `dashboard.py`** — a Tkinter interface. Your question is embedded, compared against every chunk by cosine similarity, and the top 8 above a 0.3 threshold are passed to `qwen2.5:1.5b` as context. If nothing clears the threshold it falls back to the best 3, so you get a hedged answer rather than silence.

The model is instructed to answer **only** from the supplied excerpts, which is what keeps it honest about material it hasn't seen.

---

## Requirements

- Python 3.9–3.13 (PyAV, which faster-whisper decodes audio with, has no 3.14 wheel yet)
- [ffmpeg](https://ffmpeg.org/) on `PATH`
- [Ollama](https://ollama.com/) running locally

Runs on Windows, macOS and Linux. Hardware decides the speed, not the code:

| Machine | Whisper runs on | Precision | Default model |
|---|---|---|---|
| NVIDIA GPU | CUDA | `float16` | `small` |
| Apple / no GPU | CPU | `int8` | `base` |

CTranslate2 has no Metal backend, so an Apple machine transcribes on the CPU —
which is now fast enough that it stopped being the problem it was. A measured
run of the `base` model on an Intel Mac, no GPU at all: **46 seconds of speech
in 1.7 s, 27× realtime**. An hour of lecture lands in about two minutes.

A CUDA build needs NVIDIA's cuBLAS and cuDNN libraries on `PATH`; if they are
missing, the loader falls back through `int8_float16` and `float32` and, in
the worst case, the CPU. Override anything with `WHISPER_MODEL=small` if you
would rather wait for the accuracy.

```bash
pip install -r requirements.txt

ollama pull nomic-embed-text     # embeddings
ollama pull qwen2.5:1.5b         # answering
```

### Settings

Everything is overridable by environment variable:

| Variable | Default | Does |
|---|---|---|
| `WHISPER_MODEL` | `small` on CUDA, else `base` | transcription accuracy vs speed |
| `WHISPER_DEVICE` | auto | force `cuda` or `cpu` |
| `WHISPER_COMPUTE_TYPE` | `float16` on CUDA, else `int8` | weight precision |
| `WHISPER_BEAM_SIZE` | `1` | raise to 5 for hard-to-hear audio |
| `WHISPER_VAD` | `1` | `0` transcribes silence too |
| `WHISPER_CPU_THREADS` | all cores | encoder threads on a CPU run |
| `OLLAMA_URL` | `http://localhost:11434` | where Ollama listens |
| `EMBED_MODEL` | `nomic-embed-text` | embedding model |
| `EMBED_CONCURRENCY` | `3` | embedding batches in flight at once |
| `ANSWER_MODEL` | `qwen2.5:1.5b` | answering model |
| `CLEAR_ON_START` | unset | `1` wipes all work on launch, for a clean demo |

---

## Running it

```bash
mkdir -p videos audios jsons
# put lecture recordings in videos/   (.mp4 .avi .mkv .mov)

python video_to_mp3.py       # extract audio
python mp3_to_json.py        # transcribe, with timestamps
python preprocess_json.py    # chunk and embed
python dashboard.py          # ask questions
```

Every stage is resumable. Each skips work it has already done, so adding a
lecture later does not re-process the library — and output is written to a
temporary file and renamed into place, so an interrupted run leaves no
half-finished transcript that later runs would mistake for a complete one.

---

## Why local

The obvious build here is Whisper API plus a hosted embedding model, and it would be less code. This runs offline instead because lecture recordings are often not yours to upload, an always-on API key is a running cost for something used in bursts, and the whole pipeline still works on a train.

The trade is quality: `qwen2.5:1.5b` is a small model, and answers are correspondingly plainer than a frontier model would give. Retrieval quality is doing most of the work, which is where the chunking and overlap earn their place.

---

## Layout

```
video_to_mp3.py       ffmpeg extraction, thread-pooled, resumable, atomic
mp3_to_json.py        faster-whisper transcription with timestamps, model ladder
preprocess_json.py    segment merging, batched embeddings, retry logic
dashboard.py          Tkinter UI, dot-product retrieval, prompt assembly
tests/                chunking and crash-safety tests
requirements.txt
project_blueprint.txt design notes
```

## Tests

```bash
pip install pytest && pytest tests/ -q
```

Forty-one tests covering chunk sizing, timestamp correctness, overlap,
content preservation, linear scaling, embedding-cache invalidation, parallel
batch ordering, and the crash-safety of every file the pipeline writes. They need no models and no network.
