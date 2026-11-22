# RAG Teaching Assistant

Drop lecture videos in. Ask a question. Get an answer grounded in what was actually said, with the timestamp where it was said.

Runs entirely on your machine — Whisper for transcription, Ollama for embeddings and generation. No API key, no upload, nothing leaves the laptop. Two things are downloaded once and then cached: the Whisper weights (from the Hugging Face Hub, on the first transcription) and the Ollama models (`ollama pull`). Until both are on disk the first run needs an internet connection; after that it works offline.

Transcription is the slow stage, so it runs on [faster-whisper](https://github.com/SYSTRAN/faster-whisper) (CTranslate2) rather than the reference PyTorch implementation: same weights, roughly **18–27× realtime on a plain CPU**, and no 2 GB torch install to sit through.

---

## Quickstart

From a fresh clone, one command:

```bash
./scripts/start.sh        # or: make start    (macOS: double-click start.command)
```

It creates `.venv` (with [uv](https://docs.astral.sh/uv/) if you have it, otherwise `python3 -m venv`), installs `requirements.txt`, checks that ffmpeg, faster-whisper, Ollama and both models are in place (the [startup checks](#startup-checks)), and opens the dashboard. When something is missing it prints the exact command that fixes it (`brew install ffmpeg`, `ollama pull qwen2.5:1.5b`) and, on an interactive terminal, offers to pull the missing models. It never pulls without a yes or `RAG_AUTO_PULL=1`. A missing piece does not stop the dashboard opening: you can still ask questions of an already-processed library without ffmpeg.

Running it again is quick: dependencies are reinstalled only when `requirements.txt` changes, so a second launch is the preflight and the window, about a second. `make check` (or `.venv/bin/python preflight.py`) runs the checks alone.

The launcher needs bash, so on Windows follow the manual steps under [Requirements](#requirements). It needs Python 3.11–3.14 with Tk. If the dashboard will not import, install your Python's Tk package (`brew install python-tk`, `sudo apt install python3-tk`). Intel Macs have no `onnxruntime` wheel for Python 3.14, so use 3.11–3.13 there; with uv installed the launcher fetches a suitable Python itself.

For CI or a headless smoke test, `RAG_NO_GUI=1 ./scripts/start.sh` does the same setup and checks, confirms `dashboard.py` imports, and exits without opening the window.

---

## How it works

Four stages, each a script you can run on its own.

```
videos/  ──video_to_mp3──>  audios/  ──mp3_to_json──>  jsons/  ──preprocess_json──>  chunks.json + embedding_matrix.npy (+ embedding_matrix.meta.json)
                                                                                            │
                                                                                      dashboard.py
```

**1. `video_to_mp3.py`** — extracts audio with ffmpeg. Runs conversions in a thread pool, and skips anything already converted, so re-running after adding one video costs one video.

Audio names include the full source basename (`1_lecture.mp4.mp3`), so an MP4
and MOV with the same stem remain separate lectures. Unambiguous old audio names
are reused. After successful replacement conversions, ambiguous legacy audio and
transcripts are preserved in `audios/.ambiguous/` rather than indexed twice.

A video you delete from `videos/` leaves the library the next time you run **Process Videos**: its audio, source record and transcript are moved to `audios/.orphaned/` (kept, not deleted), so it stops being transcribed and indexed, and `preprocess_json.py` drops the cached embeddings of any transcript that is gone.

**2. `mp3_to_json.py`** — transcribes with faster-whisper, keeping each segment's `start` and `end` timestamps. Picks CUDA when it's available and falls back to CPU, and picks its precision to match: `float16` on a GPU, `int8` on a CPU. Voice-activity detection drops silence before it reaches the model, so the pauses in a lecture cost nothing. A recording with no speech at all is written as an empty transcript marked `no_speech`, counted as done and skipped on later runs rather than re-transcribed and reported as a failure. Decoding is greedy rather than beam search — on lecture speech the transcript is the same and the decode is several times quicker.

The Whisper weights are looked up in the Hugging Face cache first and fetched only if they are missing, once per model rather than once per precision. With no network and nothing cached, the stage stops with a message saying the first transcription needs an internet connection (or a `WHISPER_MODEL` folder that already holds a model). `python preflight.py` and the dashboard's startup checks report the same thing before you start, and only when the weights are neither cached nor downloadable.

If the `small` model won't load it degrades to `base`, then `tiny`, and each size falls back through lower precisions first, rather than failing outright — a machine with less VRAM gets a worse transcript instead of no transcript.

**3. `preprocess_json.py`** — Whisper's segments are too short to retrieve against on their own, so `merge_segments()` combines them into ~150-word chunks with a 30-word overlap. Overlap matters: it stops an answer being cut in half at a chunk boundary. A lecture's last few sentences (under a third of a chunk of new words) are folded into the chunk before instead of becoming a near-duplicate of it. Chunks are embedded with `nomic-embed-text` in batches of 128, several batches in flight at once, with a retrying HTTP session and a per-item retry when a whole batch is rejected. Connection failures are retried, but a request that times out (`OLLAMA_TIMEOUT`) is not sent a second time, and once Ollama is unreachable the stage stops and lists the files it did not embed instead of waiting out every one.

Embeddings are cached per file under `.embed_cache/`, keyed by a hash of that file's chunk text and the model name. Adding one lecture to a library of twenty re-embeds one lecture, not twenty. Editing a transcript or switching embedding model invalidates only what changed.

Each script also prints `PROGRESS done/total` on a line of its own after every file it handles (converted, skipped or failed), which is what the dashboard's status bar reads. File names of any kind (spaces, quotes, leading dashes, non-ASCII) go through all four stages, and a console that cannot display a name prints it with escapes instead of aborting the stage.

Each script first checks the tools it depends on and exits `1` with a fix if one is missing (see [Startup checks](#startup-checks)). Then it exits `0` when every input succeeded, `2` when some inputs failed but the rest were processed, and `1` when inputs were given but none could be processed (`preprocess_json.py` also exits `1` when there are no transcripts at all). The dashboard carries on past a `2`, so one unreadable lecture does not block the others, and says which stage had failures.

The dashboard also checks model identity before loading saved embeddings. After
changing `EMBED_MODEL`, run **Process Videos** again to regenerate embeddings
from the transcripts.

Nothing on disk is pickled. Chunks are stored as `chunks.json`, vectors as
`embedding_matrix.npy` with its identity in `embedding_matrix.meta.json`, and
the per-file cache as `.npz`, all read without pickle, so a tampered file
cannot run code when it is loaded. The dashboard loads the corpus only when
all three agree on corpus, model and row count. An `embeddings.joblib` and
`.embed_cache/*.joblib` files left by an older version are ignored and can be
deleted; the first **Process Videos** run after upgrading re-embeds the whole
library in the new format (audio and transcripts are reused).

`nomic-embed-text` expects task prefixes, so chunks are embedded as
`search_document: …` and questions as `search_query: …`. Other embedding
models get the raw text.

**4. `dashboard.py`** — a Tkinter interface. Your question is embedded, compared against every chunk by cosine similarity, and the top 8 above a 0.3 threshold are passed to `qwen2.5:1.5b` as context. If nothing clears the threshold it falls back to the best 3, so you get a hedged answer rather than silence. Each answer ends with a "Sources:" line naming the lecture and M:SS–M:SS range of the best 3 chunks it was given, built from the retrieval results rather than from the model's text. The terminal shows the similarity scores of the chunks each answer used, which is what to look at when tuning `SIMILARITY_THRESHOLD`. Instructions go in the system prompt, and the excerpts and the question are wrapped in per-request tags and treated as data, so text inside a lecture cannot pose as an instruction.

**Processing from the dashboard.** **Process Videos** runs the three scripts in turn and then loads the index, so the status bar reads `Step N/4: … done/total files`. **Cancel** (enabled only while a run is going) stops the running stage and everything it started, such as the ffmpeg workers, with `SIGTERM` and a kill after 5 seconds, then deletes the `.partial` temp files they left and re-enables the buttons. Because every output is written to a temp file and renamed, a cancelled or killed stage never leaves a truncated file behind; finished work is kept and the next run resumes from it.

The corpus matrix is normalised once, in place, at load (float32, no second copy), so a question is a single matrix–vector product and allocates nothing the size of the corpus. A synthetic 50k-chunk, 768-dimension library (154 MB) loads in about a second and answers in roughly 10 ms.

The model is instructed to answer **only** from the supplied excerpts, which is what keeps it honest about material it hasn't seen.

---

## Requirements

- Python 3.11–3.14
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

To set up by hand instead of using the [Quickstart](#quickstart) launcher:

```bash
pip install -r requirements.txt

ollama pull nomic-embed-text     # embeddings
ollama pull qwen2.5:1.5b         # answering
```

### Settings

Everything is overridable by environment variable, and every value is validated
before anything runs (see [Startup checks](#startup-checks)):

| Variable | Default | Valid values | Does |
|---|---|---|---|
| `WHISPER_MODEL` | `small` on CUDA, else `base` | a Whisper size (`tiny`, `base`, `small`, `medium`, `large-v3`, `turbo`, `distil-*`, a `.en` variant), a Hugging Face repo id or a local model directory | transcription accuracy vs speed |
| `WHISPER_DEVICE` | auto | `cpu`, `cuda` | force the device |
| `WHISPER_COMPUTE_TYPE` | `float16` on CUDA, else `int8` | `int8`, `int8_float16`, `int8_float32`, `int16`, `float16`, `bfloat16`, `float32` | weight precision |
| `WHISPER_BEAM_SIZE` | `1` | whole number, 1-20 | raise to 5 for hard-to-hear audio |
| `WHISPER_VAD` | `1` | `0`, `1` | `0` transcribes silence too |
| `WHISPER_CPU_THREADS` | all cores | whole number, 0-512 (`0` = all cores) | encoder threads on a CPU run |
| `OLLAMA_URL` | `http://localhost:11434` | an `http://` or `https://` address | where Ollama listens |
| `OLLAMA_TIMEOUT` | `120` | whole number of seconds, 1-3600 | how long the dashboard and `preprocess_json.py` wait for Ollama to embed or write an answer; raise it on a slow machine |
| `EMBED_MODEL` | `nomic-embed-text` | an Ollama model name | embedding model |
| `EMBED_CONCURRENCY` | `3` | whole number, 1-32 | embedding batches in flight at once |
| `ANSWER_MODEL` | `qwen2.5:1.5b` | an Ollama model name | answering model |
| `NUM_PREDICT` | `300` | whole number, 16-4096 | most tokens the model may write for one answer; an answer that reaches it ends with `[answer truncated]` |
| `SIMILARITY_THRESHOLD` | `0.3` | number, -1 to 1 | cosine similarity below which a chunk is treated as unrelated |
| `TOP_K` | `8` | whole number, 1-100 | chunks passed to the model as context |
| `FALLBACK_K` | `3` | whole number, 1-100 | chunks used when nothing clears the threshold |
| `RAG_NO_GUI` | unset | `0`, `1` | `1` makes `scripts/start.sh` set up and check everything, then exit without opening the window |
| `RAG_AUTO_PULL` | unset | `0`, `1` | `1` lets `preflight.py` (and so the launcher) pull missing Ollama models without asking |
| `CLEAR_ON_START` | unset | `0`, `1` | `1` wipes the derived work (audio, transcripts, embeddings) on launch, for a clean demo; `videos/` is kept |

### Startup checks

The dashboard checks its setup at launch, and each pipeline script checks what
it needs before it starts. Nothing is guessed: a missing piece is reported with
the command that fixes it.

| Check | Fix shown | Disables |
|---|---|---|
| `ffmpeg` on `PATH` | `brew install ffmpeg`, `winget install ffmpeg` or `sudo apt install ffmpeg` | Process Videos |
| `faster-whisper` imports | `python -m pip install -r requirements.txt` | Process Videos |
| Whisper weights are cached, or the Hugging Face Hub is reachable to fetch them | connect once, or set `WHISPER_MODEL` to a downloaded model folder | Process Videos |
| Ollama answers at `OLLAMA_URL` | `ollama serve` | Process Videos and Ask |
| `EMBED_MODEL` is pulled | `ollama pull nomic-embed-text` (or your model) | Process Videos and Ask |
| `ANSWER_MODEL` is pulled | `ollama pull qwen2.5:1.5b` (or your model) | Ask |

Problems appear in red under the buttons and in the terminal pane (and on the
launching terminal), and only the buttons that need the missing piece are
disabled. Fix the problem, then click **Re-check Setup**; no restart is needed.
A script run on its own prints the same message to stderr and exits `1`.

`python preflight.py` runs the same checks from the command line, which is what the launcher and `make check` call. It needs only the Python standard library, so it works before requirements are installed. It prints one line per piece, the fix for each gap, and exits `0` when everything is ready and `1` otherwise. When a model is missing and the `ollama` CLI is there, it offers to pull it (a yes at the prompt on a terminal, or `RAG_AUTO_PULL=1`); it never pulls unasked.

An invalid setting is reported by name and the reason (for example
`TOP_K='many' must be a whole number`) before anything starts; the dashboard
shows it in a dialog and exits `1`.

If Ollama fails while you are asking a question (not running, too slow, or the
model is missing), the answer area shows a one-line explanation and the fix,
and the window stays usable: ask again once it is fixed.

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

The obvious build here is Whisper API plus a hosted embedding model, and it would be less code. This runs offline instead because lecture recordings are often not yours to upload, an always-on API key is a running cost for something used in bursts, and once the Whisper weights and Ollama models have been downloaded the whole pipeline still works on a train.

The trade is quality: `qwen2.5:1.5b` is a small model, and answers are correspondingly plainer than a frontier model would give. Retrieval quality is doing most of the work, which is where the chunking and overlap earn their place.

---

## Layout

```
video_to_mp3.py       ffmpeg extraction, thread-pooled, resumable, atomic
mp3_to_json.py        faster-whisper transcription with timestamps, model ladder
preprocess_json.py    segment merging, batched embeddings, retry logic
dashboard.py          Tkinter UI, cancel and progress, dot-product retrieval, prompt assembly
preflight.py          startup checks, settings validation, readable Ollama errors, the `python preflight.py` check
scripts/start.sh      one-command launcher: .venv, requirements, preflight, dashboard
start.command         macOS double-click wrapper for scripts/start.sh
Makefile              make start / make check
tests/                pipeline, dashboard, audio and crash-safety tests
tests/_pipeline.py    shared loader compiling script definitions in isolation
requirements.txt      runtime dependencies
requirements-dev.txt  adds the pinned test and lint tools CI uses
project_blueprint.txt older design notes (predates the pickle-free storage)
```

## Tests

```bash
pip install -r requirements-dev.txt && pytest tests/ -q
```

Tests covering audio conversion, transcription loop and device fallbacks, PyAV
compatibility, chunk sizing, timestamp correctness, overlap, content
preservation, linear scaling, embedding-cache invalidation, parallel batch
ordering, the crash-safety of every file the pipeline writes (including a stage
killed midway), corpus and model identity on load, prompt construction, the
dashboard's threading, cancelling a run, progress reporting, a 50k-chunk
memory bound, and all four stages on awkward file names. They need no models
and no network: the end-to-end tests use stand-in ffmpeg, Whisper and Ollama
on 127.0.0.1.
