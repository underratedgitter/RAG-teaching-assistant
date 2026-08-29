# RAG Teaching Assistant

Drop lecture videos in. Ask a question. Get an answer grounded in what was actually said, with the timestamp where it was said.

Runs entirely on your machine — Whisper for transcription, Ollama for embeddings and generation. No API key, no upload, nothing leaves the laptop.

---

## How it works

Four stages, each a script you can run on its own.

```
videos/  ──video_to_mp3──>  audios/  ──mp3_to_json──>  jsons/  ──preprocess_json──>  embeddings.joblib
                                                                                            │
                                                                                      dashboard.py
```

**1. `video_to_mp3.py`** — extracts audio with ffmpeg. Runs conversions in a thread pool, and skips anything already converted, so re-running after adding one video costs one video.

**2. `mp3_to_json.py`** — transcribes with Whisper, keeping each segment's `start` and `end` timestamps. Picks CUDA when it's available and falls back to CPU. If the `small` model won't load it degrades to `base`, then `tiny`, rather than failing outright — a machine with less VRAM gets a worse transcript instead of no transcript.

**3. `preprocess_json.py`** — Whisper's segments are too short to retrieve against on their own, so `merge_segments()` combines them into ~150-word chunks with a 30-word overlap. Overlap matters: it stops an answer being cut in half at a chunk boundary. Chunks are embedded with `nomic-embed-text` in batches of 128, with a retrying HTTP session and a per-item fallback so one bad chunk doesn't lose the batch.

**4. `dashboard.py`** — a Tkinter interface. Your question is embedded, compared against every chunk by cosine similarity, and the top 8 above a 0.3 threshold are passed to `qwen2.5:1.5b` as context. If nothing clears the threshold it falls back to the best 3, so you get a hedged answer rather than silence.

The model is instructed to answer **only** from the supplied excerpts, which is what keeps it honest about material it hasn't seen.

---

## Requirements

- Python 3.9+
- [ffmpeg](https://ffmpeg.org/) on `PATH`
- [Ollama](https://ollama.com/) running locally
- A CUDA GPU is optional — it is the difference between a five-minute transcription and a twenty-second one

```bash
pip install -r requirements.txt

ollama pull nomic-embed-text     # embeddings
ollama pull qwen2.5:1.5b         # answering
```

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

Every stage is resumable. Each skips work it has already done, so adding a lecture later does not re-process the library.

---

## Why local

The obvious build here is Whisper API plus a hosted embedding model, and it would be less code. This runs offline instead because lecture recordings are often not yours to upload, an always-on API key is a running cost for something used in bursts, and the whole pipeline still works on a train.

The trade is quality: `qwen2.5:1.5b` is a small model, and answers are correspondingly plainer than a frontier model would give. Retrieval quality is doing most of the work, which is where the chunking and overlap earn their place.

---

## Layout

```
video_to_mp3.py       ffmpeg extraction, thread-pooled, resumable
mp3_to_json.py        Whisper transcription with timestamps, model fallback
preprocess_json.py    segment merging, batched embeddings, retry logic
dashboard.py          Tkinter UI, cosine retrieval, prompt assembly
requirements.txt
project_blueprint.txt design notes
```
