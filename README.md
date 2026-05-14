# RAG Teaching Assistant

> **Drop lecture videos in. Ask questions. Get answers with exact timestamps.**

A fully local desktop app that turns hours of lecture recordings into an AI-powered Q&A system — no cloud, no API keys, no subscriptions. Just your GPU doing the work.

---

## The Problem

You recorded 10 hours of lectures. Now you need to find where the professor explained "backpropagation." Good luck scrubbing through video timelines.

## The Solution

This app watches your lectures so you don't have to (again). Upload videos, let it transcribe and index everything, then ask questions in plain English. It'll point you to the exact video and timestamp.

```
You:  "How does gradient descent work?"
App:  "In Video 2 at 14:30, the instructor explains that gradient descent
       iteratively adjusts parameters by computing partial derivatives..."
```

## Manual vs AI-Assisted Lecture Review

| Feature | Manual Method | AI-Assisted |
|---------|---------------|-------------|
| **Time** | 30-60 min to find info | <1 second query |
| **Effort** | Tedious scrubbing | Instant answers |
| **Understanding**| Passive note-taking | Deep context |
| **Format** | Short notes/bookmarks | Full context + timestamp |
| **Search** | By timeline only | Semantic/meaning-based |
| **Context** | Missing relationships | All related concepts |

> **Result:** 50-100x Faster | Better Learning Retention | Searchable Knowledge Base

---

## How It Works

```text
 Upload Videos ──> Extract Audio ──> Transcribe ──> Build Index ──> Ask Away!
     MP4             MP3            Chunks+Time     Embeddings      AI Answers
```

### Complete Pipeline
1. **Upload Videos** (MP4, AVI, MOV) via Tkinter
2. **Extract Audio** (MP3) via FFmpeg
3. **Transcribe** (JSON) via OpenAI Whisper
4. **Create Chunks** (Chunks) via Pandas
5. **Generate Embeddings** (Vectors) via Ollama
6. **Answer Questions** (LLM Query) via Ollama LLM
7. **Return Results** (Answer+Time) on Display

---

## Key Features

- **100% Local** — runs entirely on your machine, no data leaves your PC
- **GPU Accelerated** — CUDA-powered Whisper transcription with fp16 for speed
- **Smart Chunking** — overlapping text segments for better context retrieval
- **Timestamp References** — every answer points back to video + time
- **Fast Retrieval** — precomputed embeddings + cosine similarity in milliseconds
- **Model Pre-warming** — first query is fast, no cold start penalty

---

## Tech Stack

| Component | Tool | File |
|-----------|------|------|
| Desktop UI | Tkinter | `dashboard.py` |
| Video → Audio | FFmpeg (parallel) | `video_to_mp3.py` |
| Speech → Text | OpenAI Whisper (`small`) | `mp3_to_json.py` |
| Text → Vectors | Ollama + `nomic-embed-text` | `preprocess_json.py` |
| Question → Answer | Ollama + `qwen2.5:1.5b` | `dashboard.py` |
| Similarity Search | scikit-learn cosine similarity | `dashboard.py` |
| Data Storage | pandas, numpy, joblib | `preprocess_json.py` |

## Project Structure

```
dashboard.py            # The brain — UI, orchestration, Q&A
video_to_mp3.py         # Rips audio from videos (parallel)
mp3_to_json.py          # Whisper transcription → timestamped chunks
preprocess_json.py      # Chunk merging + embedding generation
requirements.txt        # Python dependencies
```

Generated at runtime:
```
videos/                 # Your uploaded videos
audios/                 # Extracted MP3 files
jsons/                  # Transcript JSONs
embeddings.joblib       # Searchable embedding database
embedding_matrix.npy    # Precomputed similarity matrix
```

## System Architecture

- **User Interface Layer:** Tkinter Dashboard
- **Processing Layer & Models:**
  - Video Processing: FFmpeg
  - Audio Processing: Whisper
  - Text Processing: Nomic Embeddings
  - Embedding Generation & Query: Qwen
- **Data Storage & Retrieval:** Videos Folder, Audios Folder, JSON Transcripts, Embeddings Database, Similarity Matrix, Cache
- **Output:** Answer with Video Timestamp & Relevance Score

## Data Flow & Storage (1 hour lecture)

| Stage | Format | Estimated Size | Reduction |
|-------|--------|----------------|-----------|
| 1. Video Files | MP4, AVI, MOV | ~1GB (500MB-2GB) | - |
| 2. Audio Extract | MP3 (Optional) | ~100MB (50-200MB) | ~10:1 |
| 3. Transcription | JSON Chunks | ~10MB (5-20MB) | ~100:1 |
| 4. Embeddings | Vectors | ~5MB (2-10MB) | ~50:1 |
| 5. Similarity Matrix | Precomputed | ~500KB-5MB | ~5:1 |
| 6. Query Result | Cached Text | ~1-5KB | ~1000:1 |

*Note: Embeddings are highly compressed vector representations. Original transcripts retain full text information for context.*

---

## Pipeline Deep Dive

### 1) App startup (`dashboard.py`)

Clears previous working data, loads any cached embeddings, fires up the GUI, and pre-warms both Ollama models in the background so your first query is instant.

### 2) Upload videos

Drag in your `.mp4`, `.avi`, `.mkv`, or `.mov` files. They're copied to `videos/`.

### 3) Processing (one click, three stages)

#### Stage 1: Video → Audio (`video_to_mp3.py`)
FFmpeg strips the audio track. Up to 4 videos convert in parallel.

#### Stage 2: Audio → Text (`mp3_to_json.py`)
Whisper `small` model transcribes with GPU acceleration (`fp16` on CUDA).
Each segment gets a timestamp. `condition_on_previous_text` keeps transcriptions coherent across long lectures.

#### Stage 3: Text → Embeddings (`preprocess_json.py`)
Small Whisper segments are merged into ~150-word overlapping chunks (30-word overlap) so retrieval captures full ideas, not sentence fragments. Each chunk is embedded via `nomic-embed-text` in batches of 128.

### 4) Ask a question

1. Your question is embedded with `nomic-embed-text`
2. Cosine similarity finds the **top 8** most relevant chunks
3. Low-relevance results are filtered out (threshold > 0.3)
4. A structured prompt with readable timestamps is sent to `qwen2.5:1.5b`
5. You get a grounded answer with exact video + timestamp references

---

## Ollama Models

| Model | Role | Why |
|-------|------|-----|
| `nomic-embed-text` | Embedding | Converts text to vectors for semantic search |
| `qwen2.5:1.5b` | Generation | Small, fast LLM that fits in ~2GB VRAM |

---

## Performance

> Designed to run well on a laptop GPU (tested on RTX 3050 4GB).

- **Parallel conversion** — FFmpeg processes up to 4 videos simultaneously
- **fp16 inference** — halves Whisper's GPU memory and doubles speed
- **Smart chunking** — overlapping 150-word chunks beat tiny sentence fragments for retrieval
- **Batch embeddings** — 128 chunks per API call instead of one-by-one
- **Precomputed matrix** — similarity search runs in milliseconds, not seconds
- **Model pre-warming** — both models loaded on startup, zero cold-start lag
- **Connection pooling** — HTTP sessions reused across all Ollama calls
- **Threaded UI** — processing never freezes the interface

---

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Make sure FFmpeg is in PATH
ffmpeg -version

# 3. Start Ollama
ollama serve

# 4. Pull models (first time only)
ollama pull nomic-embed-text
ollama pull qwen2.5:1.5b

# 5. Launch
python dashboard.py
```

---

## Usage

1. **Upload Videos** — click the button, select your lecture files
2. **Process Videos** — one click kicks off the full pipeline
3. **Ask Questions** — type naturally, hit Enter or click Ask
4. **Read the Answer** — complete with video number and timestamp

---

## Requirements

- Python 3.10+
- FFmpeg installed and in PATH
- [Ollama](https://ollama.com/) running locally
- NVIDIA GPU recommended (works on CPU, just slower)

---

## Troubleshooting

| Problem | Fix |
|---------|-----|
| Can't connect to Ollama | Run `ollama serve` first |
| No embeddings / no answers | Re-process videos from dashboard |
| FFmpeg not found | Install FFmpeg and add to PATH |
| First query is slow | Wait for "Models warmed up" in terminal |

---

## License

MIT

- **Slow transcription**
	- Use GPU-enabled PyTorch + CUDA where available
	- Keep fewer heavy processes running in parallel

## Future Improvements

- Smarter chunking strategies for better retrieval granularity
- Evaluation metrics for retrieval and answer quality
- Optional citation rendering from exact chunk timestamps
- Incremental indexing without full cache reset
