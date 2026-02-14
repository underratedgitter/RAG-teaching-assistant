# RAG Teaching Assistant

A local-first Retrieval-Augmented Generation (RAG) desktop app that turns long lecture videos into a searchable Q&A system with timestamp-aware context.

## Why this project

Long educational videos are useful, but finding one specific concept later is slow. This project solves that by converting lecture content into semantically searchable chunks and generating answers grounded in retrieved transcript context.

## Key Features

- Upload multiple videos from the desktop UI
- Convert videos to MP3 in parallel using FFmpeg
- Transcribe audio to timestamped chunks with Whisper
- Build semantic embeddings locally via Ollama
- Retrieve top relevant chunks by cosine similarity
- Generate concise answers with video/timestamp context
- Run fully local (no external hosted LLM required)

## Architecture Overview

```text
Videos -> Audio Extraction -> Transcription -> Embedding Index -> Retrieval -> Answer Generation
 MP4        MP3                JSON chunks      Vector cache      Top-K        Final response
```

## Tech Stack

- **UI**: Tkinter (`dashboard.py`)
- **Video -> Audio**: FFmpeg (`video_to_mp3.py`)
- **Speech-to-Text**: OpenAI Whisper + PyTorch (`mp3_to_json.py`)
- **Embeddings + LLM**: Ollama (`preprocess_json.py`, `dashboard.py`)
- **Embedding model**: `nomic-embed-text`
- **Generation model**: `qwen2.5:1.5b`
- **Retrieval math**: scikit-learn cosine similarity
- **Data & persistence**: pandas, numpy, joblib, JSON

## Project Structure

```text
.
├── dashboard.py               # Tkinter app + orchestration + query answering
├── video_to_mp3.py           # Batch/parallel video->audio conversion
├── mp3_to_json.py            # Whisper transcription to chunked JSON
├── preprocess_json.py        # Embedding generation + cache creation
├── requirements.txt
├── embeddings.joblib         # Generated embedding dataframe cache
├── embedding_matrix.npy      # Generated float32 matrix for fast similarity
├── videos/                   # Input videos
├── audios/                   # Intermediate MP3 files
└── jsons/                    # Transcript JSON outputs
```

## End-to-End Pipeline (Detailed)

### 1) App startup and workspace prep (`dashboard.py`)

- Creates/ensures `videos/`, `audios/`, and `jsons/`
- Clears previous working files and embedding cache on launch
- Loads existing embedding artifacts if present
- Starts desktop UI for upload, processing, and Q&A

### 2) Video upload (`dashboard.py`)

- User selects video files through a file dialog
- Files are copied into `videos/`
- UI status updates with current video count and index status

### 3) Processing pipeline (`dashboard.py` triggers 3 scripts)

#### Step 3.1: Video -> MP3 (`video_to_mp3.py`)

- Scans `videos/` for supported formats (`.mp4`, `.avi`, `.mkv`, `.mov`)
- Converts each file with FFmpeg:
	- `-vn` removes video stream
	- `-acodec libmp3lame` uses MP3 encoder
	- `-q:a 4` quality setting
	- `-threads 0` lets FFmpeg use all cores
- Uses `ThreadPoolExecutor` (up to 4 workers) for parallel conversion
- Stores outputs in `audios/` as `1_filename.mp3`, `2_filename.mp3`, etc.

#### Step 3.2: MP3 -> transcript chunks (`mp3_to_json.py`)

- Detects device (`cuda` if available, otherwise CPU)
- Loads Whisper `base` model (fallback to `tiny` on failure)
- Transcribes each audio file with language set to English
- Extracts robust chunk records per segment:
	- `number`, `title`, `start`, `end`, `text`
- Skips empty/malformed segments safely
- Saves per-audio JSON output in `jsons/` as `audio_name.mp3.json`

#### Step 3.3: Transcript -> embeddings (`preprocess_json.py`)

- Reads all transcript JSON files from `jsons/`
- Embeds chunk text via Ollama `/api/embed` with `nomic-embed-text`
- Uses batched embedding requests (`batch_size=128`) for throughput
- Includes retry + fallback behavior on request failure
- Builds unified dataframe and stores:
	- `embeddings.joblib` (records + vectors)
	- `embedding_matrix.npy` (precomputed `float32` matrix)

### 4) Question answering flow (`dashboard.py`)

When a user asks a question:

1. Embed the question with `nomic-embed-text`
2. Compute cosine similarity against precomputed chunk matrix
3. Select top-5 most relevant chunks
4. Construct context prompt from retrieved chunk metadata
5. Generate concise answer via Ollama `/api/generate` using `qwen2.5:1.5b`
6. Show response in UI with source-aware context

## Ollama Models Used

- **`nomic-embed-text`**
	- Used for both document chunk embeddings and query embeddings
	- Enables semantic retrieval (meaning-based search)

- **`qwen2.5:1.5b`**
	- Used for final answer generation from retrieved context
	- Chosen for lightweight local inference and responsiveness

## Performance Optimizations Implemented

- Parallel video conversion with bounded worker pool
- GPU-aware Whisper transcription (`cuda` when available)
- Batched embedding calls to reduce HTTP overhead
- HTTP connection pooling + retry logic in embedding stage
- Precomputed `float32` embedding matrix for faster similarity search
- Persistent artifact caching (`joblib` + `.npy`)
- Threaded processing in UI so app remains responsive

## Setup

### 1) Install Python dependencies

```bash
pip install -r requirements.txt
```

### 2) Ensure FFmpeg is installed and available in PATH

```bash
ffmpeg -version
```

### 3) Start Ollama

```bash
ollama serve
```

### 4) Pull required models (first time only)

```bash
ollama pull nomic-embed-text
ollama pull qwen2.5:1.5b
```

## Run the App

```bash
python dashboard.py
```

## Usage

1. Click **Upload Videos** and select lecture files
2. Click **Process Videos**
	 - Step 1: convert to MP3
	 - Step 2: transcribe to JSON chunks
	 - Step 3: generate embedding index
3. Enter your question and click **Ask**
4. Read generated answer in the response panel

## Data Artifacts

- `jsons/*.json`: chunked transcript files with timestamps
- `embeddings.joblib`: dataframe including text + metadata + embeddings
- `embedding_matrix.npy`: dense matrix used for fast similarity matching

## Notes

- The current startup flow clears prior videos/audios/jsons/cache each run for a fresh processing cycle.
- Keep Ollama running while asking questions.
- First-time model pulls may take several minutes depending on network speed.

## Troubleshooting

- **Cannot connect to Ollama**
	- Start server: `ollama serve`
	- Verify endpoint: `http://localhost:11434`

- **No embeddings / no answers**
	- Re-run full pipeline from dashboard
	- Check if `embeddings.joblib` and `embedding_matrix.npy` were created

- **FFmpeg conversion fails**
	- Confirm FFmpeg installation and PATH setup
	- Try `ffmpeg -version` in terminal

- **Slow transcription**
	- Use GPU-enabled PyTorch + CUDA where available
	- Keep fewer heavy processes running in parallel

## Future Improvements

- Smarter chunking strategies for better retrieval granularity
- Evaluation metrics for retrieval and answer quality
- Optional citation rendering from exact chunk timestamps
- Incremental indexing without full cache reset
