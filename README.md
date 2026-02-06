# RAG Teaching Assistant

> **Ask questions about your videos, get AI-powered answers with timestamps**

A powerful AI-driven system that lets you upload lecture videos, automatically transcribes them, and answers your questions with precise timestamps and references.

---

## Table of Contents

- [Quick Start](#quick-start)
- [How It Works](#how-it-works)
- [Pipeline Architecture](#pipeline-architecture)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
- [Configuration](#configuration)
- [Performance](#performance)
- [Troubleshooting](#troubleshooting)
- [Project Structure](#project-structure)
- [Technology Stack](#technology-stack)

---

## Quick Start

### Prerequisites

- Windows/Linux/macOS
- Python 3.10+
- FFmpeg installed
- Ollama running (`ollama serve`)

### 1. Clone and Install

```bash
cd "your/project/path"
pip install -r requirements.txt
```

### 2. Install PyTorch (GPU acceleration)

```bash
# NVIDIA GPU (CUDA 11.8)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# AMD GPU (ROCm)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm5.7

# CPU only
pip install torch torchvision torchaudio
```

### 3. Start Ollama Server

In a separate terminal:

```bash
ollama serve
```

Then pull the models:

```bash
ollama pull nomic-embed-text
ollama pull qwen2.5:1.5b
```

### 4. Run Dashboard

```bash
python dashboard.py
```

The GUI will open. Upload videos and click **"Process Videos"**.

---

## How It Works

The system follows a **Retrieval-Augmented Generation (RAG)** pipeline:

### User Workflow

```
1. Upload Videos
       ↓
2. Click "Process Videos"
       ↓
3. Automatic Processing
   ├─ Extract audio
   ├─ Transcribe speech
   └─ Create embeddings
       ↓
4. Ask Questions
       ↓
5. Get Answers
   ├─ Find relevant segments
   └─ Generate response with timestamps
```

### What Happens Behind the Scenes

1. **Video Upload** - Place MP4/AVI/MKV/MOV files in the `videos/` folder
2. **Audio Extraction** - Convert video to MP3 using FFmpeg
3. **Transcription** - Use OpenAI Whisper to transcribe audio to text
4. **Chunking** - Split transcription into manageable chunks (by speaker segments)
5. **Embedding** - Generate numeric embeddings for semantic search
6. **Indexing** - Save embeddings for fast retrieval
7. **Query** - When you ask a question, it:
   - Embeds your question
   - Searches for similar chunks using cosine similarity
   - Passes top matches to LLM
   - Returns AI-generated answer with video references

---

## Pipeline Architecture

### Data Flow

```
[VIDEO FILES]
     │
     ├─→ video_to_mp3.py (Parallel FFmpeg)
     │       └─→ [MP3 AUDIO FILES]
     │
     ├─→ mp3_to_json.py (Whisper base model)
     │       └─→ [JSON TRANSCRIPTS]
     │
     ├─→ preprocess_json.py (Ollama embeddings)
     │       └─→ [EMBEDDINGS DATABASE]
     │
[EMBEDDINGS INDEXED]
     │
     └─→ dashboard.py → User asks question → Get answer with timestamps
```

---

## Requirements

### Software

| Tool | Version | Purpose |
|------|---------|---------|
| **Python** | 3.10+ | Runtime |
| **FFmpeg** | Latest | Video/audio processing |
| **Ollama** | Latest | LLM inference |
| **CUDA** | 11.8+ | GPU acceleration |

### Hardware

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| **CPU** | 4 cores | 8+ cores |
| **RAM** | 8 GB | 16+ GB |
| **GPU** | None | NVIDIA RTX 3050+ |
| **VRAM** | N/A | 4+ GB |

### Ollama Models

```bash
ollama pull nomic-embed-text
ollama pull qwen2.5:1.5b
```

---

## Installation

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Install PyTorch (GPU)

```bash
# NVIDIA
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# AMD
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm5.7

# CPU
pip install torch torchvision torchaudio
```

### 3. Install FFmpeg

**Windows:** https://ffmpeg.org/download.html
**Linux:** `sudo apt-get install ffmpeg`
**Mac:** `brew install ffmpeg`

### 4. Start Ollama

```bash
ollama serve
```

---

## Usage

### Step 1: Upload Videos

Place videos in the `videos/` folder (MP4, AVI, MKV, MOV)

### Step 2: Start Dashboard

```bash
python dashboard.py
```

### Step 3: Click "Process Videos"

Automated pipeline:
- Converts videos to MP3 (parallel)
- Transcribes with Whisper (GPU)
- Generates embeddings
- Indexes for search

### Step 4: Ask Questions

Type natural language questions, get answers with timestamps.

---

## Configuration

### Whisper Model

Edit `mp3_to_json.py`:

```python
model = whisper.load_model("base", device=device)
```

| Model | Speed | Quality | VRAM |
|-------|-------|---------|------|
| **tiny** | ⚡⚡⚡⚡ | Poor | 1GB |
| **base** | ⚡⚡⚡ | Good | 1GB |
| **small** | ⚡⚡ | Very Good | 2GB |
| **medium** | ⚡ | Excellent | 5GB |
| **large** | 🐢 | Perfect | 10GB |

### LLM Model

Edit `dashboard.py`:

```python
"model": "qwen2.5:1.5b"
```

### Language

Edit `mp3_to_json.py`:

```python
language="en"  # 'hi', 'es', 'fr', etc.
```

---

## Performance

| Operation | CPU | GPU (RTX 3070) |
|-----------|-----|----------------|
| Video → MP3 (1 min) | 8s | 8s |
| Transcribe (5 min) | 5 min | 15 sec |
| Embeddings (100 chunks) | 3s | 3s |
| Query | 2s | 1s |
| **Total (5 min video)** | ~5 min | **~20 sec** |

---

## Troubleshooting

### Connection Issues

**"Cannot connect to Ollama"**
```bash
ollama serve
```

**"Model not found"**
```bash
ollama pull nomic-embed-text
ollama pull qwen2.5:1.5b
```

### Processing Issues

**"FFmpeg not found"** - Install from https://ffmpeg.org/download.html

**"Out of memory"** - Use smaller Whisper model (`tiny` or `base`)

**"CUDA out of memory"** - Close other apps or use CPU

---

## Project Structure

```
rag-teach/
├── dashboard.py          # GUI application
├── video_to_mp3.py       # Video conversion
├── mp3_to_json.py        # Transcription
├── preprocess_json.py    # Embeddings
├── requirements.txt      # Dependencies
├── README.md             # Documentation
├── videos/               # Input videos
├── audios/               # Extracted audio
├── jsons/                # Transcripts
├── embeddings.joblib     # Indexed data
└── embedding_matrix.npy  # Search matrix
```

---

## Technology Stack

| Layer | Technology | Purpose |
|-------|-----------|---------|
| **UI** | Tkinter | GUI |
| **Video** | FFmpeg | Conversion |
| **Audio** | Whisper | Transcription |
| **Embeddings** | Nomic | Vectors |
| **Search** | Scikit-learn | Similarity |
| **LLM** | Qwen 2.5 | Answers |
| **GPU** | PyTorch CUDA | Acceleration |

---

## License

MIT License - Free to use and modify

---

**Built with ❤️ using Whisper • Ollama • PyTorch**
