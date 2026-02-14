# RAG Teaching Assistant (Clean Start)

This project processes lecture videos into searchable text chunks and answers questions using local embeddings + an LLM.

## Current Entry Point

- Main app: `dashboard.py` (Tkinter desktop UI)
- Legacy web UI files have been removed for a clean reset.

## Project Structure

- `videos/` - Input lecture videos
- `audios/` - Extracted MP3 audio
- `jsons/` - Whisper transcription chunks
- `video_to_mp3.py` - Video to audio conversion
- `mp3_to_json.py` - Audio transcription
- `preprocess_json.py` - Embedding generation/indexing
- `dashboard.py` - Desktop interface and orchestration

## Setup

1. Install Python dependencies:

```bash
pip install -r requirements.txt
```

2. Start Ollama server (required for embeddings/answers):

```bash
ollama serve
```

3. Pull required Ollama models (first time only):

```bash
ollama pull nomic-embed-text
ollama pull qwen2.5:1.5b
```

## Run

```bash
python dashboard.py
```

## Pipeline Summary

1. Put videos in `videos/`
2. Convert videos to MP3 (`video_to_mp3.py`)
3. Transcribe MP3 to JSON chunks (`mp3_to_json.py`)
4. Build embeddings from chunks (`preprocess_json.py`)
5. Ask questions from the dashboard
