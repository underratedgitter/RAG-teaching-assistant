import streamlit as st
import os
import json
import subprocess
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
from concurrent.futures import ThreadPoolExecutor
import requests
import whisper
import torch
from sklearn.metrics.pairwise import cosine_similarity
import joblib
from datetime import datetime

warnings.filterwarnings("ignore")

# ==================== CONFIGURATION ====================
st.set_page_config(
    page_title="RAG Teaching Assistant",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

OLLAMA_URL = "http://localhost:11434"
EMBEDDING_MODEL = "nomic-embed-text"
LLM_MODEL = "qwen2.5:1.5b"
WHISPER_MODEL = "tiny"  # ⚡ Fast model for best performance

DIRS = {
    "videos": "videos",
    "audios": "audios",
    "jsons": "jsons",
    "embeddings": "embeddings.joblib",
    "matrix": "embedding_matrix.npy"
}

# Create directories
for dir_name in [DIRS["videos"], DIRS["audios"], DIRS["jsons"]]:
    Path(dir_name).mkdir(exist_ok=True)

# ==================== CACHING & SESSION STATE ====================

@st.cache_resource
def load_whisper_model():
    """Cache Whisper model in memory"""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    st.write(f"🔧 Loading Whisper on {device.upper()}...")
    return whisper.load_model(WHISPER_MODEL, device=device)

@st.cache_resource
def load_embeddings():
    """Cache embeddings in memory"""
    if Path(DIRS["embeddings"]).exists():
        return joblib.load(DIRS["embeddings"]), np.load(DIRS["matrix"])
    return None, None

def save_embeddings(embeddings_data, matrix):
    """Save embeddings with compression"""
    joblib.dump(embeddings_data, DIRS["embeddings"], compress=3)
    np.save(DIRS["matrix"], matrix.astype(np.float32))

# Initialize session state
if "processed_videos" not in st.session_state:
    st.session_state.processed_videos = set()
if "embeddings_data" not in st.session_state:
    st.session_state.embeddings_data, st.session_state.embedding_matrix = load_embeddings()

# ==================== UTILITY FUNCTIONS ====================

def check_ollama():
    """Check if Ollama server is running"""
    try:
        response = requests.get(f"{OLLAMA_URL}/api/tags", timeout=2)
        return response.status_code == 200
    except:
        return False

def get_ollama_embedding(text):
    """Get embedding from Ollama with retry logic"""
    max_retries = 3
    for attempt in range(max_retries):
        try:
            response = requests.post(
                f"{OLLAMA_URL}/api/embed",
                json={"model": EMBEDDING_MODEL, "input": text},
                timeout=30
            )
            if response.status_code == 200:
                return np.array(response.json()["embeddings"][0])
        except:
            if attempt < max_retries - 1:
                import time
                time.sleep(1)
                continue
    return None

def extract_audio(video_path):
    """Extract audio from video using FFmpeg"""
    video_name = Path(video_path).stem
    audio_path = f"{DIRS['audios']}/1_{video_name}.mp3"
    
    if Path(audio_path).exists():
        return audio_path
    
    cmd = [
        "ffmpeg", "-i", video_path,
        "-vn", "-acodec", "libmp3lame", "-q:a", "4",
        "-threads", "0", audio_path, "-y"
    ]
    
    try:
        subprocess.run(cmd, capture_output=True, check=True, timeout=600)
        return audio_path
    except Exception as e:
        st.error(f"❌ Audio extraction failed: {str(e)}")
        return None

def transcribe_audio(audio_path):
    """Transcribe audio using Whisper"""
    try:
        model = load_whisper_model()
        result = model.transcribe(audio_path, language="en", fp16=False)
        
        # Format output
        chunks = []
        video_name = Path(audio_path).stem.replace("_", " ").split(".")[0][2:]
        
        for i, segment in enumerate(result["segments"]):
            chunks.append({
                "number": "1",
                "title": video_name,
                "start": segment["start"],
                "end": segment["end"],
                "text": segment["text"].strip(),
                "chunk_id": i
            })
        
        return {
            "chunks": chunks,
            "text": result["text"],
            "language": result["language"]
        }
    except Exception as e:
        st.error(f"❌ Transcription failed: {str(e)}")
        return None

def create_embeddings_batch(chunks, batch_size=64):
    """Create embeddings in batches for speed"""
    embeddings = []
    metadata = []
    
    total_chunks = len(chunks)
    progress_bar = st.progress(0)
    
    for i in range(0, total_chunks, batch_size):
        batch = chunks[i:i+batch_size]
        batch_texts = [c["text"] for c in batch]
        
        try:
            # Batch request to Ollama
            response = requests.post(
                f"{OLLAMA_URL}/api/embed",
                json={"model": EMBEDDING_MODEL, "input": batch_texts},
                timeout=60
            )
            
            if response.status_code == 200:
                batch_embeddings = response.json()["embeddings"]
                embeddings.extend(batch_embeddings)
                metadata.extend(batch)
                
                progress = (i + len(batch)) / total_chunks
                progress_bar.progress(min(progress, 0.99))
        except Exception as e:
            st.warning(f"⚠️ Batch {i//batch_size + 1} failed: {str(e)}")
            continue
    
    progress_bar.progress(1.0)
    return embeddings, metadata

def process_video(video_path, progress_placeholder):
    """Complete video processing pipeline"""
    video_name = Path(video_path).name
    
    try:
        # Step 1: Extract audio
        with progress_placeholder.container():
            st.write("📊 Step 1/4: Extracting audio...")
        audio_path = extract_audio(video_path)
        if not audio_path:
            return False
        
        # Step 2: Transcribe
        with progress_placeholder.container():
            st.write("📊 Step 2/4: Transcribing audio (Whisper)...")
        transcript = transcribe_audio(audio_path)
        if not transcript:
            return False
        
        # Save JSON
        json_path = f"{DIRS['jsons']}/{Path(audio_path).stem}.json"
        with open(json_path, 'w') as f:
            json.dump(transcript, f, indent=2)
        
        # Step 3: Create embeddings
        with progress_placeholder.container():
            st.write("📊 Step 3/4: Creating embeddings...")
        embeddings, metadata = create_embeddings_batch(transcript["chunks"])
        
        if not embeddings:
            st.error(f"❌ No embeddings created for {video_name}")
            return False
        
        # Step 4: Update embedding database
        with progress_placeholder.container():
            st.write("📊 Step 4/4: Indexing embeddings...")
        
        if st.session_state.embeddings_data is None:
            st.session_state.embeddings_data = metadata
            st.session_state.embedding_matrix = np.array(embeddings, dtype=np.float32)
        else:
            st.session_state.embeddings_data.extend(metadata)
            new_matrix = np.vstack([
                st.session_state.embedding_matrix,
                np.array(embeddings, dtype=np.float32)
            ])
            st.session_state.embedding_matrix = new_matrix
        
        # Save to disk
        save_embeddings(st.session_state.embeddings_data, st.session_state.embedding_matrix)
        st.session_state.processed_videos.add(video_name)
        
        return True
    
    except Exception as e:
        st.error(f"❌ Failed to process {video_name}: {str(e)}")
        return False

def semantic_search(question, top_k=5):
    """Search for relevant chunks using semantic similarity"""
    if st.session_state.embeddings_data is None:
        return []
    
    # Get embedding for question
    question_embedding = get_ollama_embedding(question)
    if question_embedding is None:
        st.error("❌ Failed to embed question")
        return []
    
    # Normalize for cosine similarity
    question_embedding = question_embedding.reshape(1, -1)
    
    # Calculate similarities
    similarities = cosine_similarity(
        question_embedding,
        st.session_state.embedding_matrix
    )[0]
    
    # Get top-k
    top_indices = np.argsort(similarities)[-top_k:][::-1]
    
    results = []
    for idx in top_indices:
        if similarities[idx] > 0.3:  # Confidence threshold
            chunk = st.session_state.embeddings_data[idx]
            results.append({
                "text": chunk["text"],
                "timestamp": f"{int(chunk['start']//60)}:{int(chunk['start']%60):02d}",
                "similarity": float(similarities[idx]),
                "video": chunk["title"]
            })
    
    return results

def generate_answer(question, context_chunks):
    """Generate answer using Ollama LLM"""
    context_text = "\n\n".join([
        f"[{c['video']} at {c['timestamp']}]: {c['text']}"
        for c in context_chunks
    ])
    
    prompt = f"""Based on the following information from lecture videos, answer the question concisely:

Context:
{context_text}

Question: {question}

Answer (be specific and mention the video timestamp):"""
    
    try:
        response = requests.post(
            f"{OLLAMA_URL}/api/generate",
            json={
                "model": LLM_MODEL,
                "prompt": prompt,
                "temperature": 0.3,
                "num_predict": 150,
                "stream": False
            },
            timeout=60
        )
        
        if response.status_code == 200:
            return response.json().get("response", "No response generated")
        else:
            return "❌ Failed to generate answer"
    except Exception as e:
        return f"❌ Error: {str(e)}"

# ==================== UI COMPONENTS ====================

st.markdown("""
<style>
    .main {
        padding: 0;
    }
    .stTabs [data-baseweb="tab-list"] button [data-testid="stMarkdownContainer"] p {
        font-size: 1.25rem;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.title("🎓 RAG Teaching Assistant")
st.markdown("""
**Upload lecture videos and ask AI questions with precise timestamps**

Built with ⚡ Streamlit, 🤖 Whisper, 🔍 Ollama, and 📊 Semantic Search
""")

# Check Ollama
if not check_ollama():
    st.error("❌ Ollama server is not running! Start it with: `ollama serve`")
    st.stop()
else:
    st.success("✅ Ollama connected")

# Tabs
tabs = st.tabs(["📤 Upload & Process", "❓ Ask Questions", "📊 Dashboard"])

# ==================== TAB 1: UPLOAD & PROCESS ====================
with tabs[0]:
    st.header("Upload Lecture Videos")
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        uploaded_files = st.file_uploader(
            "Choose video files (MP4, AVI, MKV, MOV)",
            type=["mp4", "avi", "mkv", "mov"],
            accept_multiple_files=True,
            help="Upload lecture videos to process"
        )
    
    with col2:
        if st.button("🚀 Process Videos", use_container_width=True):
            if not uploaded_files:
                st.warning("⚠️ No files selected")
            else:
                st.success(f"✅ Selected {len(uploaded_files)} video(s)")
    
    if uploaded_files:
        st.write(f"**Videos to process: {len(uploaded_files)}**")
        
        progress_placeholder = st.empty()
        results = []
        
        for idx, file in enumerate(uploaded_files):
            video_path = f"{DIRS['videos']}/{file.name}"
            
            # Save uploaded file
            with open(video_path, 'wb') as f:
                f.write(file.getbuffer())
            
            st.write(f"**({idx+1}/{len(uploaded_files)}) Processing: {file.name}**")
            
            if process_video(video_path, progress_placeholder):
                st.success(f"✅ {file.name} processed successfully!")
                results.append((file.name, True))
            else:
                st.error(f"❌ {file.name} failed")
                results.append((file.name, False))
        
        # Summary
        st.divider()
        successful = sum(1 for _, success in results if success)
        st.info(f"**Summary:** {successful}/{len(results)} videos processed successfully")

# ==================== TAB 2: ASK QUESTIONS ====================
with tabs[1]:
    st.header("Ask Questions About Your Lectures")
    
    if st.session_state.embeddings_data is None:
        st.warning("⚠️ No videos processed yet. Upload videos first!")
    else:
        st.success(f"✅ {len(st.session_state.embeddings_data)} chunks indexed and ready")
        
        question = st.text_input(
            "🤔 Ask a question about your lectures:",
            placeholder="e.g., What is machine learning?",
            help="Ask any question about the uploaded lecture content"
        )
        
        if question:
            with st.spinner("🔍 Searching for relevant content..."):
                relevant_chunks = semantic_search(question, top_k=5)
            
            if not relevant_chunks:
                st.warning("⚠️ No relevant content found. Try a different question.")
            else:
                st.write("**📌 Relevant sections found:**")
                for chunk in relevant_chunks:
                    st.write(f"- {chunk['video']} @ {chunk['timestamp']} (Relevance: {chunk['similarity']:.1%})")
                
                with st.spinner("🤖 Generating answer..."):
                    answer = generate_answer(question, relevant_chunks)
                
                st.success("**✅ Answer:**")
                st.write(answer)
                
                # Source information
                st.divider()
                st.markdown("**📌 Sources:**")
                for idx, chunk in enumerate(relevant_chunks, 1):
                    with st.expander(f"Source {idx}: {chunk['video']} @ {chunk['timestamp']}"):
                        st.write(chunk['text'])

# ==================== TAB 3: DASHBOARD ====================
with tabs[2]:
    st.header("📊 System Dashboard")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("📹 Videos Processed", len(st.session_state.processed_videos))
    
    with col2:
        chunks_count = len(st.session_state.embeddings_data) if st.session_state.embeddings_data else 0
        st.metric("📝 Total Chunks", chunks_count)
    
    with col3:
        device = "GPU" if torch.cuda.is_available() else "CPU"
        st.metric("⚡ Processor", device)
    
    with col4:
        if st.session_state.embedding_matrix is not None:
            size_mb = st.session_state.embedding_matrix.nbytes / (1024 * 1024)
            st.metric("💾 Index Size", f"{size_mb:.1f} MB")
    
    st.divider()
    
    # Processed videos list
    if st.session_state.processed_videos:
        st.subheader("✅ Processed Videos")
        for video in sorted(st.session_state.processed_videos):
            st.write(f"• {video}")
    
    # System info
    st.subheader("🔧 System Information")
    col1, col2 = st.columns(2)
    
    with col1:
        st.write(f"**Whisper Model:** {WHISPER_MODEL}")
        st.write(f"**LLM Model:** {LLM_MODEL}")
        st.write(f"**Embedding Model:** {EMBEDDING_MODEL}")
    
    with col2:
        st.write(f"**Ollama Status:** {'✅ Connected' if check_ollama() else '❌ Disconnected'}")
        st.write(f"**Device:** {'🎮 CUDA Available' if torch.cuda.is_available() else '💻 CPU Only'}")
        st.write(f"**Last Updated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Clear cache button
    if st.button("🗑️ Clear All Data", use_container_width=True):
        st.session_state.clear()
        st.cache_resource.clear()
        st.success("✅ Cache cleared!")
        st.rerun()
