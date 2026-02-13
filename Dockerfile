FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    ffmpeg \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install Ollama
RUN curl https://ollama.ai/install.sh | bash

# Copy requirements
COPY requirements.txt .

# Install Python packages
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download Ollama models (KEY FOR PERFORMANCE!)
# This runs during build so models are baked into image
RUN ollama serve & \
    sleep 5 && \
    ollama pull nomic-embed-text && \
    ollama pull qwen2.5:1.5b && \
    pkill -f "ollama serve"

# Copy application
COPY streamlit_app.py .
COPY video_to_mp3.py .
COPY mp3_to_json.py .
COPY preprocess_json.py .

# Create necessary directories
RUN mkdir -p videos audios jsons

# Expose Streamlit port
EXPOSE 8501

# Health check
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health || exit 1

# Start Ollama in background + Streamlit
CMD ollama serve & \
    sleep 5 && \
    streamlit run streamlit_app.py \
    --server.port=8501 \
    --server.address=0.0.0.0 \
    --server.headless=true \
    --logger.level=error
