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
COPY requirements_streamlit.txt requirements.txt

# Install Python packages
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download Ollama models (KEY FOR PERFORMANCE!)
RUN ollama serve & \
    sleep 10 && \
    ollama pull nomic-embed-text && \
    ollama pull qwen2.5:1.5b && \
    pkill -f "ollama serve"

# Copy application
COPY streamlit_app.py .

# Create necessary directories
RUN mkdir -p videos audios jsons

# Expose port for HF Spaces
EXPOSE 7860

# Start Ollama in background + Streamlit
CMD ollama serve & \
    sleep 5 && \
    streamlit run streamlit_app.py \
    --server.port=7860 \
    --server.address=0.0.0.0 \
    --server.headless=true \
    --logger.level=error
