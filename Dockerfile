FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    ffmpeg \
    curl \
    git \
    build-essential \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Install Ollama
RUN curl https://ollama.ai/install.sh | bash

# Copy requirements
COPY requirements_streamlit.txt requirements.txt

# Install Python packages (use only binary wheels to avoid long compilation)
RUN pip install --upgrade pip setuptools wheel && \
    pip install --only-binary :all: --default-timeout=1000 -r requirements.txt || \
    pip install --default-timeout=1000 -r requirements.txt

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
