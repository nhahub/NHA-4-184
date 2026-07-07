FROM python:3.12-slim

# System dependencies (ffmpeg for Whisper, libpq for PostgreSQL)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential ffmpeg libpq-dev curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy and install Python dependencies
COPY Backend/requirements.txt /tmp/requirements.txt

# Install PyTorch CPU-only separately (saves ~1.5GB vs GPU version)
RUN sed -i '/^torch==/d' /tmp/requirements.txt && \
    pip install --no-cache-dir torch==2.10.0+cpu --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir -r /tmp/requirements.txt

# Pre-download ML models during build (avoid downloading on every container start)
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"
RUN python -c "import whisper; whisper.load_model('small')"

# Copy project code
COPY Backend/ /app/Backend/
COPY data/processed/ /app/data/processed/

# Create vector_db directory (will be mounted as volume)
RUN mkdir -p /app/data/vector_db

# Copy and set entrypoint
COPY entrypoint.sh /app/entrypoint.sh
RUN chmod +x /app/entrypoint.sh

# Working directory = Backend (where uvicorn runs)
WORKDIR /app/Backend

EXPOSE 8000

ENTRYPOINT ["/app/entrypoint.sh"]
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
