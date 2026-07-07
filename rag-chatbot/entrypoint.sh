#!/bin/bash
set -e

echo "Checking if ChromaDB needs data ingestion..."

if [ -z "$(ls -A /app/data/vector_db/ 2>/dev/null)" ]; then
    echo "ChromaDB is empty. Running data ingestion..."
    python ingest_data.py
    echo "Data ingestion complete!"
else
    echo "ChromaDB already has data. Skipping ingestion."
fi

echo "Starting application..."
exec "$@"
