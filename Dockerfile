# Race Bib Analyzer Docker Image
# Lightweight base image with Python
FROM python:3.11-slim

# Install system dependencies (Tesseract OCR only)
RUN apt-get update && apt-get install -y \
    tesseract-ocr \
    tesseract-ocr-eng \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy scripts
COPY scripts/ ./scripts/

# Make scripts executable
RUN chmod +x scripts/*.py

# Set Python path to include scripts directory
ENV PYTHONPATH=/app/scripts

# Create wrapper script for racebib command
RUN echo '#!/bin/bash' > /usr/local/bin/racebib && \
    echo 'cd /app && python -m scripts.cli "$@"' >> /usr/local/bin/racebib && \
    chmod +x /usr/local/bin/racebib

# Default command (can be overridden)
ENTRYPOINT ["racebib"]
CMD ["--help"]

