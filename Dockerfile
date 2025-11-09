# Race Bib Analyzer Docker Image
# Base image with Tesseract OCR and Python
FROM ubuntu:22.04

# Avoid interactive prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    tesseract-ocr \
    tesseract-ocr-eng \
    libopencv-dev \
    python3-opencv \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip3 install --no-cache-dir -r requirements.txt

# Copy scripts
COPY scripts/ ./scripts/

# Make scripts executable
RUN chmod +x scripts/*.py

# Set Python path to include scripts directory
ENV PYTHONPATH=/app/scripts:$PYTHONPATH

# Create wrapper script for racebib command
RUN echo '#!/bin/bash' > /usr/local/bin/racebib && \
    echo 'cd /app && python3 -m scripts.cli "$@"' >> /usr/local/bin/racebib && \
    chmod +x /usr/local/bin/racebib

# Default command (can be overridden)
ENTRYPOINT ["racebib"]
CMD ["--help"]

