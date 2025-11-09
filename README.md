# Race Bib Analyzer

A Docker-first application for downloading race photos and extracting bib numbers using OCR. The system stores bib-to-URL mappings and provides query capabilities to find images containing specific bib numbers.

## Features

- **Gallery Downloader**: Downloads full-size images from gallery pages with retry logic and rate limiting
- **Bib Detection**: OCR pipeline using Tesseract + OpenCV with multiple rotation and PSM mode support
- **Bib Storage**: Stores detected bib numbers linked to image URLs with confidence scores
- **Query System**: Find image URLs containing specific bib numbers
- **HTML Reports**: Generate interactive HTML reports with thumbnails and statistics
- **Performance Benchmarking**: Measure throughput and memory usage

## Quick Start with Docker

### Pull from GitHub Container Registry

```bash
docker pull ghcr.io/OWNER/bibanalyser:latest
```

### Download Images

```bash
docker run --rm -v $(pwd)/data:/data \
  ghcr.io/OWNER/bibanalyser:latest \
  download --url https://example.com/gallery --out /data/photos
```

### Extract Bib Numbers

```bash
docker run --rm -v $(pwd)/data:/data \
  ghcr.io/OWNER/bibanalyser:latest \
  ocr --input /data/photos --output /data/results.csv \
  --db /data/bibdb.json --image-url https://example.com/gallery/
```

### Query for Bib Numbers

```bash
docker run --rm -v $(pwd)/data:/data \
  ghcr.io/OWNER/bibanalyser:latest \
  query --bib 1234 --db /data/bibdb.json
```

## Local Development

### Prerequisites

- Python 3.9+
- Tesseract OCR:
  - macOS: `brew install tesseract`
  - Ubuntu/Debian: `sudo apt-get install tesseract-ocr`
  - Windows: [Download installer](https://github.com/UB-Mannheim/tesseract/wiki)

### Setup

```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install package (optional, for CLI commands)
pip install -e .
```

### Usage

```bash
# Download images
racebib download --url https://example.com/gallery --out ./photos

# Extract bibs
racebib ocr --input ./photos --output results.csv \
  --db bibdb.json --image-url https://example.com/gallery/

# Query bibs
racebib query --bib 1234 --db bibdb.json --format json

# Generate HTML report
python scripts/generate_report.py --db bibdb.json --output report.html

# Run benchmark
python scripts/benchmark.py --input ./photos --output benchmark.json --limit 500
```

## Documentation

- [Installation Guide](docs/installation.md) - Docker and local setup
- [Usage Guide](docs/usage.md) - Command reference and examples
- [Troubleshooting](docs/troubleshooting.md) - Common issues and solutions

## Project Structure

- `scripts/` - Main application scripts
  - `gallery_downloader.py` - Download images from gallery pages
  - `bib_finder.py` - OCR pipeline for bib detection
  - `bib_storage.py` - Storage module for bib-to-URL mappings
  - `bib_query.py` - Query command for finding bibs
  - `generate_report.py` - HTML report generator
  - `benchmark.py` - Performance benchmarking tool
  - `cli.py` - Main CLI entry point
- `tests/` - Unit tests
- `docs/` - Documentation
- `Dockerfile` - Docker container definition
- `.github/workflows/ci.yml` - CI/CD pipeline

## CI/CD

The project includes GitHub Actions workflow that:
- Runs linting and tests
- Builds Docker image
- Pushes to GitHub Container Registry (ghcr.io)

## License

MIT
