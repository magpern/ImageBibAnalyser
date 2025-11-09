# Installation Guide

## Docker Installation (Recommended)

Docker is the primary deployment method for the Race Bib Analyzer. The Docker image includes all dependencies including Tesseract OCR.

### Prerequisites

- Docker installed on your system
  - [Docker Desktop](https://www.docker.com/products/docker-desktop) for Windows/Mac
  - [Docker Engine](https://docs.docker.com/engine/install/) for Linux

### Pull from GitHub Container Registry

```bash
docker pull ghcr.io/OWNER/bibanalyser:latest
```

Replace `OWNER` with your GitHub username or organization name.

### Build from Source

```bash
git clone <repository-url>
cd bibAnalyser
docker build -t bibanalyser:latest .
```

## Local Development Setup (venv)

For local development and testing, you can set up a Python virtual environment.

### Prerequisites

- Python 3.9 or higher
- Tesseract OCR installed on your system:
  - **macOS**: `brew install tesseract`
  - **Ubuntu/Debian**: `sudo apt-get install tesseract-ocr`
  - **Windows**: Download from [Tesseract GitHub](https://github.com/UB-Mannheim/tesseract/wiki)

### Setup Steps

1. Create a virtual environment:

```bash
python -m venv .venv
```

2. Activate the virtual environment:

```bash
# Windows
.venv\Scripts\activate

# macOS/Linux
source .venv/bin/activate
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

4. (Optional) Install in development mode:

```bash
pip install -e .
```

This installs the package with CLI entry points (`racebib download`, `racebib ocr`, `racebib query`).

### Verify Installation

Test that everything works:

```bash
# Test download command
racebib download --help

# Test OCR command
racebib ocr --help

# Test query command
racebib query --help
```

## Troubleshooting

### Tesseract Not Found

If you get an error about Tesseract not being found:

1. Verify Tesseract is installed: `tesseract --version`
2. If installed but not on PATH, use `--tesseract-cmd` flag:
   ```bash
   racebib ocr --input ./images --output results.csv --tesseract-cmd /path/to/tesseract
   ```

### Docker Build Issues

If Docker build fails:

1. Ensure you have sufficient disk space
2. Check that all files are present (requirements.txt, scripts/, etc.)
3. Try building with `--no-cache` flag:
   ```bash
   docker build --no-cache -t bibanalyser:latest .
   ```

### Import Errors in Local Development

If you get import errors when running scripts directly:

1. Use the CLI entry points: `racebib download`, `racebib ocr`, `racebib query`
2. Or run as modules: `python -m scripts.gallery_downloader`
3. Ensure PYTHONPATH includes the scripts directory

