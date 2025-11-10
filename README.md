# Race Bib Analyzer

A Docker-first application for downloading race photos and extracting bib numbers using OCR. The system stores bib-to-URL mappings and provides query capabilities to find images containing specific bib numbers.

## Features

- **Gallery Downloader**: Downloads full-size images from gallery pages with retry logic and rate limiting
- **Bib Detection**: OCR pipeline using Tesseract + OpenCV with multiple rotation and PSM mode support
- **Bib Storage**: Stores detected bib numbers linked to image URLs with confidence scores
- **Query System**: Find image URLs containing specific bib numbers
- **Deep Learning Pipeline (optional)**: YOLO + PaddleOCR detector/recognizer for more robust bib extraction
- **HTML Reports**: Generate interactive HTML reports with thumbnails and statistics
- **Performance Benchmarking**: Measure throughput and memory usage

## Quick Start with Docker

**Advantage:** Docker includes Tesseract OCR, so you don't need to install it on your local machine!

### Build the Docker Image

```bash
docker build -t bibanalyser:latest .
```

Or pull from GitHub Container Registry (after pushing):

```bash
docker pull ghcr.io/magpern/ImageBibAnalyser:latest
```

### Download Images

```bash
docker run --rm -v ${PWD}/data:/data \
  bibanalyser:latest \
  download --url https://example.com/gallery --out /data/photos --filter "b*.jpg"
```

### Extract Bib Numbers

```bash
docker run --rm -v ${PWD}/data:/data \
  bibanalyser:latest \
  ocr --input /data/photos --output /data/results.csv \
  --db /data/bibdb.json --image-url https://example.com/gallery/
```

### Query for Bib Numbers

```bash
docker run --rm -v ${PWD}/data:/data \
  bibanalyser:latest \
  query --bib 1234 --db /data/bibdb.json
```

**Note:** The Docker image uses `python:3.11-slim` as the base image and `opencv-python-headless` to minimize image size. This avoids GUI/X11/OpenGL dependencies. **No need to install Tesseract OCR on your host machine when using Docker!**

## Local Development (Without Docker)

**Note:** If you use Docker (see above), you don't need to install Tesseract OCR on your local machine!

### Prerequisites

- Python 3.9+
- **Tesseract OCR** (required for bib detection):
  - **macOS**: `brew install tesseract`
  - **Ubuntu/Debian**: `sudo apt-get install tesseract-ocr`
  - **Windows**: 
    1. Download installer from [Tesseract GitHub](https://github.com/UB-Mannheim/tesseract/wiki)
    2. Install to default location: `C:\Program Files\Tesseract-OCR\`
    3. **Important**: Add Tesseract to PATH, OR use `--tesseract-cmd` flag:
       ```bash
       racebib ocr --input ./photos --output results.csv --tesseract-cmd "C:\Program Files\Tesseract-OCR\tesseract.exe"
       ```

### Setup

**Quick setup (automated):**

```bash
# Linux/macOS
./setup_venv.sh

# Windows PowerShell
.\setup_venv.ps1
```

**Manual setup:**

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

#### Complete Workflow Example

**Step 1: Download images from gallery**
```bash
racebib download --url https://example.com/gallery --out ./photos --filter "b*.jpg"
```
- `--url`: Gallery page URL
- `--out`: Where to save downloaded images
- `--filter`: Optional - only download images matching pattern (e.g., `b*.jpg` for full-size images)
- Default selector is `img` (finds all images), override with `--selector` if needed

**Step 2: Extract bib numbers (OCR)**

**Minimal command (CSV output only):**
```bash
racebib ocr --input ./photos --output results.csv
```

**Recommended command (with database for querying):**
```bash
racebib ocr --input ./photos --output results.csv \
  --db bibdb.json --image-url https://example.com/gallery/
```

### OCR Command Options

**Required:**
- `--input <folder>` - Folder containing images to process
- `--output <file.csv>` - Output CSV file path with results

**Storage & Querying:**
- `--db <file.json>` - **Optional** - Creates JSON database for querying later (recommended if you want to search for bibs)
- `--image-url <url>` - **Required if using --db** - Base URL to construct full image URLs (e.g., `https://example.com/gallery/`)

**Image Selection:**
- `--ext <extensions>` - Image file extensions to include (default: `.jpg .jpeg .png`)
- `--limit <N>` - Process only the first N images (useful for testing)
- `--sample <N>` - Randomly sample N images to process (useful for quick testing)

**Bib Number Detection:**
- `--min-digits <N>` - Minimum number of digits in bib numbers (default: 2)
- `--max-digits <N>` - Maximum number of digits in bib numbers (default: 6)
- `--bib-pattern <regex>` - Custom regex pattern for bib numbers (overrides min/max-digits)
  - Example: `--bib-pattern "\d{4}"` for exactly 4 digits
  - Example: `--bib-pattern "^[0-9]{3,5}$"` for 3-5 digits

**OCR Quality & Performance:**
- `--min-conf <0-100>` - Minimum OCR confidence threshold (default: 60)
  - Lower values = more detections but more false positives
  - Higher values = fewer detections but more accurate
- `--workers <N>` - Number of parallel workers for processing (default: 4)
  - Increase for faster processing (uses more CPU/memory)
  - Decrease if running out of memory

**Advanced OCR Settings:**
- `--psm <modes>` - Tesseract PSM (Page Segmentation Mode) to try (default: `6 7 8 11 13`)
  - Multiple modes are tried and best result is used
  - Common modes: 6 (uniform block), 7 (single line), 8 (single word), 11 (sparse text), 13 (raw line)
- `--rotations <angles>` - Image rotation angles to try (default: `0 90 -90 180`)
  - Helps detect bibs in rotated images
  - Format: `--rotations 0 90 180` (degrees)
- `--no-focus-region` - Disable region focusing (process entire image)
  - By default, focuses on center-upper 60% of image where bibs typically are
  - Use this flag if bibs might be in other locations
- `--min-text-size <ratio>` - Minimum text size ratio to consider (default: 0.01 = 1%)
  - Filters out tiny false positives from logos/text
  - Lower values = more detections but more false positives
- `--max-text-size <ratio>` - Maximum text size ratio to consider (default: 0.3 = 30%)
  - Filters out oversized detections
  - Higher values = more detections but more false positives

**Debugging & Output:**
- `--annotate-dir <folder>` - Save annotated images with detected bibs highlighted
  - Useful for verifying what OCR is detecting
  - Shows bounding boxes and confidence scores
- `--tesseract-cmd <path>` - **Windows only** - Path to Tesseract executable if not in PATH
  - Example: `--tesseract-cmd "C:\Program Files\Tesseract-OCR\tesseract.exe"`

### Common Usage Examples

**Basic usage (CSV only):**
```bash
racebib ocr --input ./photos --output results.csv
```

**With database for querying:**
```bash
racebib ocr --input ./photos --output results.csv \
  --db bibdb.json --image-url https://example.com/gallery/
```

**Test with first 10 images:**
```bash
racebib ocr --input ./photos --output results.csv --limit 10
```

**Higher accuracy (stricter confidence):**
```bash
racebib ocr --input ./photos --output results.csv --min-conf 70
```

**Faster processing (more workers):**
```bash
racebib ocr --input ./photos --output results.csv --workers 8
```

**Debug mode (see what's detected):**
```bash
racebib ocr --input ./photos --output results.csv --annotate-dir ./annotated
```

**Custom bib pattern (exactly 4 digits):**
```bash
racebib ocr --input ./photos --output results.csv --bib-pattern "\d{4}"
```

**Aggressive mode (better detection, slower):**
```bash
racebib ocr --input ./photos --output results.csv --aggressive
```
- Lowers confidence threshold to 40
- Tries more PSM modes (6, 7, 8, 11, 13)
- Tries more rotation angles (0, 45, 90, -45, -90, 135, -135, 180)
- Use if you're missing bib numbers

**Filter out false positives (if getting too many wrong detections):**
```bash
racebib ocr --input ./photos --output results.csv --min-text-size 0.02 --min-conf 50
```
- Increase `--min-text-size` to filter tiny detections (e.g., from logos)
- Increase `--min-conf` to require higher confidence scores

**Improving detection accuracy:**
1. **Lower confidence threshold** if missing bibs: `--min-conf 40` or `--min-conf 50`
2. **Use aggressive mode**: `--aggressive` (tries more combinations)
3. **Check annotated images**: `--annotate-dir ./annotated` to see what OCR detects
4. **Try different PSM modes**: `--psm 8 13` (good for single numbers/words)
5. **Add more rotations**: `--rotations 0 45 90 -45 -90 180` if bibs are tilted

**Train OCR with known examples (recommended for best results):**
If you have some images where you know the correct bib numbers, you can "teach" the system what works best:

1. **Create a ground truth file** (JSON or CSV format):
   
   **JSON format** (`ground_truth.json`):
   ```json
   {
     "image1.jpg": ["254", "506", "133", "411"],
     "image2.jpg": ["1234"],
     "image3.jpg": []
   }
   ```
   
   **CSV format** (`ground_truth.csv`):
   ```csv
   image,bibs
   image1.jpg,"254;506;133;411"
   image2.jpg,"1234"
   image3.jpg,""
   ```

2. **Run training** to find optimal parameters:
   
   **Using Docker:**
   ```bash
   docker run --rm -v ${PWD}/data:/data \
     bibanalyser:latest \
     train --input /data/photos --ground-truth /data/ground_truth.json --output /data/best_params.json
   ```
   
   **Using Local Python:**
   ```bash
   racebib train --input ./photos --ground-truth ground_truth.json --output best_params.json
   ```
   
   **With GPU acceleration (if you have NVIDIA GPU):**
   ```bash
   racebib train --input ./photos --ground-truth ground_truth.json --output best_params.json --use-gpu
   ```
   
   **Note on GPU acceleration:**
   - GPU can speed up image preprocessing by 2-5x
   - However, Tesseract OCR is CPU-only, so OCR remains the bottleneck
   - Overall speedup is typically 20-40% depending on your system
   - Requires OpenCV with CUDA support or CuPy installed
   
   This will test hundreds of parameter combinations and find the best settings for your images.

3. **Use the best parameters** from training:
   
   **Using Docker:**
   ```bash
   docker run --rm -v ${PWD}/data:/data \
     bibanalyser:latest \
     ocr --input /data/photos --output /data/results.csv \
     --min-conf <best_value> \
     --psm <best_modes> \
     --rotations <best_angles> \
     --min-text-size <best_value> \
     --max-text-size <best_value>
   ```
   
   **Using Local Python:**
   ```bash
   racebib ocr --input ./photos --output results.csv \
     --min-conf <best_value> \
     --psm <best_modes> \
     --rotations <best_angles> \
     --min-text-size <best_value> \
     --max-text-size <best_value>
   ```

   The training output will show you the exact command to use!

**Step 3: Query for specific bib numbers**
```bash
racebib query --bib 1234 --db bibdb.json
```
- `--bib`: Bib number to search for (can specify multiple times)
- `--db`: Database file created in step 2
- `--format json` - Output as JSON instead of text

**Step 4: Generate HTML report (optional)**
```bash
python scripts/generate_report.py --db bibdb.json --output report.html --annotated-dir ./annotated
```

---

### YOLO + PaddleOCR Pipeline (Advanced / Experimental)

If the classic OCR approach (Tesseract + preprocessing) struggles, you can switch to a deep-learning pipeline that uses a YOLO detector for bib regions and PaddleOCR for text recognition.

**Requirements:**
- A YOLO model trained to detect bib numbers (`.pt` file). You can train with [Ultralytics YOLOv8](https://docs.ultralytics.com) on your labeled bib dataset.
- PaddleOCR for recognition (already included in Docker image after rebuild).

**Basic command (Docker):**
```bash
docker run --rm -v ${PWD}/data:/data \
  bibanalyser:latest \
  yolo --input /data/photos --output /data/yolo_results.csv \
  --weights /data/models/bib_yolo.pt \
  --db /data/bibdb.json \
  --image-url https://example.com/gallery/ \
  --annotate-dir /data/yolo_annotated
```

**Local Python:**
```bash
racebib yolo --input ./photos --output results/yolo_results.csv \
  --weights ./models/bib_yolo.pt \
  --db bibdb.json \
  --image-url https://example.com/gallery/ \
  --annotate-dir ./yolo_annotated
```

**Useful options:**
- `--conf` / `--iou`: adjust YOLO thresholds (defaults: 0.3 / 0.45)
- `--disable-ocr`: skip OCR to only get detection boxes
- `--ocr-lang`: choose PaddleOCR language model (default `en`)
- `--save-crops`: folder to store cropped bib regions
- `--use-gpu`: enable GPU for PaddleOCR (requires CUDA environment)

> ⚠️ The YOLO pipeline depends heavily on the quality of the detector. You must train or obtain a model that recognizes bib regions. The CLI assumes the bib class is ID `0`; override with `--class-id` if needed.

---

#### Quick Reference

**Using Docker (Windows-friendly, no Tesseract install needed):**
```powershell
# Build image (first time)
docker build -t bibanalyser:latest .

# Download images
docker run --rm -v ${PWD}/data:/data bibanalyser:latest download --url <gallery-url> --out /data/photos --filter "b*.jpg"

# Extract bibs
docker run --rm -v ${PWD}/data:/data bibanalyser:latest ocr --input /data/photos --output /data/results.csv --db /data/bibdb.json --image-url <gallery-base-url>/

# Train OCR parameters (optional, but recommended)
docker run --rm -v ${PWD}/data:/data bibanalyser:latest train --input /data/photos --ground-truth /data/ground_truth.json --output /data/best_params.json

# Query bibs
docker run --rm -v ${PWD}/data:/data bibanalyser:latest query --bib 1234 --db /data/bibdb.json

# YOLO + PaddleOCR (requires trained YOLO weights)
docker run --rm -v ${PWD}/data:/data bibanalyser:latest yolo --input /data/photos --output /data/yolo_results.csv --weights /data/models/bib_yolo.pt
```

**Using Local Python (requires Tesseract OCR installed):**
```bash
# Download images
racebib download --url <gallery-url> --out ./photos --filter "b*.jpg"

# Extract bibs (minimal)
racebib ocr --input ./photos --output results.csv

# Extract bibs (with database for querying)
racebib ocr --input ./photos --output results.csv \
  --db bibdb.json --image-url <gallery-base-url>/

# Query bibs
racebib query --bib 1234 --db bibdb.json

# Train OCR parameters (optional, but recommended)
racebib train --input ./photos --ground-truth ground_truth.json --output best_params.json

# YOLO + PaddleOCR (requires trained YOLO weights)
racebib yolo --input ./photos --output results/yolo_results.csv --weights ./models/bib_yolo.pt

# Generate report
python scripts/generate_report.py --db bibdb.json --output report.html
```

## Documentation

- [Installation Guide](docs/installation.md) - Docker and local setup
- [Usage Guide](docs/usage.md) - Command reference and examples
- [Troubleshooting](docs/troubleshooting.md) - Common issues and solutions

## Project Structure

- `scripts/` - Main application scripts
  - `gallery_downloader.py` - Download images from gallery pages
  - `bib_finder.py` - OCR pipeline for bib detection
  - `bib_yolo.py` - YOLO + PaddleOCR pipeline for advanced detection
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
