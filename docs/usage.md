# Usage Guide

## Docker Usage

### Basic Commands

All commands run inside the Docker container. Use Docker volumes to mount your data directories.

**Note:** The Docker image uses `python:3.11-slim` as the base image and `opencv-python-headless` for a lightweight build without GUI dependencies.

#### Download Images from Gallery

```bash
docker run --rm -v $(pwd)/data:/data \
  ghcr.io/OWNER/bibanalyser:latest \
  download --url https://example.com/gallery --out /data/photos
```

#### Extract Bib Numbers from Images

```bash
docker run --rm -v $(pwd)/data:/data \
  ghcr.io/OWNER/bibanalyser:latest \
  ocr --input /data/photos --output /data/results.csv \
  --db /data/bibdb.json --image-url https://example.com/gallery/
```

#### Query for Bib Numbers

```bash
docker run --rm -v $(pwd)/data:/data \
  ghcr.io/OWNER/bibanalyser:latest \
  query --bib 1234 --db /data/bibdb.json
```

### Windows PowerShell Examples

```powershell
# Download images
docker run --rm -v ${PWD}/data:/data `
  ghcr.io/OWNER/bibanalyser:latest `
  download --url https://example.com/gallery --out /data/photos

# Extract bibs
docker run --rm -v ${PWD}/data:/data `
  ghcr.io/OWNER/bibanalyser:latest `
  ocr --input /data/photos --output /data/results.csv `
  --db /data/bibdb.json --image-url https://example.com/gallery/

# Query bibs
docker run --rm -v ${PWD}/data:/data `
  ghcr.io/OWNER/bibanalyser:latest `
  query --bib 1234 --db /data/bibdb.json
```

## Local Development Usage

### Download Command

Download images from a gallery page:

```bash
racebib download \
  --url https://www.hasselbyloppet.se/gallery \
  --out ./photos \
  --selector '.galleria-thumbnails img' \
  --concurrency 8
```

**Options:**
- `--url`: Gallery page URL (required)
- `--out`: Output directory for downloaded images (required)
- `--selector`: CSS selector for image elements (default: `.galleria-thumbnails img`)
- `--concurrency`: Number of parallel downloads (default: 8)

### OCR Command

Extract bib numbers from images:

```bash
racebib ocr \
  --input ./photos \
  --output results/results.csv \
  --db results/bibdb.json \
  --image-url https://example.com/gallery/ \
  --annotate-dir results/annotated \
  --workers 4 \
  --min-digits 2 \
  --max-digits 6 \
  --min-conf 60
```

**Options:**
- `--input`: Input directory with images (required)
- `--output`: Output CSV file path (required)
- `--db`: Path to bib storage database (optional, enables querying)
- `--image-url`: Base URL for constructing full image URLs (required if using --db)
- `--annotate-dir`: Directory to save annotated previews (optional)
- `--workers`: Number of parallel workers (default: 4)
- `--min-digits`: Minimum bib digits (default: 2)
- `--max-digits`: Maximum bib digits (default: 6)
- `--min-conf`: Minimum OCR confidence 0-100 (default: 60)
- `--bib-pattern`: Custom regex pattern for bib numbers (overrides min/max digits)
- `--limit`: Process only first N images
- `--sample`: Randomly sample N images
- `--ext`: Image extensions to include (default: .jpg .jpeg .png)
- `--psm`: Tesseract PSM modes to try (default: 6 7 11)
- `--rotations`: Image rotations to try (default: 0 90 -90 180)

### Query Command

Query the database for image URLs containing specific bib numbers:

```bash
racebib query \
  --bib 1234 \
  --db results/bibdb.json \
  --format json
```

**Options:**
- `--bib`: Bib number to search for (can be specified multiple times, required)
- `--db`: Path to bib storage database (required)
- `--format`: Output format: text, json, csv (default: text)
- `--min-confidence`: Minimum confidence threshold (0-100)
- `--all`: Require all specified bibs to be present (default: any)

**Examples:**

```bash
# Find URLs with bib 1234
racebib query --bib 1234 --db bibdb.json

# Find URLs with bib 1234 OR 5678
racebib query --bib 1234 --bib 5678 --db bibdb.json

# Find URLs with bib 1234 AND 5678
racebib query --bib 1234 --bib 5678 --db bibdb.json --all

# JSON output with confidence filter
racebib query --bib 1234 --db bibdb.json --format json --min-confidence 70
```

## Complete Workflow Example

1. **Download images:**
   ```bash
   racebib download --url https://example.com/gallery --out ./photos
   ```

2. **Extract bib numbers and store in database:**
   ```bash
   racebib ocr \
     --input ./photos \
     --output results.csv \
     --db bibdb.json \
     --image-url https://example.com/gallery/ \
     --annotate-dir annotated
   ```

3. **Query for specific bib:**
   ```bash
   racebib query --bib 1234 --db bibdb.json --format json
   ```

## Output Formats

### CSV Output (OCR)

The OCR command generates a CSV file with columns:
- `image`: Path to image file
- `bibs`: Semicolon-separated list of detected bib numbers

### JSON Output (OCR)

Also generates a JSON file mapping image paths to bib arrays:
```json
{
  "image1.jpg": ["1234", "5678"],
  "image2.jpg": ["9012"]
}
```

### Database Format

The database (JSON) stores:
- Image URLs
- Detected bib numbers with confidence scores
- Timestamps
- Local file paths

### Query Output

Query command outputs image URLs in the specified format:
- **text**: One URL per line
- **json**: JSON array of URLs
- **csv**: CSV with `url` column

