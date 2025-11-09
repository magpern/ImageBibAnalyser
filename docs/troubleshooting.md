# Troubleshooting Guide

## Common Issues and Solutions

### OCR Issues

### No Bibs Detected

**Symptoms:** OCR runs but finds no bib numbers in images.

**Solutions:**
1. **Check image quality**: Ensure images are clear and bibs are visible
2. **Adjust confidence threshold**: Lower `--min-conf` (default 60):
   ```bash
   racebib ocr --input ./photos --output results.csv --min-conf 40
   ```
3. **Try different PSM modes**: Some images work better with different modes:
   ```bash
   racebib ocr --input ./photos --output results.csv --psm 7 8 11
   ```
4. **Adjust digit range**: If bibs have specific digit counts:
   ```bash
   racebib ocr --input ./photos --output results.csv --min-digits 3 --max-digits 5
   ```
5. **Use custom pattern**: For specific bib formats:
   ```bash
   racebib ocr --input ./photos --output results.csv --bib-pattern "\d{4}"
   ```

### Low Detection Accuracy

**Symptoms:** Many false positives or missed bibs.

**Solutions:**
1. **Increase confidence threshold**: Filter out low-confidence detections:
   ```bash
   racebib ocr --input ./photos --output results.csv --min-conf 70
   ```
2. **Use annotated previews**: Check what OCR is detecting:
   ```bash
   racebib ocr --input ./photos --output results.csv --annotate-dir annotated
   ```
3. **Adjust preprocessing**: The current preprocessing uses CLAHE and adaptive thresholding. For different image types, you may need to modify the preprocessing in `bib_finder.py`.

### Download Issues

### 429 Rate Limited

**Symptoms:** Downloads fail with 429 errors.

**Solutions:**
1. **Reduce concurrency**: Lower the number of parallel downloads:
   ```bash
   racebib download --url https://example.com/gallery --out ./photos --concurrency 2
   ```
2. **Wait and retry**: The downloader automatically retries with exponential backoff. Wait a few minutes and try again.

### 404 Not Found

**Symptoms:** Some images return 404 errors.

**Solutions:**
1. **Check URL transformation**: The downloader converts `t__*.jpg` to `b__*.jpg`. Verify this matches the gallery's URL pattern.
2. **Check selector**: Ensure the CSS selector matches the gallery structure:
   ```bash
   racebib download --url https://example.com/gallery --out ./photos --selector "img.thumbnail"
   ```

### Connection Timeouts

**Symptoms:** Downloads timeout or fail to connect.

**Solutions:**
1. **Check internet connection**
2. **Increase timeout** (modify code or wait for future feature)
3. **Reduce concurrency** to avoid overwhelming the server

### Database/Storage Issues

### Database File Not Found

**Symptoms:** Query command fails with "Database file not found".

**Solutions:**
1. **Ensure database was created**: Run OCR with `--db` parameter first:
   ```bash
   racebib ocr --input ./photos --output results.csv --db bibdb.json --image-url https://example.com/
   ```
2. **Check file path**: Use absolute paths or ensure you're in the correct directory.

### No Results from Query

**Symptoms:** Query returns no URLs even though bibs were detected.

**Solutions:**
1. **Check bib number format**: Ensure you're querying with the exact format detected (e.g., "0123" vs "123")
2. **Check confidence threshold**: If using `--min-confidence`, ensure it's not too high
3. **Verify database contents**: Check the JSON file to see what's stored:
   ```bash
   cat bibdb.json
   ```

### Docker Issues

### Permission Denied

**Symptoms:** Docker can't write to mounted volumes.

**Solutions:**
1. **Check volume permissions**: Ensure the host directory is writable
2. **Use absolute paths**: Use full paths for volumes:
   ```bash
   docker run --rm -v /full/path/to/data:/data ...
   ```

### Tesseract Not Found in Container

**Symptoms:** OCR fails with "Tesseract not found" in Docker.

**Solutions:**
1. **Rebuild image**: Ensure Dockerfile includes Tesseract installation
2. **Check image**: Verify Tesseract is installed:
   ```bash
   docker run --rm ghcr.io/OWNER/bibanalyser:latest tesseract --version
   ```

### Performance Issues

### Slow Processing

**Symptoms:** OCR takes a very long time.

**Solutions:**
1. **Increase workers**: Use more parallel workers:
   ```bash
   racebib ocr --input ./photos --output results.csv --workers 8
   ```
2. **Limit images for testing**: Use `--limit` or `--sample`:
   ```bash
   racebib ocr --input ./photos --output results.csv --limit 10
   ```
3. **Reduce rotations/PSM modes**: Fewer combinations = faster:
   ```bash
   racebib ocr --input ./photos --output results.csv --psm 6 --rotations 0
   ```

### High Memory Usage

**Symptoms:** Process uses too much memory.

**Solutions:**
1. **Reduce workers**: Lower parallelism:
   ```bash
   racebib ocr --input ./photos --output results.csv --workers 2
   ```
2. **Process in batches**: Use `--limit` to process images in smaller batches

### General Issues

### Import Errors

**Symptoms:** "ModuleNotFoundError" or import errors.

**Solutions:**
1. **Install dependencies**: `pip install -r requirements.txt`
2. **Use CLI entry points**: Use `racebib` commands instead of running scripts directly
3. **Check PYTHONPATH**: Ensure scripts directory is in Python path

### Script Not Executable

**Symptoms:** "Permission denied" when running scripts.

**Solutions:**
1. **Use Python explicitly**: `python3 scripts/bib_finder.py ...`
2. **Use CLI**: `racebib ocr ...`
3. **Make executable** (Linux/Mac): `chmod +x scripts/*.py`

## Getting Help

If you encounter issues not covered here:

1. Check the error messages - they often contain helpful information
2. Review the annotated images (if using `--annotate-dir`) to see what OCR is detecting
3. Check the database JSON file to verify data is being stored correctly
4. Try with a small sample (`--limit 5`) to isolate issues
5. Review the logs and error output for specific error codes or messages

