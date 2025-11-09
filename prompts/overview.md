# Project Overview (for Cursor Agents)

**Goal:** Automate downloading race photos and identifying runner bib numbers to produce a mapping of `image → bibs`.

**Key components:**
- Robust page scraper for thumbnail → full image URL conversion pattern (`t__*.jpg` → `b__*.jpg`).
- Efficient downloader with retries, deduping, and parallelism.
- OCR pipeline using Tesseract + OpenCV preprocessing (CLAHE, bilateral filter, adaptive threshold), rotation trials, and regex filtering.
- CSV + JSON outputs; optional annotated images with bounding boxes for detected bib numbers.

**Stretch goals:**
- Add a detector (e.g., YOLO/DB/CRNN) for bib regions to pre-filter OCR.
- Web UI for browsing results and correcting misreads.
- Dockerfile and CI tests.
