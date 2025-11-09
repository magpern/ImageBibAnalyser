# Functional Specs & Acceptance Criteria

## Gallery Downloader
- Input: `--url`, `--out`, optional `--selector` (default `.galleria-thumbnails img`), `--concurrency`.
- Operation: parse HTML, find `<img>` elements, read `src`/`data-src`/`data-original`.
- Transform: change filename prefix `t__` → `b__`. If no `t__`, keep original URL.
- Use absolute URLs via `urljoin` relative to page.
- Download concurrently with retries; skip duplicates; never clobber existing files.
- Output: images saved into `--out` with stable names; log counts success/fail.

## Bib Finder
- Input: `--input` folder, `--output` CSV, optional `--annotate-dir`.
- Preprocess: grayscale, CLAHE, bilateral filter, adaptive threshold.
- OCR: Tesseract with PSM set {6,7,11} over rotations {0, 90, -90, 180}.
- Filter: numbers with N digits where 2 ≤ N ≤ 6 (configurable), minimum confidence ≥ 60 (configurable).
- Output: CSV with columns `image,bibs` (semicolon-separated); JSON mapping next to CSV.
- Optional: write annotated previews with bounding boxes and confidence labels.
- Performance: should handle 500–2000 images comfortably on a typical laptop.

## General
- Provide clear error messages and non-zero exit codes on failure.
- Include docstrings and inline comments where logic is non-trivial.
- Keep third-party deps minimal and standard.
