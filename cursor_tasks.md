# Cursor Task List

1) **Harden Downloader**
   - Add per-URL retry/backoff and 429/5xx handling.
   - Unit test URL transformation (`t__` → `b__`), relative/absolute URLs.
   - Optional: sitemap support or pagination hook.

2) **Improve OCR Accuracy**
   - Add `--bib-pattern` to accept custom regex (e.g., exact 4 digits).
   - Optional: add EAST/DB text detector to crop regions before OCR.
   - Evaluate PSM combinations and pick best automatically.

3) **Packaging**
   - Create a `pyproject.toml` for installable CLI `racebib` with subcommands `download` and `ocr`.
   - Add a Dockerfile with Tesseract + deps.
   - Add GitHub Actions for lint + tests.

4) **Usability**
   - Write `docs/` with examples and troubleshooting.
   - Add a simple HTML report summarizing `image → bibs` with thumbnails and links to annotated images.

5) **Performance**
   - Benchmark on 500 images; report throughput and memory.
   - Add `--limit` and `--sample` flags for quick trials.

Deliver artifacts back in this repo; do not change default behavior unless specified.
