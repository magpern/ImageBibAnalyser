# Cursor Rules

- Prefer clear, commented Python.
- Avoid breaking public CLI flags.
- Keep dependencies limited to: requests, beautifulsoup4, tqdm, opencv-python, pytesseract, numpy, pandas.
- Write small, testable functions; avoid global state.
- For OCR improvements, add flags rather than changing defaults.
- On Windows, allow passing `--tesseract-cmd` to specify Tesseract path.
- When refactoring, maintain backward compatibility or add a migration note.
