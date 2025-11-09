# VS Code Setup Guide

## Initial Setup

1. **Create Virtual Environment:**
   - Run the setup script:
     - Windows: `.\setup_venv.ps1`
     - Linux/macOS: `./setup_venv.sh`
   - Or manually:
     ```bash
     python -m venv .venv
     .venv\Scripts\activate  # Windows
     source .venv/bin/activate  # Linux/macOS
     pip install -r requirements.txt
     pip install -e .
     ```

2. **Select Python Interpreter in VS Code:**
   - Press `Ctrl+Shift+P` (or `Cmd+Shift+P` on Mac)
   - Type "Python: Select Interpreter"
   - Choose `.venv\Scripts\python.exe` (Windows) or `.venv/bin/python` (Linux/macOS)

## Debugging (F5)

Press `F5` to start debugging. You'll see these configurations:

- **Python: Download Images** - Run gallery downloader
- **Python: Extract Bibs (OCR)** - Run bib finder
- **Python: Query Bibs** - Query bib database
- **Python: Current File** - Run the currently open Python file

You can modify the arguments in `.vscode/launch.json` to customize the commands.

## Running Scripts

You can also run scripts directly:

1. Open a terminal in VS Code (`Ctrl+`` ` or `View > Terminal`)
2. Activate venv (if not auto-activated):
   ```bash
   .venv\Scripts\activate  # Windows
   source .venv/bin/activate  # Linux/macOS
   ```
3. Run scripts:
   ```bash
   python scripts/gallery_downloader.py --url https://example.com --out ./photos
   python scripts/bib_finder.py --input ./photos --output results.csv
   ```

## Troubleshooting

- **F5 doesn't work**: Make sure you've selected the Python interpreter (see step 2 above)
- **Import errors**: Ensure `PYTHONPATH` includes the scripts directory (already configured in launch.json)
- **Module not found**: Make sure venv is activated and dependencies are installed

