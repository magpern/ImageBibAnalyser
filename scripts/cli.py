"""
Race Bib Analyzer CLI — main entry point for racebib commands.
"""

import sys
from pathlib import Path

# Add scripts directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))


def main():
    """Main CLI entry point that routes to subcommands."""
    if len(sys.argv) < 2:
        print("Usage: racebib <command> [options]")
        print("\nCommands:")
        print("  download  Download images from a gallery page")
        print("  ocr       Extract bib numbers from images using OCR")
        print("  query     Query database for image URLs containing specific bib numbers")
        print("  train     Train/validate OCR parameters using images with known bib numbers")
        print("  train-yolo Train YOLO model for bib detection")
        print("  yolo      Run YOLO-based detector with optional OCR (advanced pipeline)")
        print("\nUse 'racebib <command> --help' for command-specific help")
        sys.exit(1)

    command = sys.argv[1]
    sys.argv = sys.argv[1:]  # Remove 'racebib' from argv

    if command == "download":
        from gallery_downloader import main as download_main

        download_main()
    elif command == "ocr":
        from bib_finder import main as ocr_main

        ocr_main()
    elif command == "query":
        from bib_query import main as query_main

        query_main()
    elif command == "train":
        from bib_train import main as train_main

        train_main()
    elif command == "train-yolo":
        from train_yolo import main as train_yolo_main

        train_yolo_main()
    elif command == "yolo":
        from bib_yolo import main as yolo_main

        yolo_main()
    else:
        print(f"Unknown command: {command}")
        print("Available commands: download, ocr, query, train, train-yolo, yolo")
        sys.exit(1)


if __name__ == "__main__":
    main()
