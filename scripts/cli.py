"""
Race Bib Analyzer CLI — main entry point for racebib commands.
"""

import sys
from pathlib import Path


def main():
    """Main CLI entry point that routes to subcommands."""
    if len(sys.argv) < 2:
        print("Usage: racebib <command> [options]")
        print("\nCommands:")
        print("  download  Download images from a gallery page")
        print("  ocr       Extract bib numbers from images using OCR")
        print("  query     Query database for image URLs containing specific bib numbers")
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
    else:
        print(f"Unknown command: {command}")
        print("Available commands: download, ocr, query")
        sys.exit(1)


if __name__ == "__main__":
    main()
