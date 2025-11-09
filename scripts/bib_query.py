"""
Bib Query — query the bib storage database to find image URLs containing specific bib numbers.

Usage:
  python bib_query.py --bib 1234 --db bibdb.json
  python bib_query.py --bib 1234 --bib 5678 --db bibdb.json --format json
  python bib_query.py --bib 1234 --bib 5678 --db bibdb.json --all --min-confidence 70
"""
import argparse
import json
import sys
from pathlib import Path
from typing import List

try:
    from bib_storage import BibStorage
except ImportError:
    # Handle import when running as module
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    from bib_storage import BibStorage


def output_text(urls: List[str]):
    """Output URLs as plain text, one per line."""
    for url in urls:
        print(url)


def output_json(urls: List[str]):
    """Output URLs as JSON array."""
    print(json.dumps(urls, indent=2, ensure_ascii=False))


def output_csv(urls: List[str]):
    """Output URLs as CSV."""
    print("url")
    for url in urls:
        print(url)


def main():
    ap = argparse.ArgumentParser(
        description="Query bib storage database to find image URLs containing specific bib numbers"
    )
    ap.add_argument(
        "--bib",
        action="append",
        required=True,
        help="Bib number to search for (can be specified multiple times)"
    )
    ap.add_argument(
        "--db",
        required=True,
        type=Path,
        help="Path to bib storage database (JSON file)"
    )
    ap.add_argument(
        "--format",
        choices=["text", "json", "csv"],
        default="text",
        help="Output format (default: text)"
    )
    ap.add_argument(
        "--min-confidence",
        type=float,
        default=None,
        help="Minimum confidence threshold (0-100)"
    )
    ap.add_argument(
        "--all",
        action="store_true",
        help="Require all specified bibs to be present (default: any)"
    )

    args = ap.parse_args()

    if not args.db.exists():
        print(f"Error: Database file not found: {args.db}", file=sys.stderr)
        sys.exit(1)

    storage = BibStorage(args.db)

    # Query for bibs
    if len(args.bib) == 1:
        # Single bib query
        urls = storage.get_urls_by_bib(
            args.bib[0],
            min_confidence=args.min_confidence,
        )
    else:
        # Multiple bib query
        urls = storage.query_multiple_bibs(
            args.bib,
            require_all=args.all,
            min_confidence=args.min_confidence,
        )

    # Output results
    if args.format == "json":
        output_json(urls)
    elif args.format == "csv":
        output_csv(urls)
    else:
        output_text(urls)

    # Exit with error code if no results
    if not urls:
        sys.exit(1)


if __name__ == "__main__":
    main()

