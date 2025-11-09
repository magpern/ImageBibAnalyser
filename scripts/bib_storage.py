"""
Bib Storage Module — stores and queries bib-to-URL mappings.

Provides functions to store detected bib numbers linked to image URLs,
with support for multiple bibs per image, confidence scores, and timestamps.
Uses JSON file format for simplicity and portability.
"""

import json
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Set


class BibStorage:
    """Storage backend for bib-to-URL mappings using JSON file."""

    def __init__(self, db_path: Path):
        """Initialize storage with database file path.

        Args:
            db_path: Path to JSON database file
        """
        self.db_path = Path(db_path)
        self._data: Dict[str, Dict] = {}
        self._load()

    def _load(self):
        """Load data from JSON file if it exists."""
        if self.db_path.exists():
            try:
                with open(self.db_path, "r", encoding="utf-8") as f:
                    self._data = json.load(f)
            except (json.JSONDecodeError, IOError) as e:
                print(f"Warning: Could not load database from {self.db_path}: {e}")
                self._data = {}
        else:
            self._data = {}

    def _save(self):
        """Save data to JSON file."""
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.db_path, "w", encoding="utf-8") as f:
            json.dump(self._data, f, indent=2, ensure_ascii=False)

    def store_detection(
        self,
        image_url: str,
        bibs: List[str],
        confidences: Optional[List[float]] = None,
        timestamp: Optional[float] = None,
        local_path: Optional[str] = None,
    ):
        """Store a detection result.

        Args:
            image_url: Original image URL (primary identifier)
            bibs: List of detected bib numbers
            confidences: Optional list of confidence scores (one per bib)
            timestamp: Optional timestamp (defaults to current time)
            local_path: Optional local file path if image was downloaded
        """
        if timestamp is None:
            timestamp = time.time()

        # Create confidence dict: bib -> confidence
        bib_confidences: Dict[str, float] = {}
        if confidences and len(confidences) == len(bibs):
            for bib, conf in zip(bibs, confidences):
                bib_confidences[bib] = conf
        else:
            # Default confidence if not provided
            for bib in bibs:
                bib_confidences[bib] = 0.0

        # Store or update entry
        entry = {
            "image_url": image_url,
            "bibs": bibs,
            "confidences": bib_confidences,
            "timestamp": timestamp,
            "local_path": local_path,
        }

        self._data[image_url] = entry
        self._save()

    def get_urls_by_bib(
        self,
        bib_number: str,
        min_confidence: Optional[float] = None,
    ) -> List[str]:
        """Get all image URLs containing a specific bib number.

        Args:
            bib_number: Bib number to search for
            min_confidence: Optional minimum confidence threshold

        Returns:
            List of image URLs containing the bib
        """
        urls = []
        for image_url, entry in self._data.items():
            if bib_number in entry.get("bibs", []):
                if min_confidence is not None:
                    confidences = entry.get("confidences", {})
                    if bib_number in confidences:
                        if confidences[bib_number] < min_confidence:
                            continue
                urls.append(image_url)
        return sorted(urls)

    def get_bibs_by_url(self, image_url: str) -> List[str]:
        """Get all bib numbers detected in a specific image URL.

        Args:
            image_url: Image URL to query

        Returns:
            List of bib numbers found in the image
        """
        entry = self._data.get(image_url)
        if entry:
            return entry.get("bibs", [])
        return []

    def get_entry(self, image_url: str) -> Optional[Dict]:
        """Get full entry for an image URL.

        Args:
            image_url: Image URL to query

        Returns:
            Entry dict with all metadata, or None if not found
        """
        return self._data.get(image_url)

    def query_multiple_bibs(
        self,
        bib_list: List[str],
        require_all: bool = False,
        min_confidence: Optional[float] = None,
    ) -> List[str]:
        """Find URLs containing any or all of the specified bibs.

        Args:
            bib_list: List of bib numbers to search for
            require_all: If True, require all bibs to be present (default: any)
            min_confidence: Optional minimum confidence threshold

        Returns:
            List of image URLs matching the criteria
        """
        if not bib_list:
            return []

        matching_urls: Set[str] = set()

        for image_url, entry in self._data.items():
            entry_bibs = set(entry.get("bibs", []))
            query_bibs = set(bib_list)

            if require_all:
                # All bibs must be present
                if not query_bibs.issubset(entry_bibs):
                    continue
            else:
                # At least one bib must be present
                if not query_bibs.intersection(entry_bibs):
                    continue

            # Check confidence threshold if specified
            if min_confidence is not None:
                confidences = entry.get("confidences", {})
                matching_bibs = query_bibs.intersection(entry_bibs)
                if any(confidences.get(bib, 0.0) < min_confidence for bib in matching_bibs):
                    continue

            matching_urls.add(image_url)

        return sorted(matching_urls)

    def get_all_urls(self) -> List[str]:
        """Get all image URLs in the database.

        Returns:
            List of all image URLs
        """
        return sorted(self._data.keys())

    def get_stats(self) -> Dict:
        """Get statistics about the database.

        Returns:
            Dict with statistics (total_images, total_bibs, unique_bibs, etc.)
        """
        all_bibs: Set[str] = set()
        total_detections = 0

        for entry in self._data.values():
            bibs = entry.get("bibs", [])
            all_bibs.update(bibs)
            total_detections += len(bibs)

        return {
            "total_images": len(self._data),
            "total_detections": total_detections,
            "unique_bibs": len(all_bibs),
            "avg_bibs_per_image": total_detections / len(self._data) if self._data else 0.0,
        }
