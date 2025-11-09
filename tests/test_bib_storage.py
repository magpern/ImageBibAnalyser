"""
Unit tests for bib_storage.py
"""
import json
import tempfile
import unittest
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

from bib_storage import BibStorage


class TestBibStorage(unittest.TestCase):
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.db_path = Path(self.temp_dir) / "test_db.json"
        self.storage = BibStorage(self.db_path)
    
    def tearDown(self):
        """Clean up test fixtures."""
        if self.db_path.exists():
            self.db_path.unlink()
    
    def test_store_and_retrieve_single_bib(self):
        """Test storing and retrieving a single bib."""
        self.storage.store_detection(
            image_url="https://example.com/image1.jpg",
            bibs=["1234"],
            confidences=[85.5],
        )
        
        urls = self.storage.get_urls_by_bib("1234")
        self.assertEqual(len(urls), 1)
        self.assertIn("https://example.com/image1.jpg", urls)
    
    def test_store_multiple_bibs(self):
        """Test storing multiple bibs for one image."""
        self.storage.store_detection(
            image_url="https://example.com/image1.jpg",
            bibs=["1234", "5678"],
            confidences=[85.5, 90.0],
        )
        
        bibs = self.storage.get_bibs_by_url("https://example.com/image1.jpg")
        self.assertEqual(len(bibs), 2)
        self.assertIn("1234", bibs)
        self.assertIn("5678", bibs)
    
    def test_query_multiple_bibs_any(self):
        """Test querying with multiple bibs (any match)."""
        self.storage.store_detection(
            image_url="https://example.com/image1.jpg",
            bibs=["1234"],
        )
        self.storage.store_detection(
            image_url="https://example.com/image2.jpg",
            bibs=["5678"],
        )
        self.storage.store_detection(
            image_url="https://example.com/image3.jpg",
            bibs=["1234", "5678"],
        )
        
        urls = self.storage.query_multiple_bibs(["1234", "5678"], require_all=False)
        self.assertEqual(len(urls), 3)  # All three images match
    
    def test_query_multiple_bibs_all(self):
        """Test querying with multiple bibs (all required)."""
        self.storage.store_detection(
            image_url="https://example.com/image1.jpg",
            bibs=["1234"],
        )
        self.storage.store_detection(
            image_url="https://example.com/image2.jpg",
            bibs=["1234", "5678"],
        )
        
        urls = self.storage.query_multiple_bibs(["1234", "5678"], require_all=True)
        self.assertEqual(len(urls), 1)
        self.assertIn("https://example.com/image2.jpg", urls)
    
    def test_confidence_filtering(self):
        """Test filtering by confidence threshold."""
        self.storage.store_detection(
            image_url="https://example.com/image1.jpg",
            bibs=["1234"],
            confidences=[50.0],
        )
        self.storage.store_detection(
            image_url="https://example.com/image2.jpg",
            bibs=["1234"],
            confidences=[80.0],
        )
        
        urls_low = self.storage.get_urls_by_bib("1234", min_confidence=40.0)
        self.assertEqual(len(urls_low), 2)
        
        urls_high = self.storage.get_urls_by_bib("1234", min_confidence=70.0)
        self.assertEqual(len(urls_high), 1)
        self.assertIn("https://example.com/image2.jpg", urls_high)
    
    def test_persistence(self):
        """Test that data persists across storage instances."""
        self.storage.store_detection(
            image_url="https://example.com/image1.jpg",
            bibs=["1234"],
        )
        
        # Create new storage instance
        storage2 = BibStorage(self.db_path)
        urls = storage2.get_urls_by_bib("1234")
        self.assertEqual(len(urls), 1)
    
    def test_get_stats(self):
        """Test statistics generation."""
        self.storage.store_detection(
            image_url="https://example.com/image1.jpg",
            bibs=["1234", "5678"],
        )
        self.storage.store_detection(
            image_url="https://example.com/image2.jpg",
            bibs=["9012"],
        )
        
        stats = self.storage.get_stats()
        self.assertEqual(stats['total_images'], 2)
        self.assertEqual(stats['total_detections'], 3)
        self.assertEqual(stats['unique_bibs'], 3)
        self.assertEqual(stats['avg_bibs_per_image'], 1.5)
    
    def test_get_entry(self):
        """Test retrieving full entry."""
        self.storage.store_detection(
            image_url="https://example.com/image1.jpg",
            bibs=["1234"],
            confidences=[85.5],
            local_path="/local/image1.jpg",
        )
        
        entry = self.storage.get_entry("https://example.com/image1.jpg")
        self.assertIsNotNone(entry)
        self.assertEqual(entry['image_url'], "https://example.com/image1.jpg")
        self.assertEqual(entry['bibs'], ["1234"])
        self.assertEqual(entry['confidences']['1234'], 85.5)
        self.assertEqual(entry['local_path'], "/local/image1.jpg")


if __name__ == "__main__":
    unittest.main()

