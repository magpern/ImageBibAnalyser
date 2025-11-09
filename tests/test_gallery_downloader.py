"""
Unit tests for gallery_downloader.py
"""
import unittest
from pathlib import Path
from urllib.parse import urlsplit

import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

from gallery_downloader import thumb_to_full, sanitize_filename


class TestGalleryDownloader(unittest.TestCase):
    
    def test_thumb_to_full_basic(self):
        """Test basic t__ to b__ conversion."""
        thumb_url = "https://example.com/gallery/t__12345.jpg"
        expected = "https://example.com/gallery/b__12345.jpg"
        result = thumb_to_full(thumb_url)
        self.assertEqual(result, expected)
    
    def test_thumb_to_full_no_t_prefix(self):
        """Test URL without t__ prefix remains unchanged."""
        url = "https://example.com/gallery/image12345.jpg"
        result = thumb_to_full(url)
        self.assertEqual(result, url)
    
    def test_thumb_to_full_relative_path(self):
        """Test relative path conversion."""
        thumb_url = "/gallery/t__12345.jpg"
        expected = "/gallery/b__12345.jpg"
        result = thumb_to_full(thumb_url)
        self.assertEqual(result, expected)
    
    def test_thumb_to_full_with_query(self):
        """Test URL with query string."""
        thumb_url = "https://example.com/gallery/t__12345.jpg?size=large"
        expected = "https://example.com/gallery/b__12345.jpg?size=large"
        result = thumb_to_full(thumb_url)
        self.assertEqual(result, expected)
    
    def test_thumb_to_full_with_fragment(self):
        """Test URL with fragment."""
        thumb_url = "https://example.com/gallery/t__12345.jpg#section"
        expected = "https://example.com/gallery/b__12345.jpg#section"
        result = thumb_to_full(thumb_url)
        self.assertEqual(result, expected)
    
    def test_thumb_to_full_no_path(self):
        """Test URL with no path."""
        url = "https://example.com"
        result = thumb_to_full(url)
        self.assertEqual(url, result)
    
    def test_thumb_to_full_multiple_t_prefix(self):
        """Test URL with multiple t__ occurrences (only first should be replaced)."""
        thumb_url = "https://example.com/t__12345_t__67890.jpg"
        expected = "https://example.com/b__12345_t__67890.jpg"
        result = thumb_to_full(thumb_url)
        self.assertEqual(result, expected)
    
    def test_sanitize_filename_basic(self):
        """Test basic filename sanitization."""
        name = "image 123.jpg"
        result = sanitize_filename(name)
        self.assertEqual(result, "image_123.jpg")
    
    def test_sanitize_filename_special_chars(self):
        """Test filename with special characters."""
        name = "image@#$%^&*().jpg"
        result = sanitize_filename(name)
        self.assertEqual(result, "image_________.jpg")
    
    def test_sanitize_filename_valid_chars(self):
        """Test filename with valid characters remains unchanged."""
        name = "image-123_456.jpg"
        result = sanitize_filename(name)
        self.assertEqual(result, name)
    
    def test_urljoin_absolute_url(self):
        """Test that absolute URLs are handled correctly."""
        from urllib.parse import urljoin
        base = "https://example.com/gallery/"
        absolute = "https://other.com/image.jpg"
        result = urljoin(base, absolute)
        self.assertEqual(result, absolute)  # Absolute URL should override base
    
    def test_urljoin_relative_url(self):
        """Test that relative URLs are joined with base."""
        from urllib.parse import urljoin
        base = "https://example.com/gallery/"
        relative = "image.jpg"
        result = urljoin(base, relative)
        self.assertEqual(result, "https://example.com/gallery/image.jpg")


if __name__ == "__main__":
    unittest.main()

