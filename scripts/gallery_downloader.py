""" 
Gallery Image Downloader — given a page URL and an output folder, parse the page,
find thumbnail <img> tags in a list container (e.g., .galleria-thumbnails), 
transform each thumbnail URL to the full image URL (t__*.jpg -> b__*.jpg),
and download all originals to disk.

Requirements:
  - Python 3.9+
  - pip install requests beautifulsoup4 tqdm

Usage:
  python gallery_downloader.py --url https://www.hasselbyloppet.se/some/page \
      --out ./photos --selector '.galleria-thumbnails img' --concurrency 8

Notes:
  - By default, converts any filename that starts with 't__' to 'b__'.
  - Works with absolute and relative <img src> URLs.
  - Skips duplicates and supports simple retries.
"""
import argparse
import hashlib
import os
import random
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import requests
from bs4 import BeautifulSoup
from requests.adapters import HTTPAdapter
try:
    from requests.packages.urllib3.util.retry import Retry
except ImportError:
    from urllib3.util.retry import Retry
from urllib.parse import urljoin, urlsplit, urlunsplit
from tqdm import tqdm

def thumb_to_full(url: str) -> str:
    parts = urlsplit(url)
    if not parts.path:
        return url
    if "/" in parts.path:
        dirpath, filename = parts.path.rsplit("/", 1)
    else:
        dirpath, filename = "", parts.path
    new_filename = re.sub(r"^t__", "b__", filename)
    new_path = f"{dirpath}/{new_filename}" if dirpath else new_filename
    return urlunsplit((parts.scheme, parts.netloc, new_path, parts.query, parts.fragment))

def sanitize_filename(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]", "_", name)

def make_session() -> requests.Session:
    """Create a requests session with retry strategy and proper headers."""
    s = requests.Session()
    
    # Configure retry strategy with exponential backoff
    retry_strategy = Retry(
        total=5,
        backoff_factor=1,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET", "HEAD"],
    )
    
    adapter = HTTPAdapter(
        max_retries=retry_strategy,
        pool_connections=20,
        pool_maxsize=20
    )
    s.mount("http://", adapter)
    s.mount("https://", adapter)
    s.headers.update({
        "User-Agent": "gallery-downloader/1.0 (+https://example.local)"
    })
    return s

@dataclass
class DownloadItem:
    thumb_url: str
    full_url: str

def discover_images(page_url: str, selector: str, session: requests.Session) -> List[DownloadItem]:
    """Discover images from a gallery page.
    
    Args:
        page_url: URL of the gallery page
        selector: CSS selector to find image elements
        session: Requests session
        
    Returns:
        List of DownloadItem objects
        
    Raises:
        requests.RequestException: If page cannot be fetched
    """
    try:
        r = session.get(page_url, timeout=20)
        r.raise_for_status()
    except requests.exceptions.RequestException as e:
        raise RuntimeError(f"Failed to fetch gallery page {page_url}: {e}") from e
    
    soup = BeautifulSoup(r.text, "html.parser")
    elements = soup.select(selector)

    items: List[DownloadItem] = []
    for el in elements:
        src = el.get("src") or el.get("data-src") or el.get("data-original")
        if not src:
            continue
        abs_thumb = urljoin(page_url, src)
        full = thumb_to_full(abs_thumb)
        items.append(DownloadItem(thumb_url=abs_thumb, full_url=full))

    seen = set()
    unique: List[DownloadItem] = []
    for it in items:
        if it.full_url not in seen:
            seen.add(it.full_url)
            unique.append(it)
    return unique

def choose_target_path(out_dir: Path, url: str) -> Path:
    filename = urlsplit(url).path.rsplit("/", 1)[-1] or "image"
    filename = sanitize_filename(filename)
    path = out_dir / filename
    if not path.exists():
        return path
    stem = path.stem
    suffix = path.suffix
    i = 1
    while True:
        candidate = out_dir / f"{stem}_{i}{suffix}"
        if not candidate.exists():
            return candidate
        i += 1

def download_one(
    session: requests.Session,
    url: str,
    dest: Path,
    timeout: int = 40,
    max_retries: int = 3,
) -> Tuple[str, Optional[str]]:
    """Download a single file with retry logic and exponential backoff.
    
    Args:
        session: Requests session
        url: URL to download
        dest: Destination file path
        timeout: Request timeout in seconds
        max_retries: Maximum number of retry attempts
        
    Returns:
        Tuple of (url, error_message) where error_message is None on success
    """
    last_error = None
    
    for attempt in range(max_retries):
        try:
            with session.get(url, stream=True, timeout=timeout) as resp:
                # Handle specific status codes
                if resp.status_code == 404:
                    return url, "404 Not Found"
                elif resp.status_code == 429:
                    # Rate limited - wait with exponential backoff
                    wait_time = (2 ** attempt) + (random.random() * 0.1)
                    if attempt < max_retries - 1:
                        time.sleep(wait_time)
                        continue
                    return url, f"429 Rate Limited (after {max_retries} attempts)"
                elif resp.status_code >= 500:
                    # Server error - retry with backoff
                    wait_time = (2 ** attempt) + (random.random() * 0.1)
                    if attempt < max_retries - 1:
                        time.sleep(wait_time)
                        continue
                    return url, f"{resp.status_code} Server Error (after {max_retries} attempts)"
                
                resp.raise_for_status()
                
                # Download file
                tmp = dest.with_suffix(dest.suffix + ".part")
                with open(tmp, "wb") as f:
                    for chunk in resp.iter_content(chunk_size=1024 * 128):
                        if chunk:
                            f.write(chunk)
                os.replace(tmp, dest)
                return url, None
                
        except requests.exceptions.Timeout as e:
            last_error = f"Timeout: {str(e)}"
            if attempt < max_retries - 1:
                time.sleep((2 ** attempt) + (random.random() * 0.1))
        except requests.exceptions.RequestException as e:
            last_error = f"Request error: {str(e)}"
            # Don't retry on client errors (4xx except 429)
            if hasattr(e.response, 'status_code') and 400 <= e.response.status_code < 500:
                if e.response.status_code != 429:
                    return url, f"{e.response.status_code} Client Error: {str(e)}"
            if attempt < max_retries - 1:
                time.sleep((2 ** attempt) + (random.random() * 0.1))
        except Exception as e:
            last_error = f"Unexpected error: {str(e)}"
            if attempt < max_retries - 1:
                time.sleep((2 ** attempt) + (random.random() * 0.1))
    
    return url, last_error or "Unknown error"

def download_all(items: List[DownloadItem], out_dir: Path, concurrency: int = 8) -> Tuple[int, int]:
    out_dir.mkdir(parents=True, exist_ok=True)
    ok = 0
    fail = 0
    session = make_session()

    with ThreadPoolExecutor(max_workers=max(1, concurrency)) as ex:
        futures = []
        for it in items:
            target = choose_target_path(out_dir, it.full_url)
            futures.append(ex.submit(download_one, session, it.full_url, target))

        for fut in tqdm(as_completed(futures), total=len(futures), desc="Downloading"):
            url, err = fut.result()
            if err is None:
                ok += 1
            else:
                fail += 1
    return ok, fail

def main():
    ap = argparse.ArgumentParser(description="Download full images from a gallery page")
    ap.add_argument("--url", required=True, help="Page URL to scrape")
    ap.add_argument("--out", required=True, type=Path, help="Output folder")
    ap.add_argument("--selector", default=".galleria-thumbnails img", help="CSS selector to find thumbnails")
    ap.add_argument("--concurrency", type=int, default=8, help="Parallel downloads")

    args = ap.parse_args()

    session = make_session()
    items = discover_images(args.url, args.selector, session)
    if not items:
        print("No images discovered with the given selector.")
        sys.exit(2)

    print(f"Discovered {len(items)} images.")
    ok, fail = download_all(items, args.out, concurrency=args.concurrency)
    print(f"Done. Successful: {ok}, Failed: {fail}")

if __name__ == "__main__":
    main()
