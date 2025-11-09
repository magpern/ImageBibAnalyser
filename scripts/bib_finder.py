"""
Race Bib Finder — scans a folder of race photos, extracts bib numbers with OCR,
and writes a CSV mapping image files to detected bib IDs. Optionally saves
annotated previews and stores results in a database for querying.

Requirements (install):
  - Python 3.9+
  - Tesseract OCR engine installed on your system
      macOS:   brew install tesseract
      Ubuntu:  sudo apt-get install tesseract-ocr
      Windows: Install from https://github.com/UB-Mannheim/tesseract/wiki
  - Python packages:
      pip install opencv-python pytesseract numpy pandas tqdm

Usage:
  python bib_finder.py --input /path/to/images --output results.csv \
      --ext .jpg .jpeg .png --workers 4 --annotate-dir annotated/ \
      --db bibdb.json --image-url https://example.com/gallery/

Notes:
  - By default, detects 2–6 digit numbers (typical bib formats). Adjust --min-digits/--max-digits if needed.
  - If Tesseract is not on PATH, pass --tesseract-cmd to the binary.
  - Use --db to store results for querying with bib_query.py
  - Use --image-url to specify base URL for constructing full image URLs
"""
import argparse
import os
import random
import re
import sys
from pathlib import Path
from typing import List, Tuple, Dict, Set, Optional
from urllib.parse import urljoin

import cv2
import numpy as np
import pandas as pd
import pytesseract
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

try:
    from bib_storage import BibStorage
except ImportError:
    # Handle import when running as module
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    from bib_storage import BibStorage

def build_bib_regex(min_digits: int, max_digits: int) -> re.Pattern:
    pattern = rf"\b\d{{{min_digits},{max_digits}}}\b"
    return re.compile(pattern)

def preprocess(img: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    eq = clahe.apply(gray)
    blur = cv2.bilateralFilter(eq, d=7, sigmaColor=50, sigmaSpace=50)
    thr = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                cv2.THRESH_BINARY, 31, 10)
    return thr

def run_tesseract(img: np.ndarray, psm: int = 6) -> pd.DataFrame:
    config = f"--psm {psm} -l eng --oem 3"
    data = pytesseract.image_to_data(img, config=config, output_type=pytesseract.Output.DATAFRAME)
    data['conf'] = pd.to_numeric(data['conf'], errors='coerce')
    data = data.dropna(subset=['text', 'conf'])
    data['text'] = data['text'].astype(str)
    return data

def rotate_image(img: np.ndarray, angle: int) -> np.ndarray:
    if angle % 360 == 0:
        return img
    if angle % 180 == 0:
        return cv2.rotate(img, cv2.ROTATE_180)
    if angle == 90 or angle == -270:
        return cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    if angle == -90 or angle == 270:
        return cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w/2, h/2), angle, 1.0)
    return cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)

def detect_bibs_for_image(
    path: Path,
    bib_regex: re.Pattern,
    min_confidence: int = 60,
    rotations: Tuple[int, ...] = (0, 90, -90, 180),
    psm_values: Tuple[int, ...] = (6, 7, 11),
    annotate_dir: Path | None = None,
) -> Tuple[Path, List[str], List[float]]:
    img = cv2.imread(str(path))
    if img is None:
        return path, [], []

    all_hits: Dict[str, float] = {}
    best_ann_img = None
    best_hits_for_ann: Set[str] = set()

    for angle in rotations:
        rotated = rotate_image(img, angle)
        proc = preprocess(rotated)
        for psm in psm_values:
            df = run_tesseract(proc, psm=psm)
            for _, row in df.iterrows():
                text = row['text']
                conf = float(row['conf']) if not np.isnan(row['conf']) else 0.0
                if conf < min_confidence:
                    continue
                for m in bib_regex.finditer(text):
                    bib = m.group(0)
                    if bib not in all_hits or conf > all_hits[bib]:
                        all_hits[bib] = conf

            if annotate_dir is not None and not df.empty:
                hits_in_frame: Set[str] = set()
                for _, row in df.iterrows():
                    text = str(row['text'])
                    conf = float(row['conf']) if not np.isnan(row['conf']) else 0.0
                    if conf < min_confidence:
                        continue
                    if bib_regex.fullmatch(text) or any(bib_regex.finditer(text)):
                        hits_in_frame.add(text)
                if len(hits_in_frame) > len(best_hits_for_ann):
                    vis = rotated.copy()
                    for _, row in df.iterrows():
                        x, y, w, h = int(row['left']), int(row['top']), int(row['width']), int(row['height'])
                        text = str(row['text'])
                        conf = float(row['conf']) if not np.isnan(row['conf']) else 0.0
                        if conf < min_confidence:
                            continue
                        if bib_regex.fullmatch(text) or any(bib_regex.finditer(text)):
                            cv2.rectangle(vis, (x, y), (x + w, y + h), (0, 255, 0), 2)
                            label = f"{text} ({int(conf)})"
                            cv2.putText(vis, label, (x, max(0, y - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    best_ann_img = vis
                    best_hits_for_ann = hits_in_frame

    bibs_sorted = sorted(all_hits.keys(), key=lambda k: (-all_hits[k], k))
    confidences_sorted = [all_hits[bib] for bib in bibs_sorted]

    if annotate_dir is not None and best_ann_img is not None:
        annotate_dir.mkdir(parents=True, exist_ok=True)
        out_path = annotate_dir / f"{path.stem}_annotated{path.suffix}"
        cv2.imwrite(str(out_path), best_ann_img)

    return path, bibs_sorted, confidences_sorted

def gather_images(root: Path, exts: Tuple[str, ...]) -> List[Path]:
    files: List[Path] = []
    for ext in exts:
        files.extend(root.rglob(f"*{ext}"))
        files.extend(root.rglob(f"*{ext.upper()}"))
    return sorted(set(files))

def process_all(
    inputs: List[Path],
    bib_regex: re.Pattern,
    min_conf: int,
    workers: int,
    rotations: Tuple[int, ...],
    psm_values: Tuple[int, ...],
    annotate_dir: Path | None,
) -> List[Tuple[Path, List[str], List[float]]]:
    results: List[Tuple[Path, List[str], List[float]]] = []
    with ThreadPoolExecutor(max_workers=max(1, workers)) as ex:
        futs = {
            ex.submit(
                detect_bibs_for_image,
                p,
                bib_regex,
                min_conf,
                rotations,
                psm_values,
                annotate_dir,
            ): p
            for p in inputs
        }
        for fut in tqdm(as_completed(futs), total=len(futs), desc="Scanning"):
            results.append(fut.result())
    return results

def main():
    ap = argparse.ArgumentParser(description="Race bib OCR over a folder of images")
    ap.add_argument("--input", required=True, type=Path, help="Folder with images")
    ap.add_argument("--output", required=True, type=Path, help="Output CSV path")
    ap.add_argument("--ext", nargs="+", default=(".jpg", ".jpeg", ".png"), help="Image extensions to include")
    ap.add_argument("--workers", type=int, default=4, help="Parallel workers")
    ap.add_argument("--min-digits", type=int, default=2, help="Minimum bib digits")
    ap.add_argument("--max-digits", type=int, default=6, help="Maximum bib digits")
    ap.add_argument("--min-conf", type=int, default=60, help="Minimum OCR confidence (0–100)")
    ap.add_argument("--annotate-dir", type=Path, default=None, help="If set, save annotated previews here")
    ap.add_argument("--tesseract-cmd", type=str, default=None, help="Path to tesseract binary if not on PATH")
    ap.add_argument("--psm", nargs="+", type=int, default=[6, 7, 11], help="Tesseract PSM modes to try")
    ap.add_argument("--rotations", nargs="+", type=int, default=[0, 90, -90, 180], help="Image rotations to try")
    ap.add_argument("--bib-pattern", type=str, default=None, help="Custom regex pattern for bib numbers (overrides --min-digits/--max-digits)")
    ap.add_argument("--db", type=Path, default=None, help="Path to bib storage database (JSON file)")
    ap.add_argument("--image-url", type=str, default=None, help="Base URL for constructing full image URLs (e.g., https://example.com/gallery/)")
    ap.add_argument("--limit", type=int, default=None, help="Process only first N images")
    ap.add_argument("--sample", type=int, default=None, help="Randomly sample N images to process")

    args = ap.parse_args()

    if args.tesseract_cmd:
        pytesseract.pytesseract.tesseract_cmd = args.tesseract_cmd

    exts = tuple(args.ext)
    paths = gather_images(args.input, exts)
    if not paths:
        print("No images found for the given extensions.")
        sys.exit(2)

    # Apply limit or sample if specified
    if args.sample is not None:
        if args.sample > len(paths):
            print(f"Warning: --sample {args.sample} is larger than available images ({len(paths)}). Using all images.")
        else:
            paths = random.sample(paths, min(args.sample, len(paths)))
            print(f"Randomly sampled {len(paths)} images.")
    elif args.limit is not None:
        paths = paths[:args.limit]
        print(f"Limited to first {len(paths)} images.")

    # Build regex pattern
    if args.bib_pattern:
        bib_regex = re.compile(args.bib_pattern)
    else:
        bib_regex = build_bib_regex(args.min_digits, args.max_digits)

    # Initialize storage if database path provided
    storage: Optional[BibStorage] = None
    if args.db:
        storage = BibStorage(args.db)
        print(f"Using storage database: {args.db}")

    results = process_all(
        inputs=paths,
        bib_regex=bib_regex,
        min_conf=args.min_conf,
        workers=args.workers,
        rotations=tuple(args.rotations),
        psm_values=tuple(args.psm),
        annotate_dir=args.annotate_dir,
    )

    rows = []
    for p, bibs, confidences in sorted(results, key=lambda t: str(t[0])):
        rows.append({"image": str(p), "bibs": ";".join(bibs)})
        
        # Store in database if storage is enabled
        if storage:
            # Construct image URL
            if args.image_url:
                # Use base URL + relative path from input directory
                rel_path = p.relative_to(args.input)
                image_url = urljoin(args.image_url.rstrip('/') + '/', str(rel_path).replace('\\', '/'))
            else:
                # Fallback to local file path as URL
                image_url = str(p)
            
            storage.store_detection(
                image_url=image_url,
                bibs=bibs,
                confidences=confidences,
                local_path=str(p),
            )

    df = pd.DataFrame(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output, index=False)

    json_path = args.output.with_suffix(".json")
    mapping = {str(p): bibs for p, bibs, _ in results}
    pd.Series(mapping).to_json(json_path, force_ascii=False, indent=2)

    print(f"Processed {len(paths)} images.")
    print(f"Wrote CSV to {args.output}")
    print(f"Wrote JSON to {json_path}")
    if args.annotate_dir:
        print(f"Annotated previews in {args.annotate_dir}")
    if storage:
        stats = storage.get_stats()
        print(f"Storage stats: {stats['total_images']} images, {stats['unique_bibs']} unique bibs, {stats['total_detections']} total detections")

if __name__ == "__main__":
    main()
