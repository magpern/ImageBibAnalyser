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


def preprocess(img: np.ndarray, use_gpu: bool = False) -> np.ndarray:
    """
    Preprocess image for OCR. Optionally uses GPU acceleration if available.

    Note: GPU acceleration requires OpenCV built with CUDA support or CuPy.
    Tesseract OCR itself is CPU-only, so GPU mainly speeds up preprocessing.
    """
    if use_gpu:
        try:
            from bib_finder_gpu import preprocess_gpu, is_gpu_available

            if is_gpu_available():
                return preprocess_gpu(img, use_cuda=True)
        except (ImportError, AttributeError):
            pass  # Fallback to CPU

    # CPU-based preprocessing (default)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    eq = clahe.apply(gray)
    blur = cv2.bilateralFilter(eq, d=7, sigmaColor=50, sigmaSpace=50)
    thr = cv2.adaptiveThreshold(
        blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 10
    )
    return thr


def extract_bib_region(img: np.ndarray, region_ratio: float = 0.6) -> np.ndarray:
    """Extract center-upper region where bibs are typically located.

    Args:
        img: Input image
        region_ratio: Ratio of image to keep (0.6 = keep 60% of width/height from center)

    Returns:
        Cropped image focusing on bib region
    """
    h, w = img.shape[:2]
    # Focus on center-upper region (where bibs typically are)
    # Keep center 60% of width, and upper 60% of height
    crop_w = int(w * region_ratio)
    crop_h = int(h * region_ratio)
    start_x = (w - crop_w) // 2
    start_y = 0  # Start from top
    return img[start_y : start_y + crop_h, start_x : start_x + crop_w]


def run_tesseract(img: np.ndarray, psm: int = 6) -> pd.DataFrame:
    config = f"--psm {psm} -l eng --oem 3"
    data = pytesseract.image_to_data(img, config=config, output_type=pytesseract.Output.DATAFRAME)
    data["conf"] = pd.to_numeric(data["conf"], errors="coerce")
    data = data.dropna(subset=["text", "conf"])
    data["text"] = data["text"].astype(str)
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
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    return cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)


def detect_bibs_for_image(
    path: Path,
    bib_regex: re.Pattern,
    min_confidence: int = 60,
    rotations: Tuple[int, ...] = (0, 90, -90, 180),
    psm_values: Tuple[int, ...] = (6, 7, 11),
    annotate_dir: Path | None = None,
    focus_region: bool = True,
    min_text_size: float = 0.01,
    max_text_size: float = 0.3,
    use_gpu: bool = False,
) -> Tuple[Path, List[str], List[float]]:
    img = cv2.imread(str(path))
    if img is None:
        return path, [], []

    all_hits: Dict[str, Tuple[float, float]] = {}  # bib -> (confidence, size_ratio)
    best_ann_img = None
    best_hits_for_ann: Set[str] = set()
    img_h, img_w = img.shape[:2]
    img_area = img_h * img_w

    for angle in rotations:
        rotated = rotate_image(img, angle)

        # Optionally focus on bib region (center-upper portion)
        crop_offset_x = 0
        crop_offset_y = 0
        if focus_region:
            h_rot, w_rot = rotated.shape[:2]
            crop_w = int(w_rot * 0.6)
            crop_h = int(h_rot * 0.6)
            crop_offset_x = (w_rot - crop_w) // 2
            crop_offset_y = 0
            work_img = rotated[
                crop_offset_y : crop_offset_y + crop_h, crop_offset_x : crop_offset_x + crop_w
            ]
        else:
            work_img = rotated

        proc = preprocess(work_img, use_gpu=use_gpu)
        for psm in psm_values:
            df = run_tesseract(proc, psm=psm)
            for _, row in df.iterrows():
                text = str(row["text"]).strip()
                conf = float(row["conf"]) if not np.isnan(row["conf"]) else 0.0
                if conf < min_confidence:
                    continue

                # Check if text matches bib pattern exactly (preferred) or contains it
                bib_match = None
                if bib_regex.fullmatch(text):
                    # Exact match - highest priority
                    bib_match = text
                else:
                    # Check if text contains a bib number
                    matches = list(bib_regex.finditer(text))
                    if matches:
                        # Prefer longer matches (more digits)
                        bib_match = max(matches, key=lambda m: len(m.group(0))).group(0)

                if bib_match:
                    # Calculate size ratio of detected text
                    w = float(row["width"])
                    h = float(row["height"])
                    text_area = w * h
                    size_ratio = text_area / img_area if img_area > 0 else 0

                    # Filter by size - bibs should be reasonably sized
                    if size_ratio < min_text_size or size_ratio > max_text_size:
                        continue

                    # Store with confidence and size (prefer higher confidence, then larger size)
                    if bib_match not in all_hits:
                        all_hits[bib_match] = (conf, size_ratio)
                    else:
                        old_conf, old_size = all_hits[bib_match]
                        # Update if new detection has higher confidence, or same confidence but larger size
                        if conf > old_conf or (conf == old_conf and size_ratio > old_size):
                            all_hits[bib_match] = (conf, size_ratio)

            if annotate_dir is not None and not df.empty:
                hits_in_frame: Set[str] = set()
                for _, row in df.iterrows():
                    text = str(row["text"]).strip()
                    conf = float(row["conf"]) if not np.isnan(row["conf"]) else 0.0
                    if conf < min_confidence:
                        continue

                    # Check size filter
                    w = float(row["width"])
                    h = float(row["height"])
                    text_area = w * h
                    size_ratio = text_area / img_area if img_area > 0 else 0
                    if size_ratio < min_text_size or size_ratio > max_text_size:
                        continue

                    if bib_regex.fullmatch(text):
                        hits_in_frame.add(text)
                    else:
                        matches = list(bib_regex.finditer(text))
                        if matches:
                            hits_in_frame.add(max(matches, key=lambda m: len(m.group(0))).group(0))

                if len(hits_in_frame) > len(best_hits_for_ann):
                    vis = rotated.copy()
                    # Draw region focus if enabled
                    if focus_region:
                        h_vis, w_vis = rotated.shape[:2]
                        crop_w = int(w_vis * 0.6)
                        crop_h = int(h_vis * 0.6)
                        start_x = (w_vis - crop_w) // 2
                        cv2.rectangle(vis, (start_x, 0), (start_x + crop_w, crop_h), (255, 0, 0), 2)

                    for _, row in df.iterrows():
                        # Adjust coordinates if we cropped the image
                        x = int(row["left"]) + crop_offset_x
                        y = int(row["top"]) + crop_offset_y
                        w = int(row["width"])
                        h = int(row["height"])
                        text = str(row["text"]).strip()
                        conf = float(row["conf"]) if not np.isnan(row["conf"]) else 0.0
                        if conf < min_confidence:
                            continue

                        # Check size filter (use original image area)
                        text_area = w * h
                        size_ratio = text_area / img_area if img_area > 0 else 0
                        if size_ratio < min_text_size or size_ratio > max_text_size:
                            continue

                        is_bib = False
                        if bib_regex.fullmatch(text):
                            is_bib = True
                        else:
                            matches = list(bib_regex.finditer(text))
                            if matches:
                                is_bib = True
                                text = max(matches, key=lambda m: len(m.group(0))).group(0)

                        if is_bib:
                            cv2.rectangle(vis, (x, y), (x + w, y + h), (0, 255, 0), 2)
                            label = f"{text} ({int(conf)})"
                            cv2.putText(
                                vis,
                                label,
                                (x, max(0, y - 5)),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.6,
                                (0, 255, 0),
                                2,
                            )
                    best_ann_img = vis
                    best_hits_for_ann = hits_in_frame

    # Sort by confidence (descending), then by size (descending)
    bibs_sorted = sorted(all_hits.keys(), key=lambda k: (-all_hits[k][0], -all_hits[k][1], k))
    confidences_sorted = [all_hits[bib][0] for bib in bibs_sorted]

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
    focus_region: bool = True,
    min_text_size: float = 0.01,
    max_text_size: float = 0.3,
    use_gpu: bool = False,
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
                focus_region,
                min_text_size,
                max_text_size,
                use_gpu,
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
    ap.add_argument(
        "--ext", nargs="+", default=(".jpg", ".jpeg", ".png"), help="Image extensions to include"
    )
    ap.add_argument("--workers", type=int, default=4, help="Parallel workers")
    ap.add_argument("--min-digits", type=int, default=2, help="Minimum bib digits")
    ap.add_argument("--max-digits", type=int, default=6, help="Maximum bib digits")
    ap.add_argument(
        "--min-conf",
        type=int,
        default=60,
        help="Minimum OCR confidence (0–100). Lower values = more detections but more false positives. Try 40-50 if missing bibs.",
    )
    ap.add_argument(
        "--annotate-dir", type=Path, default=None, help="If set, save annotated previews here"
    )
    ap.add_argument(
        "--tesseract-cmd", type=str, default=None, help="Path to tesseract binary if not on PATH"
    )
    ap.add_argument(
        "--psm",
        nargs="+",
        type=int,
        default=[6, 7, 8, 11, 13],
        help="Tesseract PSM modes to try (default: 6,7,8,11,13). More modes = better detection but slower. Try --psm 8 13 for single words/numbers.",
    )
    ap.add_argument(
        "--rotations",
        nargs="+",
        type=int,
        default=[0, 90, -90, 180],
        help="Image rotations to try in degrees (default: 0,90,-90,180). Add more angles if bibs are tilted.",
    )
    ap.add_argument(
        "--bib-pattern",
        type=str,
        default=None,
        help="Custom regex pattern for bib numbers (overrides --min-digits/--max-digits)",
    )
    ap.add_argument(
        "--db", type=Path, default=None, help="Path to bib storage database (JSON file)"
    )
    ap.add_argument(
        "--image-url",
        type=str,
        default=None,
        help="Base URL for constructing full image URLs (e.g., https://example.com/gallery/)",
    )
    ap.add_argument("--limit", type=int, default=None, help="Process only first N images")
    ap.add_argument("--sample", type=int, default=None, help="Randomly sample N images to process")
    ap.add_argument(
        "--aggressive",
        action="store_true",
        help="Aggressive mode: lower confidence threshold, more PSM modes, more rotations",
    )
    ap.add_argument(
        "--no-focus-region",
        action="store_true",
        help="Disable region focusing (process entire image). By default, focuses on center-upper region where bibs typically are.",
    )
    ap.add_argument(
        "--min-text-size",
        type=float,
        default=0.01,
        help="Minimum text size ratio (relative to image area) to consider as bib. Default 0.01 (1%%). Filters out tiny false positives.",
    )
    ap.add_argument(
        "--max-text-size",
        type=float,
        default=0.3,
        help="Maximum text size ratio (relative to image area) to consider as bib. Default 0.3 (30%%). Filters out oversized detections.",
    )

    args = ap.parse_args()

    if args.tesseract_cmd:
        pytesseract.pytesseract.tesseract_cmd = args.tesseract_cmd

    # Apply aggressive mode settings
    if args.aggressive:
        # Override defaults for better detection
        if args.min_conf == 60:  # Default value
            args.min_conf = 40
        # Check if using default PSM modes
        default_psm = {6, 7, 8, 11, 13}
        if set(args.psm) == default_psm or set(args.psm) == {6, 7, 11}:
            args.psm = [6, 7, 8, 11, 13]
        # Check if using default rotations
        default_rotations = {0, 90, -90, 180}
        if set(args.rotations) == default_rotations:
            args.rotations = [0, 45, 90, -45, -90, 135, -135, 180]
        print("Aggressive mode enabled: lower confidence, more PSM modes, more rotations")

    exts = tuple(args.ext)
    paths = gather_images(args.input, exts)
    if not paths:
        print("No images found for the given extensions.")
        sys.exit(2)

    # Apply limit or sample if specified
    if args.sample is not None:
        if args.sample > len(paths):
            print(
                f"Warning: --sample {args.sample} is larger than available images ({len(paths)}). Using all images."
            )
        else:
            paths = random.sample(paths, min(args.sample, len(paths)))
            print(f"Randomly sampled {len(paths)} images.")
    elif args.limit is not None:
        paths = paths[: args.limit]
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

    # Check GPU availability if requested
    if args.use_gpu:
        try:
            from bib_finder_gpu import is_gpu_available, get_gpu_info

            if is_gpu_available():
                gpu_info = get_gpu_info()
                print(f"GPU acceleration enabled: {gpu_info}")
            else:
                print("Warning: --use-gpu specified but GPU not available. Falling back to CPU.")
                args.use_gpu = False
        except ImportError:
            print(
                "Warning: GPU support not available. Install opencv-contrib-python or cupy for GPU acceleration."
            )
            args.use_gpu = False

    results = process_all(
        inputs=paths,
        bib_regex=bib_regex,
        min_conf=args.min_conf,
        workers=args.workers,
        rotations=tuple(args.rotations),
        psm_values=tuple(args.psm),
        annotate_dir=args.annotate_dir,
        focus_region=not args.no_focus_region,
        min_text_size=args.min_text_size,
        max_text_size=args.max_text_size,
        use_gpu=args.use_gpu,
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
                image_url = urljoin(
                    args.image_url.rstrip("/") + "/", str(rel_path).replace("\\", "/")
                )
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
        print(
            f"Storage stats: {stats['total_images']} images, {stats['unique_bibs']} unique bibs, {stats['total_detections']} total detections"
        )


if __name__ == "__main__":
    main()
