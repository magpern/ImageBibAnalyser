"""
YOLO-based Bib Detector with optional OCR recognition.

This pipeline uses a YOLO model (via Ultralytics) to detect bib regions in
race photos and optionally runs PaddleOCR on the cropped detections to extract
numeric bib identifiers. Results can be saved to CSV/JSON and optionally
persisted in the JSON database used by the rest of the project.

Prerequisites:
  - pip install ultralytics paddleocr (see requirements.txt)
  - A YOLO weights file trained to detect bib numbers (pass via --weights)

Usage (detection + OCR):
  racebib yolo \
      --input ./photos \
      --output results/yolo_results.csv \
      --weights ./models/bib_yolo.pt \
      --db bibdb.json \
      --image-url https://example.com/gallery/

If you do not have a YOLO model yet, train one first (see README for guidance).
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

# Set Ultralytics config directory to persistent location if not already set
# Default to /app/.ultralytics (persists in container) or /tmp/Ultralytics as fallback
if "YOLO_CONFIG_DIR" not in os.environ:
    # Try persistent location first, fallback to /tmp if not writable
    persistent_dir = Path("/app/.ultralytics")
    try:
        persistent_dir.mkdir(parents=True, exist_ok=True)
        test_file = persistent_dir / ".test_write"
        test_file.touch()
        test_file.unlink()
        os.environ["YOLO_CONFIG_DIR"] = str(persistent_dir)
    except (PermissionError, OSError):
        # Fallback to /tmp if persistent location isn't writable
        os.environ["YOLO_CONFIG_DIR"] = "/tmp/Ultralytics"

# Ensure the config directory exists and is writable
# Ultralytics creates a nested "Ultralytics" subdirectory inside YOLO_CONFIG_DIR
config_dir = Path(os.environ.get("YOLO_CONFIG_DIR", "/app/.ultralytics"))
ultralytics_subdir = config_dir / "Ultralytics"
ultralytics_subdir.mkdir(parents=True, exist_ok=True)
try:
    # Test write permissions in the nested directory
    test_file = ultralytics_subdir / ".test_write"
    test_file.touch()
    test_file.unlink()
except (PermissionError, OSError):
    # Final fallback to /tmp if the configured directory isn't writable
    os.environ["YOLO_CONFIG_DIR"] = "/tmp/Ultralytics"
    Path("/tmp/Ultralytics/Ultralytics").mkdir(parents=True, exist_ok=True)

from ultralytics import YOLO

try:
    from paddleocr import PaddleOCR
except ImportError:  # pragma: no cover - handled at runtime
    PaddleOCR = None  # type: ignore

try:
    from bib_storage import BibStorage
except ImportError:  # pragma: no cover - handled when run as script
    sys.path.insert(0, str(Path(__file__).parent))
    from bib_storage import BibStorage


@dataclass
class DetectionResult:
    bib_number: str
    detection_conf: float
    ocr_conf: float
    bbox: Tuple[int, int, int, int]


def load_yolo_model(weights_path: Path, device: str) -> YOLO:
    try:
        model = YOLO(str(weights_path))
    except Exception as exc:  # pragma: no cover - runtime dependency
        raise RuntimeError(f"Failed to load YOLO model from {weights_path}: {exc}") from exc

    # Validate device and print what's actually being used
    device = device.lower()
    actual_device = str(model.device)
    print(f"YOLO model loaded. Requested device: {device}, Actual device: {actual_device}", file=sys.stderr)
    
    if device in ("cuda", "0", "cuda:0") and not actual_device.startswith("cuda"):
        print("Warning: CUDA requested but not available; running on CPU instead.", file=sys.stderr)
    elif device in ("cuda", "0", "cuda:0") and actual_device.startswith("cuda"):
        print(f"✓ GPU acceleration enabled: {actual_device}", file=sys.stderr)
    
    return model


def init_paddle_ocr(use_gpu: bool, lang: str) -> PaddleOCR | None:
    if PaddleOCR is None:
        raise RuntimeError(
            "PaddleOCR is not installed. Install with 'pip install paddleocr paddlepaddle'."
        )
    try:
        # PaddleOCR API has changed in newer versions - use minimal parameters
        # Note: We use YOLO for detection, so PaddleOCR is only used for recognition (text extraction)
        # GPU support and other options are handled automatically by PaddleOCR based on available hardware
        ocr = PaddleOCR(lang=lang)
        return ocr
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(f"Failed to initialize PaddleOCR: {exc}") from exc


def crop_image(
    img: np.ndarray, bbox: Tuple[int, int, int, int], focus_white_area: bool = False
) -> np.ndarray:
    """Crop image to bounding box, optionally focusing on white area (excluding top/bottom bands).
    
    Args:
        img: Input image
        bbox: Bounding box (x1, y1, x2, y2)
        focus_white_area: If True, crop to focus on center white area (exclude top 20% and bottom 10%)
    """
    x1, y1, x2, y2 = bbox
    h, w = img.shape[:2]
    x1 = max(0, min(w - 1, x1))
    x2 = max(0, min(w, x2))
    y1 = max(0, min(h - 1, y1))
    y2 = max(0, min(h, y2))
    if x2 <= x1 or y2 <= y1:
        return img[0:0, 0:0]
    
    if focus_white_area:
        # Focus on white area: exclude top 20% (black band with year) and bottom 10% (sponsor area)
        crop_h = y2 - y1
        top_crop = int(crop_h * 0.2)  # Exclude top 20%
        bottom_crop = int(crop_h * 0.1)  # Exclude bottom 10%
        y1_focused = y1 + top_crop
        y2_focused = y2 - bottom_crop
        if y2_focused > y1_focused:
            return img[y1_focused:y2_focused, x1:x2]
    
    return img[y1:y2, x1:x2]


def rotate_image(img: np.ndarray, angle: float) -> np.ndarray:
    """Rotate image by given angle in degrees.
    
    Args:
        img: Input image
        angle: Rotation angle in degrees (positive = counterclockwise)
    
    Returns:
        Rotated image
    """
    if abs(angle) < 0.1:
        return img
    
    h, w = img.shape[:2]
    center = (w / 2, h / 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    
    # Calculate new image dimensions to avoid cropping
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])
    new_w = int((h * sin) + (w * cos))
    new_h = int((h * cos) + (w * sin))
    
    # Adjust rotation matrix for new dimensions
    M[0, 2] += (new_w / 2) - center[0]
    M[1, 2] += (new_h / 2) - center[1]
    
    return cv2.warpAffine(img, M, (new_w, new_h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)


def detect_skew_angle(img: np.ndarray) -> float:
    """Detect skew angle in image using Hough line transform.
    
    Args:
        img: Input grayscale image
    
    Returns:
        Detected skew angle in degrees (positive = counterclockwise)
    """
    # Convert to grayscale if needed
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img.copy()
    
    # Apply edge detection
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    
    # Detect lines using Hough transform
    lines = cv2.HoughLines(edges, 1, np.pi / 180, 100)
    
    if lines is None or len(lines) == 0:
        return 0.0
    
    # Calculate angles of detected lines
    angles = []
    for line in lines[:20]:  # Use first 20 lines
        rho, theta = line[0]
        # Convert to degrees and normalize to -45 to 45 range
        angle = np.degrees(theta) - 90
        if angle < -45:
            angle += 90
        elif angle > 45:
            angle -= 90
        angles.append(angle)
    
    if not angles:
        return 0.0
    
    # Return median angle (more robust than mean)
    return float(np.median(angles))


def preprocess_crop_for_ocr(crop: np.ndarray, method: str = "default") -> np.ndarray:
    """Preprocess cropped bib image for better OCR recognition.
    
    Args:
        crop: Cropped bib image (BGR)
        method: Preprocessing method ("default", "aggressive", "inverse")
    
    Returns:
        Preprocessed image (BGR)
    """
    if crop.size == 0:
        return crop
    
    # Convert to grayscale
    if len(crop.shape) == 3:
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    else:
        gray = crop.copy()
    
    if method == "aggressive":
        # More aggressive preprocessing: high contrast enhancement
        clahe = cv2.createCLAHE(clipLimit=5.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        # Apply adaptive threshold
        thresh = cv2.adaptiveThreshold(
            enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
        )
        # Convert back to BGR
        return cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)
    elif method == "inverse":
        # Try inverse thresholding (for white text on dark background)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        # Invert and threshold
        inverted = cv2.bitwise_not(enhanced)
        thresh = cv2.adaptiveThreshold(
            inverted, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
        )
        return cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)
    else:
        # Default: moderate contrast enhancement
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        # Convert back to BGR
        return cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)


def run_ocr_on_crop(
    ocr: PaddleOCR,
    crop: np.ndarray,
    bib_regex: Optional[re.Pattern],
    min_digits: int,
    max_digits: int,
    try_rotations: bool = True,
) -> List[Tuple[str, float]]:
    """Run OCR on cropped bib image, optionally trying multiple rotations for skewed bibs.
    
    Args:
        ocr: PaddleOCR instance
        crop: Cropped bib image
        bib_regex: Optional regex pattern for bib numbers
        min_digits: Minimum number of digits in bib number
        max_digits: Maximum number of digits in bib number
        try_rotations: If True, try OCR at multiple angles to handle skewed bibs
    
    Returns:
        List of (bib_number, confidence) tuples
    """
    if crop.size == 0:
        return []

    # Try OCR at original angle first with different preprocessing methods
    all_results: List[Tuple[str, float]] = []
    angles_to_try = [0.0]  # Always try original first
    
    # Try different preprocessing methods
    preprocessing_methods = ["default", "aggressive", "inverse"]
    
    for prep_method in preprocessing_methods:
        # Preprocess crop
        preprocessed_crop = preprocess_crop_for_ocr(crop, method=prep_method)
        
        # Try OCR on preprocessed crop
        try:
            results = ocr.ocr(preprocessed_crop, det=False, rec=True, cls=False)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            results = ocr.ocr(preprocessed_crop)  # type: ignore[arg-type]
        
        if results:
            # Parse OCR results
            all_ocr_texts = _parse_ocr_results(results)
            prep_results = _extract_bib_numbers_from_ocr_texts(
                all_ocr_texts, bib_regex, min_digits, max_digits
            )
            all_results.extend(prep_results)
    
    # If no good results from preprocessing, try original crop without preprocessing
    try:
        results = ocr.ocr(crop, det=False, rec=True, cls=False)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        results = ocr.ocr(crop)  # type: ignore[arg-type]
    
    if results:
        # Parse OCR results for original angle
        all_ocr_texts = _parse_ocr_results(results)
        original_results = _extract_bib_numbers_from_ocr_texts(
            all_ocr_texts, bib_regex, min_digits, max_digits
        )
        all_results.extend(original_results)
    
    # If no results from any preprocessing method or original crop, try rotations if enabled
    if not all_results or max((conf for _, conf in all_results), default=0.0) < 0.5:
        # Only try rotations if we haven't found good results yet
        if try_rotations:
            # Auto-detect skew angle
            detected_skew = detect_skew_angle(crop)
            if abs(detected_skew) > 2.0:  # Only correct if skew > 2 degrees
                angles_to_try = [-detected_skew]
            else:
                angles_to_try = []
            
            # Also try a few small angles for slight skews (more conservative)
            for angle in [-10, -5, 5, 10]:
                if abs(angle - detected_skew) > 3.0:  # Don't duplicate detected angle
                    angles_to_try.append(float(angle))
            
            # Try OCR at additional angles if needed
            for angle in angles_to_try:
                # Rotate crop if needed
                if abs(angle) > 0.1:
                    rotated_crop = rotate_image(crop, angle)
                else:
                    rotated_crop = crop
                
                # Try with preprocessing on rotated crop
                for prep_method in ["default", "aggressive", "inverse"]:
                    preprocessed_crop = preprocess_crop_for_ocr(rotated_crop, method=prep_method)
                    
                    # Run OCR on rotated crop
                    try:
                        results = ocr.ocr(preprocessed_crop, det=False, rec=True, cls=False)  # type: ignore[arg-type]
                    except (TypeError, ValueError):
                        results = ocr.ocr(preprocessed_crop)  # type: ignore[arg-type]
                    
                    if not results:
                        continue
                    
                    # Parse OCR results for this angle
                    all_ocr_texts = _parse_ocr_results(results)
                    
                    # Process OCR texts to find bib numbers for this angle
                    angle_results = _extract_bib_numbers_from_ocr_texts(
                        all_ocr_texts, bib_regex, min_digits, max_digits
                    )
                    
                    # Only add results if they're high confidence (avoid noise from rotations)
                    for bib, conf in angle_results:
                        if conf > 0.6:  # Only keep high-confidence results from rotations
                            all_results.append((bib, conf))
    
    # If no results from any method, return empty
    if not all_results:
        return []
    
    # Deduplicate and prefer highest confidence for each unique bib number
    bib_dict: Dict[str, float] = {}
    for bib, conf in all_results:
        # Filter out results that don't match expected length
        if min_digits <= len(bib) <= max_digits:
            if bib not in bib_dict or conf > bib_dict[bib]:
                bib_dict[bib] = conf
    
    # Return only the best result (highest confidence) to avoid random numbers
    if bib_dict:
        best_bib = max(bib_dict.items(), key=lambda x: x[1])
        return [best_bib]
    
    return []


def _parse_ocr_results(results: List) -> List[Tuple[str, float]]:
    """Parse PaddleOCR results into list of (text, confidence) tuples.
    
    Args:
        results: PaddleOCR result structure (varies by version)
    
    Returns:
        List of (text, confidence) tuples
    """
    all_ocr_texts: List[Tuple[str, float]] = []
    
    # PaddleOCR result structure has changed in newer versions:
    # New format (v2.7+): List of dicts with 'rec_texts' and 'rec_scores' keys
    # Old format: [[(text, confidence)]] or [[(bbox, (text, confidence))]]
    if isinstance(results, list) and len(results) > 0:
        first_result = results[0]
        if isinstance(first_result, dict) and 'rec_texts' in first_result:
            # New format: dict with 'rec_texts' and 'rec_scores' lists
            rec_texts = first_result.get('rec_texts', [])
            rec_scores = first_result.get('rec_scores', [])
            # Ensure both lists have the same length
            min_len = min(len(rec_texts), len(rec_scores))
            for i in range(min_len):
                text = str(rec_texts[i])
                conf = float(rec_scores[i])
                all_ocr_texts.append((text, conf))
        else:
            # Old format: nested lists
            for res_group in results:
                if not res_group:
                    continue
                for res in res_group:
                    # Check if result is in new format: (bbox, (text, confidence))
                    if isinstance(res[0], (list, tuple)) and len(res[0]) == 4:
                        # New format: (bbox, (text, confidence))
                        text_data = res[1]
                        if isinstance(text_data, tuple) and len(text_data) >= 2:
                            text = text_data[0]
                            conf = float(text_data[1])
                        else:
                            continue
                    elif isinstance(res[1], (list, tuple)) and len(res) >= 2:
                        # Might be: (text, (confidence)) or (text, confidence)
                        text = res[0]
                        conf_data = res[1]
                        if isinstance(conf_data, (list, tuple)):
                            conf = float(conf_data[0]) if conf_data else 0.0
                        else:
                            conf = float(conf_data)
                    else:
                        # Try direct access: (text, confidence)
                        text = res[0]
                        try:
                            conf = float(res[1])
                        except (ValueError, TypeError, IndexError):
                            continue
                    all_ocr_texts.append((text, conf))
    
    return all_ocr_texts


def _extract_bib_numbers_from_ocr_texts(
    all_ocr_texts: List[Tuple[str, float]],
    bib_regex: Optional[re.Pattern],
    min_digits: int,
    max_digits: int,
) -> List[Tuple[str, float]]:
    """Extract bib numbers from OCR text results.
    
    Args:
        all_ocr_texts: List of (text, confidence) tuples from OCR
        bib_regex: Optional regex pattern for bib numbers
        min_digits: Minimum number of digits in bib number
        max_digits: Maximum number of digits in bib number
    
    Returns:
        List of (bib_number, confidence) tuples
    """
    texts: List[Tuple[str, float]] = []
    
    # Process all OCR texts to find bib numbers
    # Filter out common years (2020-2030) and date patterns
    year_pattern = re.compile(r"\b(20[2-3][0-9]|19[0-9]{2})\b")  # Years 1900-2039
    
    # Collect all digit sequences from all OCR texts
    all_digit_sequences: List[Tuple[str, float]] = []
    
    for text, conf in all_ocr_texts:
        if bib_regex:
            matches = list(bib_regex.finditer(text))
            if not matches:
                continue
            for match in matches:
                matched_text = match.group(0)
                # Filter out years
                if not year_pattern.match(matched_text):
                    all_digit_sequences.append((matched_text, conf))
        else:
            # Find all digit sequences in the text
            # First try with min_digits requirement
            digits = re.findall(rf"\d{{{min_digits},{max_digits}}}", text)
            for d in digits:
                # Filter out years (2020-2030 range, or 4-digit years in general)
                if not year_pattern.match(d):
                    all_digit_sequences.append((d, conf))
            
            # If min_digits > 1 and we found no results, also try single digits
            # This helps with bibs like "1", "2", "3" but only if no multi-digit numbers found
            if min_digits > 1 and not digits:
                single_digits = re.findall(r'\b\d\b', text)  # Single digits as whole words
                for d in single_digits:
                    # Filter out years and only accept if it's a standalone digit (not part of larger number)
                    if not year_pattern.match(d):
                        # Only add single digits if confidence is high (to avoid false positives)
                        if conf > 0.8:  # Higher threshold for single digits
                            all_digit_sequences.append((d, conf))
    
    # Process digit sequences: prefer longest, try to combine if needed
    if len(all_digit_sequences) > 1:
        # Sort by length (longest first), then by confidence
        all_digit_sequences.sort(key=lambda x: (-len(x[0]), -x[1]))
        
        # Always take the longest sequence as primary
        longest = all_digit_sequences[0]
        texts.append(longest)
        
        # If longest is short (1-2 digits), try combining with adjacent sequences
        # This handles cases where OCR splits "506" into "5" and "06"
        if len(longest[0]) <= 2:
            for seq, conf in all_digit_sequences[1:]:
                # Try combining: "5" + "06" = "506"
                combined = longest[0] + seq
                if min_digits <= len(combined) <= max_digits:
                    # Use average confidence
                    avg_conf = (longest[1] + conf) / 2
                    texts.append((combined, avg_conf))
                    # Also try reverse: "06" + "5" = "065" (less likely but possible)
                    combined_rev = seq + longest[0]
                    if min_digits <= len(combined_rev) <= max_digits:
                        texts.append((combined_rev, avg_conf))
                    break
    elif len(all_digit_sequences) == 1:
        # Single sequence found - accept it even if short (for cases like "16")
        texts.append(all_digit_sequences[0])
    
    # Also check if any OCR text contains digits mixed with text (e.g., "16RUNNER" -> extract "16")
    # This helps with cases where OCR reads the whole bib including text
    for text, conf in all_ocr_texts:
        # Look for digit sequences in text that might have been missed
        # Extract all digit sequences from the text, even if they're part of larger text
        digit_matches = re.findall(r'\d+', text)
        for digits in digit_matches:
            if min_digits <= len(digits) <= max_digits:
                # Check if we already have this in all_digit_sequences
                if not any(d == digits for d, _ in all_digit_sequences):
                    # Add it with slightly lower confidence (it was mixed with text)
                    texts.append((digits, conf * 0.9))
    
    return texts


def annotate_image(
    image: np.ndarray,
    detections: List[DetectionResult],
    output_path: Path,
) -> None:
    if image.size == 0:
        return
    vis = image.copy()
    for det in detections:
        x1, y1, x2, y2 = det.bbox
        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
        label = f"{det.bib_number} ({det.detection_conf:.2f}/{det.ocr_conf:.2f})"
        cv2.putText(
            vis,
            label,
            (x1, max(0, y1 - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), vis)


def gather_images(root: Path, exts: Tuple[str, ...]) -> List[Path]:
    files: List[Path] = []
    for ext in exts:
        files.extend(root.rglob(f"*{ext}"))
        files.extend(root.rglob(f"*{ext.upper()}"))
    return sorted(set(files))


def main() -> None:
    ap = argparse.ArgumentParser(
        description="YOLO-based Bib Detector with optional OCR recognition"
    )
    ap.add_argument("--input", required=True, type=Path, help="Folder with images")
    ap.add_argument("--output", required=True, type=Path, help="Output CSV path")
    ap.add_argument(
        "--weights", required=True, type=Path, help="YOLO weights file trained for bib detection"
    )
    ap.add_argument(
        "--ext", nargs="+", default=(".jpg", ".jpeg", ".png"), help="Image extensions to include"
    )
    ap.add_argument(
        "--conf", type=float, default=0.3, help="YOLO confidence threshold (default: 0.3)"
    )
    ap.add_argument(
        "--iou", type=float, default=0.45, help="YOLO IoU threshold for NMS (default: 0.45)"
    )
    ap.add_argument(
        "--device", type=str, default="cpu", help="Device to run YOLO on (cpu, cuda, cuda:0, etc.)"
    )
    ap.add_argument(
        "--class-id", type=int, default=0, help="YOLO class ID corresponding to bibs (default: 0)"
    )
    ap.add_argument(
        "--bib-pattern",
        type=str,
        default=None,
        help="Custom regex pattern for bib numbers (overrides min/max digits)",
    )
    ap.add_argument(
        "--min-digits", type=int, default=2, help="Minimum bib digits if regex not provided"
    )
    ap.add_argument(
        "--max-digits", type=int, default=6, help="Maximum bib digits if regex not provided"
    )
    ap.add_argument(
        "--disable-ocr", action="store_true", help="Disable OCR and only output detections"
    )
    ap.add_argument(
        "--ocr-lang",
        type=str,
        default="en",
        help="PaddleOCR language code (default: en). See PaddleOCR docs for options.",
    )
    ap.add_argument("--use-gpu", action="store_true", help="Use GPU for PaddleOCR if available")
    ap.add_argument(
        "--annotate-dir", type=Path, default=None, help="Save annotated images with detections here"
    )
    ap.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of images to process (useful for testing)",
    )
    ap.add_argument(
        "--incremental",
        action="store_true",
        help="Write CSV results incrementally (after each image) instead of at the end",
    )
    ap.add_argument(
        "--save-crops",
        type=Path,
        default=None,
        help="If set, save cropped bib regions to this folder",
    )
    ap.add_argument(
        "--focus-white-area",
        action="store_true",
        help="Crop bib region to focus on white area (exclude top black band with year and bottom sponsor area)",
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

    args = ap.parse_args()

    if not args.weights.exists():
        print(f"Error: YOLO weights not found at {args.weights}", file=sys.stderr)
        sys.exit(1)

    if args.disable_ocr and (args.db is not None):
        print(
            "Warning: --disable-ocr is set, database will only store detection metadata.",
            file=sys.stderr,
        )

    bib_regex: Optional[re.Pattern] = None
    if args.bib_pattern:
        try:
            bib_regex = re.compile(args.bib_pattern)
        except re.error as exc:
            print(f"Error: invalid regex for --bib-pattern: {exc}", file=sys.stderr)
            sys.exit(1)

    # Initialize models
    model = load_yolo_model(args.weights, args.device)
    ocr: Optional[PaddleOCR] = None
    if not args.disable_ocr:
        ocr = init_paddle_ocr(use_gpu=args.use_gpu, lang=args.ocr_lang)

    # Gather images
    exts = tuple(args.ext)
    images = gather_images(args.input, exts)
    if not images:
        print("No images found for the given extensions.")
        sys.exit(2)
    
    # Apply limit if specified
    if args.limit:
        images = images[: args.limit]
        print(f"Processing limited to {len(images)} images", file=sys.stderr)

    # Setup storage/database
    storage: Optional[BibStorage] = None
    if args.db:
        storage = BibStorage(args.db)
        print(f"Using storage database: {args.db}")

    rows: List[Dict[str, object]] = []
    detections_per_image: Dict[str, List[DetectionResult]] = {}
    total_detections_all = 0
    total_filtered_by_class_all = 0

    # Setup incremental CSV writing if requested
    csv_file = None
    csv_writer = None
    if args.incremental:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        csv_file = open(args.output, "w", newline="", encoding="utf-8")
        csv_writer = csv.DictWriter(csv_file, fieldnames=["image", "bibs", "detections"])
        csv_writer.writeheader()
        csv_file.flush()

    for img_path in tqdm(images, desc="Processing images"):
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"Warning: failed to read image {img_path}", file=sys.stderr)
            continue

        # Run YOLO prediction
        results = model.predict(
            source=str(img_path),
            conf=args.conf,
            iou=args.iou,
            device=args.device,
            verbose=False,
        )

        detections: List[DetectionResult] = []
        img_total_detections = 0
        img_filtered_by_class = 0
        low_confidence_detections = []  # Track detections below threshold
        
        for res in results:
            if res.boxes is None:
                continue
            for box in res.boxes:
                img_total_detections += 1
                cls_id = int(box.cls.item())
                if cls_id != args.class_id:
                    img_filtered_by_class += 1
                    if len(detections) == 0 and img_total_detections <= 3:
                        # Debug: print first few mismatches
                        print(
                            f"Debug: Skipping detection with class_id={cls_id} (expected {args.class_id}), "
                            f"conf={box.conf.item():.3f}",
                            file=sys.stderr,
                        )
                    continue
                conf = float(box.conf.item())
                
                # Track detections below confidence threshold
                if conf < args.conf:
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                    low_confidence_detections.append((conf, (x1, y1, x2, y2)))
                    continue
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                crop = crop_image(img, (x1, y1, x2, y2), focus_white_area=args.focus_white_area)
                ocr_hits: List[Tuple[str, float]] = []
                if ocr is not None:
                    ocr_hits = run_ocr_on_crop(
                        ocr, crop, bib_regex, args.min_digits, args.max_digits
                    )
                    # Debug: show what OCR found if no bib numbers extracted (simplified)
                    if not ocr_hits and len(detections) < 5:
                        try:
                            # Run OCR again to see raw results (extract only text, not full structure)
                            raw_results = ocr.ocr(crop)  # type: ignore[arg-type]
                            if raw_results:
                                # Extract only text from results, not the full structure with arrays
                                texts_found = []
                                if isinstance(raw_results, list) and len(raw_results) > 0:
                                    first_result = raw_results[0]
                                    if isinstance(first_result, dict) and 'rec_texts' in first_result:
                                        texts_found = first_result.get('rec_texts', [])
                                    elif isinstance(first_result, list):
                                        for res in first_result:
                                            if isinstance(res, (list, tuple)) and len(res) >= 2:
                                                text_data = res[1]
                                                if isinstance(text_data, tuple) and len(text_data) >= 1:
                                                    texts_found.append(text_data[0])
                                if texts_found:
                                    print(
                                        f"Debug OCR for {img_path.name} bbox({x1},{y1},{x2},{y2}): "
                                        f"Found texts: {texts_found}",
                                        file=sys.stderr,
                                    )
                        except Exception:
                            pass

                if not ocr_hits and ocr is None:
                    # Store detection without OCR
                    detections.append(
                        DetectionResult(
                            bib_number="",
                            detection_conf=conf,
                            ocr_conf=0.0,
                            bbox=(x1, y1, x2, y2),
                        )
                    )
                else:
                    if not ocr_hits:
                        # OCR enabled but no matches - store placeholder
                        detections.append(
                            DetectionResult(
                                bib_number="",
                                detection_conf=conf,
                                ocr_conf=0.0,
                                bbox=(x1, y1, x2, y2),
                            )
                        )
                    else:
                        for bib_text, ocr_conf in ocr_hits:
                            detections.append(
                                DetectionResult(
                                    bib_number=bib_text,
                                    detection_conf=conf,
                                    ocr_conf=ocr_conf,
                                    bbox=(x1, y1, x2, y2),
                                )
                            )

                # Save crops if requested
                if args.save_crops and crop.size != 0:
                    args.save_crops.mkdir(parents=True, exist_ok=True)
                    crop_name = f"{img_path.stem}_{x1}_{y1}_{x2}_{y2}.jpg"
                    cv2.imwrite(str(args.save_crops / crop_name), crop)

        detections_per_image[str(img_path)] = detections
        total_detections_all += img_total_detections
        total_filtered_by_class_all += img_filtered_by_class
        
        # Debug: show low-confidence detections if no detections found
        if len(detections) == 0 and low_confidence_detections:
            low_conf_str = ", ".join([f"({c:.3f}, {bbox})" for c, bbox in low_confidence_detections[:3]])
            print(
                f"Debug: No detections above threshold ({args.conf}) for {img_path.name}. "
                f"Found {len(low_confidence_detections)} detections below threshold: {low_conf_str}",
                file=sys.stderr,
            )

        # Prepare CSV row aggregated by unique bib numbers
        bibs = [d.bib_number for d in detections if d.bib_number]
        unique_bibs = sorted(set(bibs))
        row = {
            "image": str(img_path),
            "bibs": ";".join(unique_bibs),
            "detections": len(detections),
        }
        rows.append(row)
        
        # Write incrementally if requested
        if args.incremental and csv_writer:
            csv_writer.writerow(row)
            csv_file.flush()  # Ensure it's written to disk
            
            # Also write JSON incrementally
            json_path = args.output.with_suffix(".json")
            # Read existing JSON if it exists
            if json_path.exists():
                with open(json_path, "r", encoding="utf-8") as f:
                    output_mapping = json.load(f)
            else:
                output_mapping = {}
            
            # Update with current image's detections
            output_mapping[str(img_path)] = {
                "detections": [
                    {
                        "bib": det.bib_number,
                        "detection_conf": det.detection_conf,
                        "ocr_conf": det.ocr_conf,
                        "bbox": det.bbox,
                    }
                    for det in detections
                ]
            }
            
            # Write updated JSON
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(output_mapping, f, indent=2, ensure_ascii=False)

        # Handle annotations
        if args.annotate_dir:
            annotate_path = args.annotate_dir / f"{img_path.stem}_yolo{img_path.suffix}"
            annotate_image(img, detections, annotate_path)

        # Store in database
        if storage:
            if args.image_url:
                rel_path = img_path.relative_to(args.input)
                image_url = args.image_url.rstrip("/") + "/" + str(rel_path).replace("\\", "/")
            else:
                image_url = str(img_path)

            # Use highest OCR confidence per bib
            bib_conf_map: Dict[str, float] = {}
            for det in detections:
                if det.bib_number:
                    cur = bib_conf_map.get(det.bib_number, 0.0)
                    bib_conf_map[det.bib_number] = max(cur, det.ocr_conf)

            storage.store_detection(
                image_url=image_url,
                bibs=list(bib_conf_map.keys()),
                confidences=list(bib_conf_map.values()),
                local_path=str(img_path),
            )

    # Write CSV (if not already written incrementally)
    if not args.incremental:
        df = pd.DataFrame(rows)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(args.output, index=False)
    else:
        # Close the incremental CSV file
        if csv_file:
            csv_file.close()

    # JSON mapping
    json_path = args.output.with_suffix(".json")
    output_mapping: Dict[str, Dict[str, object]] = {}
    for image, dets in detections_per_image.items():
        output_mapping[image] = {
            "detections": [
                {
                    "bib": det.bib_number,
                    "detection_conf": det.detection_conf,
                    "ocr_conf": det.ocr_conf,
                    "bbox": det.bbox,
                }
                for det in dets
            ]
        }

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(output_mapping, f, indent=2, ensure_ascii=False)

    print(f"Wrote CSV to {args.output}")
    print(f"Wrote detailed JSON to {json_path}")
    print(
        f"\nSummary: Processed {len(images)} images, "
        f"total YOLO detections: {total_detections_all}, "
        f"filtered by class_id: {total_filtered_by_class_all}, "
        f"final detections: {sum(len(d) for d in detections_per_image.values())}",
        file=sys.stderr,
    )
    if args.annotate_dir:
        print(f"Annotated images saved to {args.annotate_dir}")
    if args.save_crops:
        print(f"Cropped bib images saved to {args.save_crops}")


if __name__ == "__main__":
    main()
