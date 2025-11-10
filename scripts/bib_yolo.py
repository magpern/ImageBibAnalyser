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
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
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

    # Validate device
    device = device.lower()
    if device == "cuda" and not model.device.type.startswith("cuda"):
        print("Warning: CUDA requested but not available; running on CPU instead.", file=sys.stderr)
    return model


def init_paddle_ocr(use_gpu: bool, lang: str) -> PaddleOCR | None:
    if PaddleOCR is None:
        raise RuntimeError(
            "PaddleOCR is not installed. Install with 'pip install paddleocr paddlepaddle'."
        )
    try:
        ocr = PaddleOCR(
            use_angle_cls=True,
            use_gpu=use_gpu,
            lang=lang,
            det=False,
            rec=True,
        )
        return ocr
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(f"Failed to initialize PaddleOCR: {exc}") from exc


def crop_image(img: np.ndarray, bbox: Tuple[int, int, int, int]) -> np.ndarray:
    x1, y1, x2, y2 = bbox
    h, w = img.shape[:2]
    x1 = max(0, min(w - 1, x1))
    x2 = max(0, min(w, x2))
    y1 = max(0, min(h - 1, y1))
    y2 = max(0, min(h, y2))
    if x2 <= x1 or y2 <= y1:
        return img[0:0, 0:0]
    return img[y1:y2, x1:x2]


def run_ocr_on_crop(
    ocr: PaddleOCR,
    crop: np.ndarray,
    bib_regex: Optional[re.Pattern],
    min_digits: int,
    max_digits: int,
) -> List[Tuple[str, float]]:
    if crop.size == 0:
        return []

    # PaddleOCR expects BGR images; ensure copy to contiguous array
    results = ocr.ocr(crop, det=False, rec=True, cls=False)  # type: ignore[arg-type]
    texts: List[Tuple[str, float]] = []
    if not results:
        return texts

    # Results structure: [[(text, (confidence))]]
    for res_group in results:
        for res in res_group:
            text = res[0]
            conf = float(res[1])
            if bib_regex:
                matches = list(bib_regex.finditer(text))
                if not matches:
                    continue
                for match in matches:
                    texts.append((match.group(0), conf))
            else:
                digits = re.findall(rf"\d{{{min_digits},{max_digits}}}", text)
                for d in digits:
                    texts.append((d, conf))
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
    ap = argparse.ArgumentParser(description="YOLO-based Bib Detector with optional OCR recognition")
    ap.add_argument("--input", required=True, type=Path, help="Folder with images")
    ap.add_argument("--output", required=True, type=Path, help="Output CSV path")
    ap.add_argument("--weights", required=True, type=Path, help="YOLO weights file trained for bib detection")
    ap.add_argument("--ext", nargs="+", default=(".jpg", ".jpeg", ".png"), help="Image extensions to include")
    ap.add_argument("--conf", type=float, default=0.3, help="YOLO confidence threshold (default: 0.3)")
    ap.add_argument("--iou", type=float, default=0.45, help="YOLO IoU threshold for NMS (default: 0.45)")
    ap.add_argument("--device", type=str, default="cpu", help="Device to run YOLO on (cpu, cuda, cuda:0, etc.)")
    ap.add_argument("--class-id", type=int, default=0, help="YOLO class ID corresponding to bibs (default: 0)")
    ap.add_argument(
        "--bib-pattern",
        type=str,
        default=None,
        help="Custom regex pattern for bib numbers (overrides min/max digits)",
    )
    ap.add_argument("--min-digits", type=int, default=2, help="Minimum bib digits if regex not provided")
    ap.add_argument("--max-digits", type=int, default=6, help="Maximum bib digits if regex not provided")
    ap.add_argument("--disable-ocr", action="store_true", help="Disable OCR and only output detections")
    ap.add_argument(
        "--ocr-lang",
        type=str,
        default="en",
        help="PaddleOCR language code (default: en). See PaddleOCR docs for options.",
    )
    ap.add_argument("--use-gpu", action="store_true", help="Use GPU for PaddleOCR if available")
    ap.add_argument("--annotate-dir", type=Path, default=None, help="Save annotated images with detections here")
    ap.add_argument("--save-crops", type=Path, default=None, help="If set, save cropped bib regions to this folder")
    ap.add_argument("--db", type=Path, default=None, help="Path to bib storage database (JSON file)")
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
        print("Warning: --disable-ocr is set, database will only store detection metadata.", file=sys.stderr)

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

    # Setup storage/database
    storage: Optional[BibStorage] = None
    if args.db:
        storage = BibStorage(args.db)
        print(f"Using storage database: {args.db}")

    rows: List[Dict[str, object]] = []
    detections_per_image: Dict[str, List[DetectionResult]] = {}

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
        for res in results:
            if res.boxes is None:
                continue
            for box in res.boxes:
                cls_id = int(box.cls.item())
                if cls_id != args.class_id:
                    continue
                conf = float(box.conf.item())
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                crop = crop_image(img, (x1, y1, x2, y2))
                ocr_hits: List[Tuple[str, float]] = []
                if ocr is not None:
                    ocr_hits = run_ocr_on_crop(ocr, crop, bib_regex, args.min_digits, args.max_digits)

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

        # Prepare CSV row aggregated by unique bib numbers
        bibs = [d.bib_number for d in detections if d.bib_number]
        unique_bibs = sorted(set(bibs))
        rows.append(
            {
                "image": str(img_path),
                "bibs": ";".join(unique_bibs),
                "detections": len(detections),
            }
        )

        # Handle annotations
        if args.annotate_dir:
            annotate_path = args.annotate_dir / f"{img_path.stem}_yolo{img_path.suffix}"
            annotate_image(img, detections, annotate_path)

        # Store in database
        if storage:
            if args.image_url:
                rel_path = img_path.relative_to(args.input)
                image_url = (
                    args.image_url.rstrip("/") + "/" + str(rel_path).replace("\\", "/")
                )
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

    # Write CSV
    df = pd.DataFrame(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output, index=False)

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
    if args.annotate_dir:
        print(f"Annotated images saved to {args.annotate_dir}")
    if args.save_crops:
        print(f"Cropped bib images saved to {args.save_crops}")


if __name__ == "__main__":
    main()


