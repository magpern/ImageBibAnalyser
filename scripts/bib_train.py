"""
Bib OCR Training/Validation Tool

Uses a set of images with known bib numbers to find optimal OCR parameters.
This helps "teach" the system what works best for your specific images.

Usage:
  python bib_train.py --input ./photos --ground-truth ground_truth.json --output best_params.json

Ground truth format (JSON):
  {
    "image1.jpg": ["254", "506", "133", "411"],
    "image2.jpg": ["1234"],
    "image3.jpg": []
  }

Or CSV format:
  image,bibs
  image1.jpg,"254;506;133;411"
  image2.jpg,"1234"
  image3.jpg,""
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Set, Optional
import itertools
from tqdm import tqdm

try:
    from bib_finder import (
        detect_bibs_for_image,
        build_bib_regex,
        gather_images,
    )
except ImportError:
    import sys

    sys.path.insert(0, str(Path(__file__).parent))
    from bib_finder import (
        detect_bibs_for_image,
        build_bib_regex,
        gather_images,
    )


def load_ground_truth(gt_path: Path) -> Dict[str, Set[str]]:
    """Load ground truth from JSON or CSV file."""
    gt: Dict[str, Set[str]] = {}

    if gt_path.suffix.lower() == ".json":
        with open(gt_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            for img, bibs in data.items():
                gt[img] = set(str(b) for b in bibs) if bibs else set()
    else:
        # CSV format
        import pandas as pd

        df = pd.read_csv(gt_path)
        for _, row in df.iterrows():
            img = str(row["image"])
            bibs_str = str(row.get("bibs", ""))
            if bibs_str and bibs_str != "nan":
                bibs = set(b.strip() for b in bibs_str.split(";") if b.strip())
            else:
                bibs = set()
            gt[img] = bibs

    return gt


def evaluate_params(
    image_paths: List[Path],
    ground_truth: Dict[str, Set[str]],
    bib_regex,
    min_conf: int,
    rotations: Tuple[int, ...],
    psm_values: Tuple[int, ...],
    focus_region: bool,
    min_text_size: float,
    max_text_size: float,
    show_progress: bool = False,
    use_gpu: bool = False,
) -> Dict[str, float]:
    """Evaluate a parameter set and return metrics."""
    true_positives = 0
    false_positives = 0
    false_negatives = 0
    total_gt_bibs = 0

    iterator = tqdm(
        image_paths,
        desc="  Processing images",
        leave=False,
        disable=not show_progress,
        mininterval=0.5,
    )
    for img_path in iterator:
        img_name = img_path.name
        gt_bibs = ground_truth.get(img_name, set())
        total_gt_bibs += len(gt_bibs)

        # Update progress bar description to show current image
        if show_progress:
            iterator.set_description(f"  Processing: {img_name[:30]}...")

        try:
            _, detected_bibs, _ = detect_bibs_for_image(
                img_path,
                bib_regex,
                min_conf,
                rotations,
                psm_values,
                annotate_dir=None,
                focus_region=focus_region,
                min_text_size=min_text_size,
                max_text_size=max_text_size,
                use_gpu=use_gpu,
            )
        except Exception as e:
            # Log error but continue with other images
            if show_progress:
                tqdm.write(f"  Error processing {img_name}: {e}")
            detected_bibs = []

        detected_set = set(detected_bibs)

        # Calculate metrics
        tp = len(detected_set & gt_bibs)
        fp = len(detected_set - gt_bibs)
        fn = len(gt_bibs - detected_set)

        true_positives += tp
        false_positives += fp
        false_negatives += fn

    # Calculate precision, recall, F1
    precision = (
        true_positives / (true_positives + false_positives)
        if (true_positives + false_positives) > 0
        else 0.0
    )
    recall = (
        true_positives / (true_positives + false_negatives)
        if (true_positives + false_negatives) > 0
        else 0.0
    )
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "true_positives": true_positives,
        "false_positives": false_positives,
        "false_negatives": false_negatives,
        "total_gt": total_gt_bibs,
    }


def main():
    ap = argparse.ArgumentParser(
        description="Train/validate OCR parameters using images with known bib numbers"
    )
    ap.add_argument("--input", required=True, type=Path, help="Folder with images")
    ap.add_argument(
        "--ground-truth",
        required=True,
        type=Path,
        help="Ground truth file (JSON or CSV) with known image->bibs mappings",
    )
    ap.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output file for best parameters (JSON). If not specified, prints to stdout.",
    )
    ap.add_argument("--min-digits", type=int, default=2, help="Minimum bib digits (default: 2)")
    ap.add_argument("--max-digits", type=int, default=6, help="Maximum bib digits (default: 6)")
    ap.add_argument(
        "--bib-pattern",
        type=str,
        default=None,
        help="Custom regex pattern for bib numbers",
    )
    ap.add_argument(
        "--ext",
        nargs="+",
        default=(".jpg", ".jpeg", ".png"),
        help="Image extensions to include",
    )
    ap.add_argument(
        "--verbose",
        action="store_true",
        help="Print detailed results for each parameter combination",
    )
    ap.add_argument(
        "--use-gpu",
        action="store_true",
        help="Use GPU acceleration for image preprocessing (requires OpenCV with CUDA or CuPy). "
        "Note: Tesseract OCR is CPU-only, so GPU mainly speeds up preprocessing. "
        "Can provide 2-5x speedup for preprocessing, but OCR remains the bottleneck.",
    )

    args = ap.parse_args()

    # Load ground truth
    if not args.ground_truth.exists():
        print(f"Error: Ground truth file not found: {args.ground_truth}")
        sys.exit(1)

    print(f"Loading ground truth from {args.ground_truth}...")
    ground_truth = load_ground_truth(args.ground_truth)
    print(f"Loaded {len(ground_truth)} images with known bib numbers")

    # Gather images
    exts = tuple(args.ext)
    all_images = gather_images(args.input, exts)
    # Filter to only images in ground truth
    image_paths = [img for img in all_images if img.name in ground_truth]
    print(f"Found {len(image_paths)} images matching ground truth")

    if not image_paths:
        print("Error: No images found that match ground truth entries")
        sys.exit(1)

    # Build regex
    if args.bib_pattern:
        import re

        bib_regex = re.compile(args.bib_pattern)
    else:
        bib_regex = build_bib_regex(args.min_digits, args.max_digits)

    # Define parameter search space
    min_conf_values = [40, 50, 60, 70]
    psm_combinations = [
        (6, 7, 11),
        (6, 7, 8, 11),
        (6, 7, 8, 11, 13),
        (8, 13),  # Single word/line modes
    ]
    rotation_combinations = [
        (0, 90, -90, 180),
        (0, 45, 90, -45, -90, 135, -135, 180),
    ]
    focus_region_values = [True, False]
    min_text_size_values = [0.005, 0.01, 0.02]
    max_text_size_values = [0.2, 0.3, 0.4]

    print("\nTesting parameter combinations...")
    print("This may take a while depending on the number of images...\n")

    best_params = None
    best_f1 = -1.0
    results = []

    total_combinations = (
        len(min_conf_values)
        * len(psm_combinations)
        * len(rotation_combinations)
        * len(focus_region_values)
        * len(min_text_size_values)
        * len(max_text_size_values)
    )
    print(f"Testing {total_combinations} parameter combinations on {len(image_paths)} images...")
    print(f"This may take a while. Progress will be shown below.\n")

    # Create all combinations as a list for progress tracking
    all_combinations = list(
        itertools.product(
            min_conf_values,
            psm_combinations,
            rotation_combinations,
            focus_region_values,
            min_text_size_values,
            max_text_size_values,
        )
    )

    # Use tqdm for progress bar
    print(f"\nStarting evaluation of {len(all_combinations)} combinations...")
    print("First combination may take 1-2 minutes to process all images.\n")

    with tqdm(
        total=len(all_combinations), desc="Testing combinations", unit="comb", mininterval=1.0
    ) as pbar:
        for combo_idx, (
            min_conf,
            psm_vals,
            rotations,
            focus_region,
            min_text_size,
            max_text_size,
        ) in enumerate(all_combinations, 1):
            # Update progress bar description with current parameters
            best_f1_display = best_f1 if best_f1 > 0 else 0.0
            pbar.set_description(
                f"Testing: conf={min_conf}, psm={len(psm_vals)} modes, focus={focus_region}, "
                f"best F1={best_f1_display:.3f}"
            )

            # Show start of first combination
            if combo_idx == 1:
                tqdm.write(f"\n[1/{len(all_combinations)}] Starting first combination...")
                tqdm.write(f"  Parameters: conf={min_conf}, psm={psm_vals}, rotations={rotations}")
                tqdm.write(
                    f"  Processing {len(image_paths)} images (this may take 1-2 minutes)...\n"
                )

            try:
                metrics = evaluate_params(
                    image_paths,
                    ground_truth,
                    bib_regex,
                    min_conf,
                    rotations,
                    psm_vals,
                    focus_region,
                    min_text_size,
                    max_text_size,
                    show_progress=args.verbose
                    or (
                        combo_idx == 1
                    ),  # Always show progress for first combo, then respect verbose
                    use_gpu=getattr(args, "use_gpu", False),
                )
            except Exception as e:
                tqdm.write(f"\nError in combination {combo_idx}: {e}")
                # Use default metrics for failed combination
                metrics = {
                    "precision": 0.0,
                    "recall": 0.0,
                    "f1": 0.0,
                    "true_positives": 0,
                    "false_positives": 0,
                    "false_negatives": 0,
                    "total_gt": 0,
                }

            params = {
                "min_conf": min_conf,
                "psm": list(psm_vals),
                "rotations": list(rotations),
                "focus_region": focus_region,
                "min_text_size": min_text_size,
                "max_text_size": max_text_size,
            }

            result = {**params, **metrics}
            results.append(result)

            if args.verbose:
                tqdm.write(
                    f"Conf={min_conf:2d} PSM={psm_vals} Focus={focus_region} "
                    f"Size=[{min_text_size:.3f},{max_text_size:.3f}] "
                    f"→ Precision={metrics['precision']:.3f} Recall={metrics['recall']:.3f} "
                    f"F1={metrics['f1']:.3f}"
                )

            if metrics["f1"] > best_f1:
                best_f1 = metrics["f1"]
                best_params = params
                best_metrics = metrics
                # Show when we find a new best
                tqdm.write(
                    f"✨ New best! F1={best_f1:.3f} (P={metrics['precision']:.3f}, R={metrics['recall']:.3f}) "
                    f"with conf={min_conf}, psm={psm_vals}, focus={focus_region}"
                )

            pbar.update(1)

    # Print results
    print("\n" + "=" * 80)
    print("BEST PARAMETERS:")
    print("=" * 80)
    print(json.dumps(best_params, indent=2))
    print("\nMetrics:")
    print(f"  Precision: {best_metrics['precision']:.3f}")
    print(f"  Recall:    {best_metrics['recall']:.3f}")
    print(f"  F1 Score:  {best_metrics['f1']:.3f}")
    print(
        f"  TP: {best_metrics['true_positives']}, FP: {best_metrics['false_positives']}, FN: {best_metrics['false_negatives']}"
    )

    # Show top 5 results
    results_sorted = sorted(results, key=lambda x: x["f1"], reverse=True)
    print("\n" + "=" * 80)
    print("TOP 5 PARAMETER COMBINATIONS:")
    print("=" * 80)
    for i, r in enumerate(results_sorted[:5], 1):
        print(f"\n{i}. F1={r['f1']:.3f} (P={r['precision']:.3f}, R={r['recall']:.3f})")
        print(f"   min_conf={r['min_conf']}, psm={r['psm']}, focus_region={r['focus_region']}")
        print(f"   min_text_size={r['min_text_size']:.3f}, max_text_size={r['max_text_size']:.3f}")

    # Save best parameters
    if args.output:
        output_data = {
            "best_parameters": best_params,
            "metrics": best_metrics,
            "top_5": [
                {
                    "parameters": {
                        k: v
                        for k, v in r.items()
                        if k
                        not in [
                            "precision",
                            "recall",
                            "f1",
                            "true_positives",
                            "false_positives",
                            "false_negatives",
                            "total_gt",
                        ]
                    },
                    "metrics": {
                        k: v
                        for k, v in r.items()
                        if k
                        in [
                            "precision",
                            "recall",
                            "f1",
                            "true_positives",
                            "false_positives",
                            "false_negatives",
                            "total_gt",
                        ]
                    },
                }
                for r in results_sorted[:5]
            ],
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(output_data, f, indent=2)
        print(f"\nSaved best parameters to {args.output}")
        print("\nTo use these parameters, run:")
        print(f"  racebib ocr --input <folder> --output <csv> \\")
        print(f"    --min-conf {best_params['min_conf']} \\")
        print(f"    --psm {' '.join(map(str, best_params['psm']))} \\")
        print(f"    --rotations {' '.join(map(str, best_params['rotations']))} \\")
        if not best_params["focus_region"]:
            print(f"    --no-focus-region \\")
        print(f"    --min-text-size {best_params['min_text_size']} \\")
        print(f"    --max-text-size {best_params['max_text_size']}")


if __name__ == "__main__":
    main()
