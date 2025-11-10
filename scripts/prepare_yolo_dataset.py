"""
Helper script to prepare YOLO dataset structure.

This script helps organize images and labels into the YOLO format.

Usage:
    # Create dataset structure from existing images
    python prepare_yolo_dataset.py --input ./photos --output ./yolo_dataset

    # Split existing dataset into train/val
    python prepare_yolo_dataset.py --input ./yolo_dataset --split 0.8
"""

import argparse
import random
import shutil
import sys
from pathlib import Path
from typing import List, Tuple


def create_dataset_structure(output_dir: Path):
    """Create YOLO dataset directory structure."""
    for split in ["train", "val", "test"]:
        (output_dir / split / "images").mkdir(parents=True, exist_ok=True)
        (output_dir / split / "labels").mkdir(parents=True, exist_ok=True)


def split_dataset(
    input_dir: Path, output_dir: Path, train_ratio: float = 0.8, val_ratio: float = 0.1
):
    """Split images and labels into train/val/test sets."""
    images_dir = input_dir / "images"
    labels_dir = input_dir / "labels"

    if not images_dir.exists():
        print(f"Error: Images directory not found: {images_dir}")
        sys.exit(1)

    # Get all image files
    image_extensions = {".jpg", ".jpeg", ".png", ".bmp"}
    image_files = [f for f in images_dir.iterdir() if f.suffix.lower() in image_extensions]

    if not image_files:
        print(f"Error: No images found in {images_dir}")
        sys.exit(1)

    # Shuffle
    random.shuffle(image_files)

    # Calculate split sizes
    total = len(image_files)
    train_count = int(total * train_ratio)
    val_count = int(total * val_ratio)
    test_count = total - train_count - val_count

    print(f"Found {total} images")
    print(f"  Train: {train_count} ({train_count/total*100:.1f}%)")
    print(f"  Val: {val_count} ({val_count/total*100:.1f}%)")
    print(f"  Test: {test_count} ({test_count/total*100:.1f}%)")

    # Create structure
    create_dataset_structure(output_dir)

    # Copy files
    splits = [
        ("train", image_files[:train_count]),
        ("val", image_files[train_count : train_count + val_count]),
        ("test", image_files[train_count + val_count :]),
    ]

    for split_name, files in splits:
        print(f"\nCopying {split_name} set...")
        for img_file in files:
            # Copy image
            shutil.copy2(img_file, output_dir / split_name / "images" / img_file.name)

            # Copy corresponding label if exists
            label_file = labels_dir / f"{img_file.stem}.txt"
            if label_file.exists():
                shutil.copy2(label_file, output_dir / split_name / "labels" / label_file.name)
            else:
                # Create empty label file if no label exists
                (output_dir / split_name / "labels" / f"{img_file.stem}.txt").touch()

    print(f"\nDataset structure created in: {output_dir}")


def create_dataset_config(output_dir: Path, num_classes: int = 1, class_names: List[str] = None):
    """Create dataset.yaml file for YOLO training."""
    if class_names is None:
        class_names = ["bib"]

    config = {
        "path": str(output_dir.absolute()),
        "train": "train/images",
        "val": "val/images",
        "test": "test/images",
        "nc": num_classes,
        "names": class_names,
    }

    import yaml

    yaml_path = output_dir / "dataset.yaml"
    with open(yaml_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False)

    print(f"Dataset config created: {yaml_path}")
    return yaml_path


def main():
    ap = argparse.ArgumentParser(
        description="Prepare YOLO dataset structure",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Input directory with images (and optionally labels/)",
    )
    ap.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output directory for YOLO dataset structure",
    )
    ap.add_argument(
        "--split",
        type=float,
        default=0.8,
        help="Train/val split ratio (e.g., 0.8 = 80%% train, 20%% val/test)",
    )
    ap.add_argument(
        "--val-ratio",
        type=float,
        default=0.1,
        help="Validation ratio (rest goes to test)",
    )
    ap.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible splits",
    )
    ap.add_argument(
        "--classes",
        nargs="+",
        default=["bib"],
        help="Class names (default: bib)",
    )

    args = ap.parse_args()

    random.seed(args.seed)

    # Check if input already has YOLO structure
    input_images = args.input / "images"
    input_labels = args.input / "labels"

    if input_images.exists() and input_labels.exists():
        # Already in YOLO format, just split
        print("Input appears to be in YOLO format. Splitting dataset...")
        split_dataset(args.input, args.output, args.split, args.val_ratio)
    else:
        # Assume flat structure with images
        print("Creating YOLO dataset structure from flat image directory...")
        print("Note: You'll need to create label files (.txt) for each image.")
        create_dataset_structure(args.output)

        # Copy all images to train (user can manually organize later)
        image_extensions = {".jpg", ".jpeg", ".png", ".bmp"}
        image_files = [
            f for f in args.input.iterdir() if f.is_file() and f.suffix.lower() in image_extensions
        ]

        if not image_files:
            print(f"Error: No images found in {args.input}")
            sys.exit(1)

        print(f"Found {len(image_files)} images")
        print("Copying to train/ directory (you can reorganize later)...")

        for img_file in image_files:
            shutil.copy2(img_file, args.output / "train" / "images" / img_file.name)
            # Create empty label file
            (args.output / "train" / "labels" / f"{img_file.stem}.txt").touch()

        print(f"\nDataset structure created in: {args.output}")
        print("\nNext steps:")
        print("1. Label your images using LabelImg (https://github.com/HumanSignal/labelImg)")
        print("2. Save labels in YOLO format to the labels/ directories")
        print("3. Split into train/val/test using: --split option")

    # Create dataset.yaml
    try:
        import yaml

        create_dataset_config(args.output, len(args.classes), args.classes)
    except ImportError:
        print("\nWarning: PyYAML not installed. Install with: pip install pyyaml")
        print("You'll need to create dataset.yaml manually for training.")


if __name__ == "__main__":
    main()
