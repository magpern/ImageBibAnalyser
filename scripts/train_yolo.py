"""
Train YOLO model for bib detection.

This script provides a convenient wrapper around Ultralytics YOLO training.
Defaults to YOLO11 (newer, better accuracy and speed) but supports YOLOv8 for compatibility.

Usage:
    python train_yolo.py --data ./yolo_dataset --epochs 100 --name bib_detector

Requirements:
    pip install ultralytics
"""

import argparse
import sys
from pathlib import Path

try:
    from ultralytics import YOLO
except ImportError:
    print("Error: ultralytics not installed. Install with: pip install ultralytics")
    sys.exit(1)


def main():
    ap = argparse.ArgumentParser(
        description="Train YOLO model for bib detection",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument(
        "--data",
        required=True,
        type=Path,
        help="Path to YOLO dataset directory (should contain train/val folders with images/ and labels/ subfolders)",
    )
    ap.add_argument(
        "--model",
        type=str,
        default="yolo11n.pt",
        help="Pre-trained model to start from (yolo11n.pt, yolo11s.pt, yolo11m.pt, yolo11l.pt, yolo11x.pt for YOLO11, or yolov8n.pt, yolov8s.pt, etc. for YOLOv8)",
    )
    ap.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Number of training epochs",
    )
    ap.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Image size for training (640, 1280, etc.)",
    )
    ap.add_argument(
        "--batch",
        type=int,
        default=16,
        help="Batch size (adjust based on GPU memory: 8 for 4GB, 16 for 8GB, 32 for 16GB+)",
    )
    ap.add_argument(
        "--device",
        type=str,
        default="0",
        help="Device to use (0, 1 for GPU, 'cpu' for CPU)",
    )
    ap.add_argument(
        "--name",
        type=str,
        default="bib_detector",
        help="Project name (model will be saved in runs/detect/<name>/weights/best.pt)",
    )
    ap.add_argument(
        "--project",
        type=Path,
        default=Path("./runs"),
        help="Project directory",
    )
    ap.add_argument(
        "--patience",
        type=int,
        default=50,
        help="Early stopping patience (stop if no improvement for N epochs)",
    )
    ap.add_argument(
        "--workers",
        type=int,
        default=2,
        help="Number of data loading workers (reduced for Docker shared memory limits)",
    )

    args = ap.parse_args()

    # Validate dataset structure
    data_path = Path(args.data)
    if not data_path.exists():
        print(f"Error: Dataset directory not found: {data_path}")
        sys.exit(1)

    train_images = data_path / "train" / "images"
    train_labels = data_path / "train" / "labels"
    val_images = data_path / "val" / "images"
    val_labels = data_path / "val" / "labels"

    if not train_images.exists() or not train_labels.exists():
        print(f"Error: Dataset structure incorrect. Expected:")
        print(f"  {data_path}/train/images/")
        print(f"  {data_path}/train/labels/")
        print(f"  {data_path}/val/images/")
        print(f"  {data_path}/val/labels/")
        sys.exit(1)

    train_img_count = len(list(train_images.glob("*"))) if train_images.exists() else 0
    val_img_count = len(list(val_images.glob("*"))) if val_images.exists() else 0

    if train_img_count == 0:
        print(f"Error: No training images found in {train_images}")
        sys.exit(1)

    print(f"Dataset found:")
    print(f"  Training images: {train_img_count}")
    print(f"  Validation images: {val_img_count}")
    print(f"\nStarting training...")
    print(f"  Model: {args.model}")
    print(f"  Epochs: {args.epochs}")
    print(f"  Image size: {args.imgsz}")
    print(f"  Batch size: {args.batch}")
    print(f"  Device: {args.device}")
    print()

    # Load model
    try:
        model = YOLO(args.model)
    except Exception as e:
        print(f"Error loading model {args.model}: {e}")
        print("Available YOLO11 models: yolo11n.pt, yolo11s.pt, yolo11m.pt, yolo11l.pt, yolo11x.pt")
        print("Available YOLOv8 models: yolov8n.pt, yolov8s.pt, yolov8m.pt, yolov8l.pt, yolov8x.pt")
        print("Note: YOLO11 is recommended (better accuracy and speed)")
        sys.exit(1)

    # Check for data.yaml file (Ultralytics expects YAML file path, not directory)
    data_yaml = data_path / "data.yaml"
    if data_yaml.exists():
        data_arg = str(data_yaml)
    else:
        # Fallback: use directory path (Ultralytics might handle it)
        data_arg = str(data_path)
        print(f"Warning: data.yaml not found in {data_path}, using directory path")

    # Train
    try:
        results = model.train(
            data=data_arg,
            epochs=args.epochs,
            imgsz=args.imgsz,
            batch=args.batch,
            name=args.name,
            project=str(args.project),
            device=args.device,
            patience=args.patience,
            workers=args.workers,
            save=True,
            plots=True,
            verbose=True,
        )

        print("\n" + "=" * 80)
        print("Training completed!")
        print("=" * 80)
        print(f"\nBest model saved to: {args.project}/detect/{args.name}/weights/best.pt")
        print(f"Last model saved to: {args.project}/detect/{args.name}/weights/last.pt")
        print(f"\nTraining results: {args.project}/detect/{args.name}/")

        # Show final metrics
        if hasattr(results, "results_dict"):
            metrics = results.results_dict
            print(f"\nFinal Metrics:")
            if "metrics/mAP50(B)" in metrics:
                print(f"  mAP50: {metrics['metrics/mAP50(B)']:.4f}")
            if "metrics/mAP50-95(B)" in metrics:
                print(f"  mAP50-95: {metrics['metrics/mAP50-95(B)']:.4f}")

        print(f"\nTo use this model:")
        print(
            f"  racebib yolo --input <folder> --output <csv> --weights {args.project}/detect/{args.name}/weights/best.pt"
        )

    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user.")
        print(f"Partial model saved to: {args.project}/detect/{args.name}/weights/last.pt")
        sys.exit(1)
    except Exception as e:
        print(f"\nError during training: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
