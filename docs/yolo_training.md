# Training YOLO for Bib Detection

This guide walks you through training a YOLO model to detect race bib numbers in images.

## Overview

Training a YOLO model requires:
1. **Labeled images** - Images with bounding boxes around bib numbers
2. **Training script** - Using Ultralytics YOLO (YOLO11 recommended)
3. **Validation** - Testing the model on new images

## Docker vs Local Training

**Docker Training (Recommended if you have NVIDIA GPU):**
- Use `Dockerfile.train` for GPU-accelerated training
- Requires nvidia-docker or Docker with GPU support
- Includes all dependencies pre-configured

**Local Training:**
- Requires manual setup of PyTorch with CUDA
- More flexible for development
- See requirements in `requirements.txt`

## Step 1: Prepare Your Training Data

### Option A: Use Labeling Tools (Recommended)

**LabelImg** (Free, cross-platform):
1. Download from: https://github.com/HumanSignal/labelImg/releases
2. Install and open
3. Open directory with your race photos
4. Draw bounding boxes around each bib number
5. Save as YOLO format (creates `.txt` files alongside images)

**CVAT** (Web-based, more features):
- https://cvat.org/
- Better for teams, supports video annotation

**Roboflow** (Cloud-based, easiest):
- https://roboflow.com/
- Free tier available, auto-format conversion

### Option B: Use the Helper Script

We provide a helper script to convert existing annotations or create a dataset structure:

```bash
python scripts/prepare_yolo_dataset.py --input ./photos --output ./yolo_dataset
```

## Step 2: Organize Your Dataset

Your dataset should look like this:

```
yolo_dataset/
├── train/
│   ├── images/
│   │   ├── image1.jpg
│   │   ├── image2.jpg
│   │   └── ...
│   └── labels/
│       ├── image1.txt
│       ├── image2.txt
│       └── ...
├── val/
│   ├── images/
│   └── labels/
└── test/
    ├── images/
    └── labels/
```

**Label file format** (YOLO):
```
class_id center_x center_y width height
```

For bib detection, you typically have one class (bib = class 0):
```
0 0.5 0.5 0.2 0.3
```

Where coordinates are normalized (0-1) relative to image size.

## Step 3: Train the Model

### Quick Start (Using our helper script)

**Using Docker (with GPU support):**
```bash
# Option 1: Build training image locally
docker build -f Dockerfile.train -t bibanalyser-train:latest .

# Option 2: Pull from GitHub Container Registry (after pushing)
docker pull ghcr.io/magpern/ImageBibAnalyser-train:latest

# Train with GPU
docker run --gpus all --rm -v ${PWD}/data:/data \
  ghcr.io/magpern/ImageBibAnalyser-train:latest \
  python -m scripts.train_yolo \
  --data /data/yolo_dataset \
  --epochs 100 \
  --imgsz 640 \
  --batch 16 \
  --name bib_detector \
  --device 0
```

**Using Local Python:**
```bash
# Train with default settings
python scripts/train_yolo.py \
  --data ./yolo_dataset \
  --epochs 100 \
  --imgsz 640 \
  --batch 16 \
  --name bib_detector
```

### Manual Training (Using Ultralytics directly)

```python
from ultralytics import YOLO

# Load a pre-trained model (YOLO11 recommended - better accuracy and speed)
# YOLO11: yolo11n.pt (nano), yolo11s.pt (small), yolo11m.pt (medium), yolo11l.pt (large), yolo11x.pt (xlarge)
# YOLOv8: yolov8n.pt, yolov8s.pt, yolov8m.pt, yolov8l.pt, yolov8x.pt (older, still supported)
model = YOLO('yolo11n.pt')  # Start with YOLO11 nano for faster training

# Train the model
results = model.train(
    data='./yolo_dataset',  # Path to dataset
    epochs=100,            # Number of training epochs
    imgsz=640,             # Image size
    batch=16,              # Batch size (adjust based on GPU memory)
    name='bib_detector',   # Project name
    project='./runs',      # Project directory
    device=0,              # GPU device (0, 1, etc.) or 'cpu'
    patience=50,           # Early stopping patience
    save=True,             # Save checkpoints
    plots=True,            # Generate training plots
)

# Validate
metrics = model.val()
print(f"mAP50: {metrics.box.map50}")
print(f"mAP50-95: {metrics.box.map}")
```

### Training Options

**Model size** (choose based on speed vs accuracy):

**YOLO11 (Recommended - better accuracy and speed):**
- `yolo11n.pt` - Nano (fastest, smallest, lower accuracy)
- `yolo11s.pt` - Small (good balance)
- `yolo11m.pt` - Medium (better accuracy)
- `yolo11l.pt` - Large (high accuracy)
- `yolo11x.pt` - XLarge (best accuracy, slowest)

**YOLOv8 (Older, still supported):**
- `yolov8n.pt` - Nano
- `yolov8s.pt` - Small
- `yolov8m.pt` - Medium
- `yolov8l.pt` - Large
- `yolov8x.pt` - XLarge

**Note:** YOLO11 offers ~2% higher accuracy and ~22% fewer parameters (faster inference) compared to YOLOv8.

**Batch size** (adjust based on GPU memory):
- GPU with 4GB: `batch=8`
- GPU with 8GB: `batch=16`
- GPU with 16GB+: `batch=32`
- CPU: `batch=4`

**Image size**:
- `imgsz=640` - Standard (good balance)
- `imgsz=1280` - Higher resolution (better for small bibs, slower)

## Step 4: Validate and Test

After training, the model is saved in `./runs/detect/bib_detector/weights/best.pt`

Test it on your images:

```bash
# Using our CLI
racebib yolo --input ./test_photos --output ./test_results.csv \
  --weights ./runs/detect/bib_detector/weights/best.pt \
  --annotate-dir ./test_annotated

# Or using Ultralytics directly
from ultralytics import YOLO
model = YOLO('./runs/detect/bib_detector/weights/best.pt')
results = model('./test_photos', save=True, conf=0.3)
```

## Step 5: Improve Your Model

### If detection is poor:

1. **More training data** - Add more labeled images (aim for 100+ minimum)
2. **Better labels** - Ensure bounding boxes are tight around bibs
3. **Data augmentation** - Ultralytics does this automatically, but you can adjust:
   ```python
   model.train(
       data='./yolo_dataset',
       hsv_h=0.015,  # Hue augmentation
       hsv_s=0.7,    # Saturation augmentation
       hsv_v=0.4,    # Value augmentation
       degrees=10,   # Rotation augmentation
       translate=0.1, # Translation augmentation
       scale=0.5,    # Scale augmentation
   )
   ```
4. **Longer training** - Increase epochs (200-300)
5. **Larger model** - Try yolo11m or yolo11l instead of yolo11n (or yolov8m/yolov8l if using YOLOv8)

### If you have too many false positives:

1. **Increase confidence threshold** - Use `--conf 0.5` or higher
2. **More negative examples** - Add images without bibs to training set
3. **Better labeling** - Ensure you're not including non-bib text

## Example: Complete Training Workflow

```bash
# 1. Prepare dataset (if using helper script)
python scripts/prepare_yolo_dataset.py --input ./photos --output ./yolo_dataset

# 2. Train model
python scripts/train_yolo.py \
  --data ./yolo_dataset \
  --epochs 100 \
  --imgsz 640 \
  --batch 16 \
  --name bib_detector \
  --device 0

# 3. Test on validation set
racebib yolo --input ./yolo_dataset/val/images \
  --output ./val_results.csv \
  --weights ./runs/detect/bib_detector/weights/best.pt \
  --annotate-dir ./val_annotated

# 4. Use on new images
racebib yolo --input ./new_photos \
  --output ./results.csv \
  --weights ./runs/detect/bib_detector/weights/best.pt \
  --db bibdb.json \
  --image-url https://example.com/gallery/
```

## Tips for Better Results

1. **Diversity in training data**:
   - Different lighting conditions
   - Different angles/distances
   - Different bib colors/styles
   - Different backgrounds

2. **Label quality matters**:
   - Tight bounding boxes (not too loose)
   - Include partially visible bibs
   - Consistent labeling across images

3. **Start small, scale up**:
   - Begin with 50-100 images
   - Train for 50 epochs
   - Test and iterate
   - Add more data as needed

4. **Monitor training**:
   - Watch validation mAP (should increase)
   - Check training plots in `./runs/detect/bib_detector/`
   - Stop if overfitting (validation mAP decreases)

## Using Pre-trained Models

If you don't want to train from scratch, you can:
1. Use a pre-trained YOLO model and fine-tune on your data
2. Look for publicly available bib detection models
3. Use transfer learning from similar object detection tasks

## Resources

- **Ultralytics YOLOv8 Docs**: https://docs.ultralytics.com
- **LabelImg**: https://github.com/HumanSignal/labelImg
- **Roboflow**: https://roboflow.com (free dataset hosting and augmentation)
- **YOLO Paper**: https://arxiv.org/abs/2304.00501

