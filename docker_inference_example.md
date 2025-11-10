# Docker Commands for Using Trained YOLO Model

## Important: Save Model During Training

If you used `--rm` during training, the model was saved inside the container and is now gone. For future training, mount the runs directory:

```bash
docker run --gpus all --rm --name bib-trainer --shm-size=2g \
  -v ${PWD}/data:/data \
  -v ${PWD}/runs:/app/runs \
  bibanalyser-train:latest train-yolo \
  --data /data/yolo_dataset --epochs 100 --imgsz 640 --batch 16 --name bib_detector
```

This saves the model to `./runs/detect/bib_detector/weights/best.pt` on your host.

## Using the Trained Model for Inference

### Option 1: If model is saved on host (recommended)

**With PaddleOCR cache (recommended - avoids re-downloading models):**
```bash
docker run --rm --name bib-inference \
  -v ${PWD}/data:/data \
  -v ${PWD}/runs:/app/runs \
  -v ${PWD}/.paddlex:/root/.paddlex \
  bibanalyser:latest yolo \
  --input /data/photos \
  --output /data/results.csv \
  --weights /app/runs/bib_detector/weights/best.pt \
  --conf 0.5 \
  --device cpu
```

**Without cache (models downloaded each time):**
```bash
docker run --rm --name bib-inference \
  -v ${PWD}/data:/data \
  -v ${PWD}/runs:/app/runs \
  bibanalyser:latest yolo \
  --input /data/photos \
  --output /data/results.csv \
  --weights /app/runs/bib_detector/weights/best.pt \
  --conf 0.5 \
  --device cpu
```

### Option 2: If model is still in a stopped container

First, copy the model from the container:
```bash
docker cp bib-trainer:/app/runs/detect/bib_detector/weights/best.pt ./runs/detect/bib_detector/weights/best.pt
```

Then use Option 1.

### Option 3: With GPU support (if available)

```bash
docker run --gpus all --rm --name bib-inference \
  -v ${PWD}/data:/data \
  -v ${PWD}/runs:/app/runs \
  -v ${PWD}/.paddlex:/root/.paddlex \
  bibanalyser:latest yolo \
  --input /data/photos \
  --output /data/results.csv \
  --weights /app/runs/bib_detector/weights/best.pt \
  --conf 0.5 \
  --device cuda:0
```

### With OCR recognition (PaddleOCR)

```bash
docker run --rm --name bib-inference \
  -v ${PWD}/data:/data \
  -v ${PWD}/runs:/app/runs \
  -v ${PWD}/.paddlex:/root/.paddlex \
  bibanalyser:latest yolo \
  --input /data/photos \
  --output /data/results.csv \
  --weights /app/runs/bib_detector/weights/best.pt \
  --conf 0.5 \
  --device cpu \
  --ocr-lang en
```

### Save annotated images

```bash
docker run --rm --name bib-inference \
  -v ${PWD}/data:/data \
  -v ${PWD}/runs:/app/runs \
  -v ${PWD}/.paddlex:/root/.paddlex \
  bibanalyser:latest yolo \
  --input /data/photos \
  --output /data/results.csv \
  --weights /app/runs/bib_detector/weights/best.pt \
  --conf 0.5 \
  --device cpu \
  --annotate-dir /data/annotated
```

## Notes

- **PaddleOCR Cache**: Add `-v ${PWD}/.paddlex:/root/.paddlex` to persist PaddleOCR models between runs (saves download time)
- **Model Path**: The model path is `/app/runs/bib_detector/weights/best.pt` (not `/app/runs/detect/bib_detector/weights/best.pt`)
- **Confidence**: Default is 0.3, but 0.5 is recommended for better accuracy (fewer false positives)

