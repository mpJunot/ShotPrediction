# Docker Setup Guide

## Dockerfile Overview

The Dockerfile creates an optimized environment for training YOLO models with CUDA support.

## Key Features

- **Base Image**: NVIDIA CUDA 11.8.0 with cuDNN 8 runtime
- **Python**: 3.10
- **GPU Support**: CUDA 11.8 compatible
- **Dependencies**: All required libraries for YOLOv8 training

## Improvements Made

### 1. System Dependencies

Added required libraries for OpenCV and other dependencies:

- `libglib2.0-0`: GLib library
- `libsm6`, `libxext6`, `libxrender-dev`: X11 libraries for OpenCV
- `libgomp1`: OpenMP library
- `python3.10-dev`: Python development headers

### 2. Installation Order Optimization

- PyTorch installed first (largest dependency, changes less frequently)
- Requirements installed after (can leverage Docker cache better)
- Project files copied last (most frequent changes)

### 3. Environment Variables

- `PYTHONUNBUFFERED=1`: Ensures Python output is not buffered (important for Docker logs)
- `CUDA_VISIBLE_DEVICES=0`: Sets default GPU device

### 4. Build Optimization

- Uses `--no-cache-dir` for pip to reduce image size
- Removes apt cache with `rm -rf /var/lib/apt/lists/*`
- Upgrades pip before installing packages

## Building the Image

```bash
docker build -t basketball-trainer .
```

## Running Training

### Basic Usage (with volume mounts)

```bash
docker run --gpus all \
  -v $(pwd)/runs:/app/runs \
  -v $(pwd)/data.yaml:/app/data.yaml \
  basketball-trainer
```

### Custom Parameters

```bash
docker run --gpus all \
  -v $(pwd)/runs:/app/runs \
  -v $(pwd)/data.yaml:/app/data.yaml \
  basketball-trainer \
  python yolo_cuda_trainer.py -d data.yaml -e 100 -i 640 -b 16
```

### With Dataset Volume

If your dataset is in a local directory:

```bash
docker run --gpus all \
  -v $(pwd)/runs:/app/runs \
  -v $(pwd)/data.yaml:/app/data.yaml \
  -v $(pwd)/dataset:/app/dataset \
  basketball-trainer
```

### Interactive Mode (for debugging)

```bash
docker run --gpus all -it \
  -v $(pwd)/runs:/app/runs \
  -v $(pwd)/data.yaml:/app/data.yaml \
  basketball-trainer \
  /bin/bash
```

## Jupyter Notebook Access

To run Jupyter notebook in the container:

```bash
docker run --gpus all \
  -p 8888:8888 \
  -v $(pwd):/app \
  basketball-trainer \
  jupyter notebook --ip=0.0.0.0 --allow-root --no-browser
```

Then access at `http://localhost:8888`

## Troubleshooting

### GPU Not Detected

Ensure you have:

1. NVIDIA Docker runtime installed
2. `--gpus all` flag in docker run command
3. Compatible NVIDIA drivers

Check GPU access:

```bash
docker run --gpus all basketball-trainer nvidia-smi
```

### Out of Memory

Reduce batch size in training command:

```bash
python yolo_cuda_trainer.py -d data.yaml -e 50 -b 8
```

### Missing Dependencies

If you encounter missing system libraries, add them to the Dockerfile's `apt-get install` section.

## Image Size Optimization

The current image is optimized for functionality. To reduce size further:

- Use multi-stage builds
- Remove Jupyter if not needed
- Use Alpine-based images (but may have compatibility issues)

## Best Practices

1. **Always mount volumes** for data and outputs to persist results
2. **Use specific tags** instead of `latest` for reproducibility
3. **Set resource limits** if running on shared systems:
   ```bash
   docker run --gpus all --memory="16g" --cpus="8" ...
   ```
4. **Use .dockerignore** to exclude unnecessary files (already configured)
