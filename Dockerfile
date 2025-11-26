# Dockerfile for Basketball Trajectory Analyzer Training
# This Dockerfile sets up an environment for training YOLO models

FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

# Set working directory
WORKDIR /app

# Install system dependencies
# Includes libraries needed for opencv-python and other dependencies
RUN apt-get update && apt-get install -y \
  python3.10 \
  python3-pip \
  python3.10-dev \
  git \
  wget \
  libglib2.0-0 \
  libsm6 \
  libxext6 \
  libxrender-dev \
  libgomp1 \
  libglib2.0-0 \
  && rm -rf /var/lib/apt/lists/*

# Set Python 3.10 as default
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1 && \
  update-alternatives --install /usr/bin/pip pip /usr/bin/pip3 1

# Upgrade pip to latest version
RUN pip install --upgrade pip setuptools wheel

# Install PyTorch with CUDA support first (largest dependency)
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cu118

# Install Python dependencies from requirements.txt
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install Jupyter for notebook execution (optional)
RUN pip install --no-cache-dir jupyter ipykernel

# Copy project files (done after pip installs to leverage Docker cache)
COPY . .

# Create directories for training outputs
RUN mkdir -p runs/detect/train/weights

# Expose Jupyter port (optional, for notebook access)
EXPOSE 8888

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV CUDA_VISIBLE_DEVICES=0

# Default command: train the model
# Override with: docker run <image> python yolo_cuda_trainer.py -d data.yaml -e 50
# Or mount data.yaml as volume: docker run -v $(pwd)/data.yaml:/app/data.yaml <image>
CMD ["python", "yolo_cuda_trainer.py", "-d", "data.yaml", "-e", "50"]

