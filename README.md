# Basketball Trajectory Analyzer

A comprehensive basketball trajectory analysis system using YOLO object detection and physics-based trajectory prediction.

## Features

- Real-time basketball detection using YOLOv8
- Basketball rim and player detection
- Shot phase detection (position, release, followthrough)
- Physics-based trajectory prediction
- Shot probability calculation
- Real-time visualization with trajectory overlay
- Web interface via Streamlit
- Desktop application via OpenCV
- Configurable analysis parameters

## Installation

### Prerequisites

- Python 3.8 or higher
- CUDA-compatible GPU (optional, for better performance)

### Virtual Environment (Recommended)

It's recommended to use a virtual environment to isolate dependencies:

```bash
# Create virtual environment
python3 -m venv .venv

# Activate it
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

Or use the provided script:

```bash
./activate_venv.sh
```

### Install from requirements.txt

```bash
pip install -r requirements.txt
```

### Install as package (development)

```bash
pip install -e .
```

### Install with GPU support

```bash
pip install -r requirements.txt
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

## Usage

### Basic Usage

**Application OpenCV (Desktop):**

```bash
python scripts/main.py
```

**Application Streamlit (Web):**

```bash
# From project root
python playground/run_streamlit.py

# Or directly with Streamlit
cd playground
streamlit run streamlit_app.py
```

### Controls

- **SPACE**: Pause/Resume
- **q**: Quit
- **+**: Increase playback speed
- **-**: Decrease playback speed
- **r**: Reset basket position

### Configuration

Edit the configuration in `basketball_analyzer/config.py`:

```python
# Physics constants
GRAVITY = 9.81  # m/s²
DEFAULT_FPS = 30
DEFAULT_PX_PER_METER = 150

# Detection parameters
MIN_SPEED_THRESHOLD = 0.5
BALL_TRACK_HISTORY = 20
```

## Project Structure

```
ShotPrediction/
├── basketball_analyzer/         # Main Python package
│   ├── __init__.py
│   ├── analyzer.py               # Main analysis class
│   ├── detector.py               # YOLO object detection
│   ├── trajectory.py             # Trajectory prediction
│   ├── visualizer.py             # Results visualization
│   ├── config.py                 # Configuration and constants
│   ├── utils.py                  # Utility functions
│   ├── shot_detector.py          # Shot detection logic
│   └── shot_phase_detector.py    # Shot phase detection (copyme.pt)
│
├── models/                       # Trained YOLO models
│   ├── shot.pt                   # YOLO model for basic basketball detection
│   └── copyme.pt                 # YOLO model for shot phase detection
│
├── assets/                       # Example videos and media
│   ├── basket.mp4
│   ├── shot.mp4
│   ├── 3.mp4
│   └── amaze.mp4
│
├── docs/                         # Technical documentation
│   ├── BASKETBALL_CALCULATIONS.md
│   └── TRAJECTORY_CALCULATION.md
│
├── notebooks/                    # Jupyter notebooks
│   └── DatasetTraning.ipynb      # Dataset training notebook
│
├── scripts/                      # Execution scripts
│   └── main.py                  # Main OpenCV application
│
├── playground/                   # Streamlit web application
│   ├── streamlit_app.py         # Streamlit web application
│   ├── run_streamlit.py         # Streamlit launcher script
│   ├── styles.css               # Custom CSS for Streamlit app
│   └── README.md                # Streamlit playground documentation
│
├── README.md                     # Main documentation
├── STRUCTURE.md                  # Project structure guide
├── activate_venv.sh              # Virtual environment activation script
├── requirements.txt              # Python dependencies
├── setup.py                      # Package configuration
└── data.yaml                    # YOLO dataset configuration
```

See `STRUCTURE.md` for more details on project organization.

## Dependencies

### Core Dependencies

- **opencv-python**: Computer vision and image processing
- **ultralytics**: YOLOv8 object detection
- **torch**: Deep learning framework
- **numpy**: Numerical computing
- **scipy**: Scientific computing

### Optional Dependencies

- **numba**: JIT compilation for performance
- **matplotlib**: Additional visualization
- **pandas**: Data analysis

## Model Requirements

Place your trained YOLO model file in the models directory:

- `models/shot.pt`: YOLO model trained for basketball detection (basic detection)
- `models/copyme.pt`: YOLO model trained for shot phase detection (detects different phases of a shot)

### Basic Detection Model (shot.pt)

The model should detect:

- Class 0: Basketball
- Class 1: Player
- Class 2: Basketball rim

### Shot Phase Detection Model (copyme.pt)

This model detects different phases of a basketball shot, providing more detailed analysis of the shooting motion and trajectory phases.

The model detects three shot phases:

- **Class 0: shot_followthrough** - Follow-through motion after release
- **Class 1: shot_position** - Player positioning and preparation phase
- **Class 2: shot_release** - Ball release phase

This model can be enabled in the analyzer to track the progression of a shot through these different phases.

## Model Training

### Training Options

You have several options to train your models:

1. **macOS with Docker (CPU)** - Detailed guide below
   - Simple to set up
   - Works without GPU
   - Slower (several hours for 50 epochs)
   - Ideal for testing and validating the process

2. **Linux with GPU** - Faster
   - Use the same Docker command with `--gpus all`
   - Much faster training (minutes/hours)
   - Requires NVIDIA GPU with CUDA

3. **Cloud (Google Colab, AWS, GCP)** - Fastest
   - Free GPU access on Google Colab
   - Use the provided Jupyter notebook: `notebooks/DatasetTraning.ipynb`

**Choose the option that suits your hardware and needs.**

---

### macOS Training Guide (Step by Step)

**Recommended option if you are on Mac and want to test quickly.**

This guide walks you through training a YOLO model on macOS with Docker.

#### Prerequisites

- **macOS** (tested on macOS 13+)
- **Docker Desktop** installed and running
- **At least 10GB of free disk space**
- **Roboflow API Key** (free, see step 2)
- **Stable Internet connection**

**Important note:** On macOS, training is done in CPU mode (no GPU). Training will be slower than with a GPU, but works perfectly for testing and validating the process. For fast training, use a Linux machine with GPU or a cloud service.

---

### Step 1: Check Docker Desktop

**Objective:** Ensure Docker is installed and running.

1. Open **Docker Desktop** from Applications
2. Wait for the Docker icon in the menu bar to turn green (Docker is ready)
3. Open a terminal and check:

```bash
# Check Docker version
docker --version

# Check that Docker is running
docker info

# Check available disk space (need at least 10GB)
df -h
```

**Success:** You should see the Docker version and system information.

**Error:** Make sure Docker Desktop is running and wait a few seconds.

---

### Step 2: Get a Roboflow API Key

**Objective:** Retrieve the API key to automatically download datasets.

1. Go to [https://app.roboflow.com/](https://app.roboflow.com/)
2. Create a free account (or log in)
3. Click on your **avatar** (top right) → **Account Settings**
4. In the **API Keys** section, copy your API key
5. In your terminal, set the environment variable:

```bash
export ROBOFLOW_API_KEY="your_api_key_here"
```

**Verification:** Check that the key is set:

```bash
echo $ROBOFLOW_API_KEY
```

You should see your key displayed.

---

### Step 3: Download Datasets

**Objective:** Automatically download the required datasets.

1. Make sure you are in the project directory:

```bash
cd /path/to/ShotPrediction
```

2. Download the main dataset (basketball):

```bash
python scripts/download_datasets.py --dataset basketball --target datasets
```

**Verification:** Check that files are downloaded:

```bash
ls -la datasets/basketball/
```

You should see the `train/`, `valid/`, `test/` folders with their `images/` and `labels/` subfolders.

**Note:** Download may take a few minutes depending on your connection.

---

### Step 4: Build Docker Image

**Objective:** Create the Docker image containing all necessary tools.

1. Make sure you are in the project directory:

```bash
cd /path/to/ShotPrediction
```

2. Build the Docker image:

```bash
docker build -t basketball-trainer .
```

**Estimated time:** 10-20 minutes the first time (downloading dependencies). Subsequent builds will be faster thanks to Docker cache.

**Verification:** Check that the image is created:

```bash
docker images | grep basketball-trainer
```

You should see the `basketball-trainer` image listed.

**Error:**
- Check that Docker Desktop is running
- Check your disk space: `df -h`
- See the "Troubleshooting" section below

---

### Step 5: Run Test Training (1 epoch)

**Objective:** Verify everything works with a quick training.

1. Run training with **1 epoch only** to test:

```bash
docker run -v $(pwd)/runs:/app/runs \
           -v $(pwd)/data.yaml:/app/data.yaml \
           -v $(pwd)/datasets:/app/datasets \
           basketball-trainer \
           python yolo_cuda_trainer.py -d data.yaml -e 1 -i 640 -b 8
```

**Estimated time:** 30-60 minutes in CPU mode (on Mac).

**Verification:** At the end of training, check that results are saved:

```bash
ls -la runs/detect/train*/weights/
```

You should see the `best.pt` and `last.pt` files.

---

### Step 6: Run Full Training

**Objective:** Train the model with all optimal parameters.

Once the test is successful, run the full training:

```bash
docker run -v $(pwd)/runs:/app/runs \
           -v $(pwd)/data.yaml:/app/data.yaml \
           -v $(pwd)/datasets:/app/datasets \
           basketball-trainer \
           python yolo_cuda_trainer.py -d data.yaml -e 50 -i 640 -b 8
```

**Estimated time:** Several hours in CPU mode (on Mac). Training can be left in the background.

**Available parameters:**

- `-e, --epochs` : Number of epochs (default: 50)
- `-i, --size` : Image size (640, 800, 1280, etc.) (default: 640)
- `-b, --batch` : Batch size (default: 8 for CPU)
- `-d, --data` : Dataset YAML file (default: `data.yaml`)

---

### Step 7: Retrieve Trained Model

**Objective:** Copy the best model to the `models/` folder.

Once training is complete:

1. Find the results folder (usually `runs/detect/train/` or `runs/detect/train5/`, etc.):

```bash
ls -la runs/detect/
```

2. Copy the best model:

```bash
cp runs/detect/train*/weights/best.pt models/shot.pt
```

**Verification:** Check that the file is copied:

```bash
ls -lh models/shot.pt
```

You should see a `.pt` file of several tens of MB.

---

### Essential Commands Summary

For a quick test, run these commands in order:

```bash
# 1. Set Roboflow API key
export ROBOFLOW_API_KEY="your_api_key"

# 2. Download dataset
python scripts/download_datasets.py --dataset basketball --target datasets

# 3. Build Docker image
docker build -t basketball-trainer .

# 4. Test with 1 epoch
docker run -v $(pwd)/runs:/app/runs \
           -v $(pwd)/data.yaml:/app/data.yaml \
           -v $(pwd)/datasets:/app/datasets \
           basketball-trainer \
           python yolo_cuda_trainer.py -d data.yaml -e 1 -i 640 -b 8

# 5. (Optional) Full training
docker run -v $(pwd)/runs:/app/runs \
           -v $(pwd)/data.yaml:/app/data.yaml \
           -v $(pwd)/datasets:/app/datasets \
           basketball-trainer \
           python yolo_cuda_trainer.py -d data.yaml -e 50 -i 640 -b 8
```

---

### Linux GPU Training (Fast Alternative)

**Recommended option if you have access to a Linux machine with NVIDIA GPU.**

If you are on Linux with an NVIDIA GPU, training will be **much faster**:

1. **Install NVIDIA Container Toolkit** (if not already done):
```bash
# Ubuntu/Debian
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list
sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
sudo systemctl restart docker
```

2. **Build Docker image** (same command as on Mac):
```bash
docker build -t basketball-trainer .
```

3. **Run training with GPU**:
```bash
docker run --gpus all \
           -v $(pwd)/runs:/app/runs \
           -v $(pwd)/data.yaml:/app/data.yaml \
           -v $(pwd)/datasets:/app/datasets \
           basketball-trainer \
           python yolo_cuda_trainer.py -d data.yaml -e 50 -i 640 -b 16
```

**Differences from Mac:**
- Added `--gpus all` to enable GPU
- Higher batch size (`-b 16` instead of `-b 8`) because GPU has more memory
- **Training time:** Minutes/hours instead of several hours

---

### Cloud Training (Google Colab, etc.)

**Recommended option if you don't have a local GPU.**

1. Open the Jupyter notebook: `notebooks/DatasetTraning.ipynb`
2. Upload it to [Google Colab](https://colab.research.google.com/)
3. Execute cells in order
4. Colab provides free GPU access for training

**Advantages:**
- Free GPU access
- No need to install Docker
- Intuitive Jupyter interface

---

#### Troubleshooting Docker Build Issues

**Error: "input/output error" or "failed to solve"**
- Ensure Docker Desktop is running
- Check available disk space: `df -h` (need at least 10GB free)
- Clean Docker cache: `docker system prune -a`
- Restart Docker Desktop
- Try rebuilding: `docker build --no-cache -t basketball-trainer .`

**Low Disk Space - How to Free Up Space**

If you're running low on disk space (less than 10GB free), here are steps to free up space:

1. **Clean Docker (when Docker Desktop is running):**
```bash
# Check Docker disk usage
docker system df

# Remove all unused Docker resources (images, containers, volumes, cache)
docker system prune -a --volumes -f

# This can free several GB of space
```

2. **Clean macOS system files:**
```bash
# Empty Trash
# Remove old iOS simulators (can free 10-20GB)
# Use macOS Storage Management: Apple Menu > About This Mac > Storage > Manage
```

3. **Clean Python cache and virtual environments:**
```bash
# Remove Python cache files
find . -type d -name __pycache__ -exec rm -r {} +
find . -type f -name "*.pyc" -delete

# Remove old virtual environments if not needed
```

4. **Alternative: Use Cloud Training**
   - If local disk space is limited, consider using cloud services (Google Colab, AWS, GCP)
   - The Jupyter notebook can be run on Google Colab with free GPU access

**Error: "Cannot connect to the Docker daemon"**
- Start Docker Desktop application
- Wait for Docker to fully initialize (check system tray/status)

**Error: "--gpus all" not supported (macOS)**
- On macOS, GPU support is limited. Use CPU-only mode:
```bash
docker run -v $(pwd)/runs:/app/runs -v $(pwd)/data.yaml:/app/data.yaml basketball-trainer
```
- Or use a Linux machine/cloud instance for GPU training

**Build takes too long or fails during download**
- The CUDA base image is large (~1.2GB). Ensure stable internet connection
- First build may take 10-20 minutes depending on connection speed
- Subsequent builds will be faster due to Docker layer caching

### Training with Python Script

#### 1. Install dependencies

```bash
pip install -r requirements.txt
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

#### 2. Prepare dataset

Ensure your dataset is organized in YOLO format:

```
dataset/
├── train/
│   └── images/
├── valid/
│   └── images/
└── test/
    └── images/
```

Update `data.yaml` with the correct paths to your dataset.

#### Download datasets automatically (Roboflow)

If you have a Roboflow API key, you can retrieve official datasets with one command:

```bash
export ROBOFLOW_API_KEY="your_api_key"
# Main dataset (ball/player/rim)
python scripts/download_datasets.py --dataset basketball --target datasets

# Shot phases dataset
python scripts/download_datasets.py --dataset shotanalysis --target datasets

# Download everything
python scripts/download_datasets.py --dataset all --target datasets

# No API key? Use direct mode:
python scripts/download_datasets.py --dataset all --target datasets --method direct
```

Archives are extracted to the provided folder (`datasets` above). Then mount this folder in Docker:

```bash
docker run -v $(pwd)/runs:/app/runs \
           -v $(pwd)/data.yaml:/app/data.yaml \
           -v $(pwd)/datasets:/app/datasets \
           basketball-trainer
```

#### 3. Run training script

```bash
python yolo_cuda_trainer.py -d data.yaml -e 50 -i 640 -b 16
```

The script will automatically:

- Detect available GPU (CUDA/MPS/CPU)
- Optimize CUDA settings for performance
- Auto-detect optimal batch size based on GPU memory
- Save best model weights to `runs/detect/train/weights/best.pt`

### Training with Jupyter Notebook

#### 1. Open the notebook

```bash
jupyter notebook notebooks/DatasetTraning.ipynb
```

#### 2. Follow the notebook steps

The notebook includes:

- Installation of required libraries (ultralytics, roboflow)
- Dataset download from Roboflow (or use your own dataset)
- Model training with YOLOv8
- Model export instructions

#### 3. Download trained model

After training completes, download the model from:

- `runs/detect/train/weights/best.pt` (best model)
- `runs/detect/train/weights/last.pt` (last checkpoint)

### Dataset Information

#### Basic Detection Model (shot.pt)

**Dataset:** [Basketball Detection Dataset](https://universe.roboflow.com/cricket-qnb5l/basketball-xil7x/dataset/1)

**Classes:**

- Class 0: `ball` - Basketball
- Class 1: `human` - Player
- Class 2: `rim` - Basketball rim

**Configuration:** See `data.yaml` for dataset paths and class definitions.

#### Shot Phase Detection Model (copyme.pt)

**Dataset:** [Shot Analysis Dataset](https://universe.roboflow.com/copyme-3cenq/shotanalysis/dataset/21)

**Classes:**

- Class 0: `shot_followthrough` - Follow-through motion after release
- Class 1: `shot_position` - Player positioning and preparation phase
- Class 2: `shot_release` - Ball release phase (note: model uses 'shot_realese' spelling)

**Configuration:** The model configuration is embedded in the `.pt` file.

## Running the Project

### Option 1: Streamlit Web Application (Recommended)

The Streamlit application provides an interactive web interface for video analysis.

#### Launch with Python script:

```bash
# From project root
python playground/run_streamlit.py
```

#### Launch directly with Streamlit:

```bash
cd playground
streamlit run streamlit_app.py
```

The application will open in your default web browser at `http://localhost:8501`.

**Features:**

- Upload video files or use webcam
- Real-time trajectory analysis
- Shot probability calculation
- Shot phase detection visualization
- Configurable detection parameters
- Real-time metrics display

### Option 2: OpenCV Desktop Application

For a desktop application with OpenCV:

```bash
python scripts/main.py
```

**Controls:**

- **SPACE**: Pause/Resume
- **q**: Quit
- **+**: Increase playback speed
- **-**: Decrease playback speed
- **r**: Reset basket position

### Option 3: Install as Package

Install the package in development mode:

```bash
pip install -e .
```

Then use the command-line interface:

```bash
basketball-analyzer
```

## Performance Optimization

### GPU Support

For CUDA support, install PyTorch with CUDA:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

### CPU-only Installation

If you don't have a GPU, the application will run on CPU (slower):

```bash
pip install -r requirements.txt
```

The models will automatically use CPU if no GPU is available.
