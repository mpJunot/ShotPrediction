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

### Prerequisites for Training

- CUDA-compatible GPU (recommended for faster training)
- Docker and Docker Compose (for containerized training)
- Or Python 3.8+ with CUDA toolkit installed

### Training with Docker (Recommended)

The easiest way to reproduce the training is using Docker:

#### 1. Build the Docker image

```bash
docker build -t basketball-trainer .
```

#### 2. Run training with default parameters

```bash
docker run --gpus all -v $(pwd)/runs:/app/runs -v $(pwd)/data.yaml:/app/data.yaml basketball-trainer
```

#### 3. Run training with custom parameters

```bash
docker run --gpus all \
  -v $(pwd)/runs:/app/runs \
  -v $(pwd)/data.yaml:/app/data.yaml \
  basketball-trainer \
  python yolo_cuda_trainer.py -d data.yaml -e 100 -i 640 -b 16
```

**Parameters:**

- `-d, --data`: Path to dataset YAML file (default: `data.yaml`)
- `-e, --epochs`: Number of training epochs (default: 50)
- `-i, --size`: Image size (640, 800, 1280, etc.) (default: 640)
- `-b, --batch`: Batch size (-1 for auto-detection) (default: -1)
- `-w, --workers`: Number of dataloader workers (default: 8)

#### 4. Access training results

After training, the model weights will be saved in `runs/detect/train/weights/best.pt`. Copy this file to `models/` directory:

```bash
cp runs/detect/train/weights/best.pt models/shot.pt
```

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
