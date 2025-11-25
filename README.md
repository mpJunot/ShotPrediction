# Basketball Trajectory Analyzer

A comprehensive basketball trajectory analysis system using YOLO object detection and physics-based trajectory prediction.

## Features

- Real-time basketball detection using YOLOv8
- Basketball rim and player detection
- Physics-based trajectory prediction
- Shot probability calculation
- Real-time visualization with trajectory overlay
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

See `SETUP_VENV.md` for detailed instructions.

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
python scripts/run_streamlit.py
# or
streamlit run scripts/streamlit_app.py
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
│   └── shot_detector.py          # Shot detection logic
│
├── models/                       # Trained YOLO models
│   └── shot.pt                   # YOLO model for basketball detection
│
├── assets/                       # Example videos and media
│   ├── match.mp4
│   ├── basket.mp4
│   ├── a.webm
│   └── c.webm
│
├── docs/                         # Technical documentation
│   ├── BASKETBALL_CALCULATIONS.md
│   └── TRAJECTORY_CALCULATION.md
│
├── notebooks/                    # Jupyter notebooks
│   └── DatasetTraning.ipynb      # Dataset training notebook
│
├── scripts/                      # Execution scripts
│   ├── main.py                  # Main OpenCV application
│   ├── streamlit_app.py         # Streamlit web application
│   ├── run_streamlit.py         # Streamlit launcher script
│   └── styles.css               # Custom CSS for Streamlit app
│
├── README.md                     # Main documentation
├── STRUCTURE.md                  # Project structure guide
├── SETUP_VENV.md                 # Virtual environment setup guide
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

- `models/shot.pt`: YOLO model trained for basketball detection

The model should detect:

- Class 0: Basketball
- Class 1: Player
- Class 2: Basketball rim

## Performance Optimization

### GPU Support

For CUDA support, install PyTorch with CUDA:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```
