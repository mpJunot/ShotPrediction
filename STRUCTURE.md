# ShotPrediction Project Structure

## Folder Organization

```
ShotPrediction/
├── README.md                    # Main project documentation
├── requirements.txt             # Python dependencies
├── setup.py                     # Package configuration
├── data.yaml                    # YOLO dataset configuration
├── .gitignore                   # Git ignored files
│
├── basketball_analyzer/         # Main Python package
│   ├── __init__.py
│   ├── analyzer.py              # Main analysis class
│   ├── detector.py              # YOLO object detection
│   ├── trajectory.py            # Trajectory prediction
│   ├── visualizer.py            # Results visualization
│   ├── config.py                # Configuration and constants
│   ├── utils.py                 # Utility functions
│   └── shot_detector.py
│
├── models/                      # Trained YOLO models
│   ├── shot.pt                 # Basic basketball detection model
│   └── copyme.pt               # Shot phase detection model
│
├── assets/                      # Example videos and media
│   ├── match.mp4
│   ├── basket.mp4
│   ├── a.webm
│   └── c.webm
│
├── docs/                        # Technical documentation
│   ├── BASKETBALL_CALCULATIONS.md
│   ├── CALCULS_TRAJECTOIRE.md
│   └── TRAJECTORY_CALCULATION.md
│
├── notebooks/                   # Jupyter notebooks
│   └── DatasetTraning.ipynb
│
└── scripts/                     # Execution scripts
    ├── main.py                  # Main OpenCV application
    ├── streamlit_app.py         # Streamlit web application
    └── run_streamlit.py         # Streamlit launcher script
```

## Usage

### OpenCV Application (Desktop)

```bash
python scripts/main.py
```

### Streamlit Application (Web)

```bash
python scripts/run_streamlit.py
# or
streamlit run scripts/streamlit_app.py
```

## Important Notes

- All paths in scripts are relative to the **project root**
- Models should be placed in the `models/` folder
- Example videos should be in the `assets/` folder
- Technical documentation is in the `docs/` folder
