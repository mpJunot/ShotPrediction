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

### Options d'entraînement

Vous avez plusieurs options pour entraîner vos modèles :

1. **macOS avec Docker (CPU)** - Guide détaillé ci-dessous ⬇️
   - ✅ Simple à mettre en place
   - ✅ Fonctionne sans GPU
   - ⚠️ Plus lent (plusieurs heures pour 50 epochs)
   - 💡 Idéal pour tester et valider le processus

2. **Linux avec GPU** - Plus rapide
   - Utilisez la même commande Docker avec `--gpus all`
   - Entraînement beaucoup plus rapide (quelques minutes/heures)
   - Nécessite un GPU NVIDIA avec CUDA

3. **Cloud (Google Colab, AWS, GCP)** - Le plus rapide
   - Accès gratuit à GPU sur Google Colab
   - Utilisez le notebook Jupyter fourni : `notebooks/DatasetTraning.ipynb`

**Choisissez l'option qui vous convient le mieux selon votre matériel et vos besoins.**

---

### Guide d'entraînement sur macOS (Étape par étape)

**⚠️ Option recommandée si vous êtes sur Mac et voulez tester rapidement.**

Ce guide vous accompagne pas à pas pour entraîner un modèle YOLO sur macOS avec Docker.

#### Prérequis

- **macOS** (testé sur macOS 13+)
- **Docker Desktop** installé et en cours d'exécution
- **Au moins 10GB d'espace disque libre**
- **Clé API Roboflow** (gratuite, voir étape 2)
- **Connexion Internet** stable

**Note importante :** Sur macOS, l'entraînement se fait en mode CPU (pas de GPU). L'entraînement sera plus lent qu'avec un GPU, mais fonctionne parfaitement pour tester et valider le processus. Pour un entraînement rapide, utilisez une machine Linux avec GPU ou un service cloud.

---

### Étape 1 : Vérifier Docker Desktop

**Objectif :** S'assurer que Docker est installé et fonctionne.

1. Ouvrez **Docker Desktop** depuis Applications
2. Attendez que l'icône Docker dans la barre de menu soit verte (Docker est prêt)
3. Ouvrez un terminal et vérifiez :

```bash
# Vérifier la version Docker
docker --version

# Vérifier que Docker fonctionne
docker info

# Vérifier l'espace disque disponible (besoin d'au moins 10GB)
df -h
```

**✅ Si tout fonctionne :** Vous devriez voir la version Docker et des informations système.

**❌ Si erreur :** Assurez-vous que Docker Desktop est bien lancé et attendez quelques secondes.

---

### Étape 2 : Obtenir une clé API Roboflow

**Objectif :** Récupérer la clé API pour télécharger automatiquement les datasets.

1. Allez sur [https://app.roboflow.com/](https://app.roboflow.com/)
2. Créez un compte gratuit (ou connectez-vous)
3. Cliquez sur votre **avatar** (en haut à droite) → **Account Settings**
4. Dans la section **API Keys**, copiez votre clé API
5. Dans votre terminal, définissez la variable d'environnement :

```bash
export ROBOFLOW_API_KEY="votre_cle_api_ici"
```

**✅ Vérification :** Vérifiez que la clé est bien définie :

```bash
echo $ROBOFLOW_API_KEY
```

Vous devriez voir votre clé affichée.

---

### Étape 3 : Télécharger les datasets

**Objectif :** Télécharger automatiquement les jeux de données nécessaires.

1. Assurez-vous d'être dans le répertoire du projet :

```bash
cd /chemin/vers/ShotPrediction
```

2. Téléchargez le dataset principal (basketball) :

```bash
python scripts/download_datasets.py --dataset basketball --target datasets
```

**✅ Vérification :** Vérifiez que les fichiers sont bien téléchargés :

```bash
ls -la datasets/basketball/
```

Vous devriez voir les dossiers `train/`, `valid/`, `test/` avec leurs sous-dossiers `images/` et `labels/`.

**Note :** Le téléchargement peut prendre quelques minutes selon votre connexion.

---

### Étape 4 : Construire l'image Docker

**Objectif :** Créer l'image Docker contenant tous les outils nécessaires.

1. Assurez-vous d'être dans le répertoire du projet :

```bash
cd /chemin/vers/ShotPrediction
```

2. Construisez l'image Docker :

```bash
docker build -t basketball-trainer .
```

**⏱️ Temps estimé :** 10-20 minutes la première fois (téléchargement des dépendances). Les fois suivantes seront plus rapides grâce au cache Docker.

**✅ Vérification :** Vérifiez que l'image est bien créée :

```bash
docker images | grep basketball-trainer
```

Vous devriez voir l'image `basketball-trainer` listée.

**❌ Si erreur :** 
- Vérifiez que Docker Desktop est bien lancé
- Vérifiez votre espace disque : `df -h`
- Consultez la section "Troubleshooting" ci-dessous

---

### Étape 5 : Lancer un test d'entraînement (1 epoch)

**Objectif :** Vérifier que tout fonctionne avec un entraînement rapide.

1. Lancez l'entraînement avec **1 seul epoch** pour tester :

```bash
docker run -v $(pwd)/runs:/app/runs \
           -v $(pwd)/data.yaml:/app/data.yaml \
           -v $(pwd)/datasets:/app/datasets \
           basketball-trainer \
           python yolo_cuda_trainer.py -d data.yaml -e 1 -i 640 -b 8
```

**⏱️ Temps estimé :** 30-60 minutes en mode CPU (sur Mac).

**✅ Vérification :** À la fin de l'entraînement, vérifiez que les résultats sont sauvegardés :

```bash
ls -la runs/detect/train*/weights/
```

Vous devriez voir les fichiers `best.pt` et `last.pt`.

---

### Étape 6 : Lancer l'entraînement complet

**Objectif :** Entraîner le modèle avec tous les paramètres optimaux.

Une fois le test réussi, lancez l'entraînement complet :

```bash
docker run -v $(pwd)/runs:/app/runs \
           -v $(pwd)/data.yaml:/app/data.yaml \
           -v $(pwd)/datasets:/app/datasets \
           basketball-trainer \
           python yolo_cuda_trainer.py -d data.yaml -e 50 -i 640 -b 8
```

**⏱️ Temps estimé :** Plusieurs heures en mode CPU (sur Mac). L'entraînement peut être laissé en arrière-plan.

**Paramètres disponibles :**

- `-e, --epochs` : Nombre d'epochs (défaut: 50)
- `-i, --size` : Taille des images (640, 800, 1280, etc.) (défaut: 640)
- `-b, --batch` : Taille du batch (défaut: 8 pour CPU)
- `-d, --data` : Fichier YAML du dataset (défaut: `data.yaml`)

---

### Étape 7 : Récupérer le modèle entraîné

**Objectif :** Copier le meilleur modèle dans le dossier `models/`.

Une fois l'entraînement terminé :

1. Trouvez le dossier de résultats (généralement `runs/detect/train/` ou `runs/detect/train5/`, etc.) :

```bash
ls -la runs/detect/
```

2. Copiez le meilleur modèle :

```bash
cp runs/detect/train*/weights/best.pt models/shot.pt
```

**✅ Vérification :** Vérifiez que le fichier est bien copié :

```bash
ls -lh models/shot.pt
```

Vous devriez voir un fichier `.pt` de plusieurs dizaines de Mo.

---

### Résumé des commandes essentielles

Pour un test rapide, exécutez ces commandes dans l'ordre :

```bash
# 1. Définir la clé API Roboflow
export ROBOFLOW_API_KEY="votre_cle_api"

# 2. Télécharger le dataset
python scripts/download_datasets.py --dataset basketball --target datasets

# 3. Construire l'image Docker
docker build -t basketball-trainer .

# 4. Tester avec 1 epoch
docker run -v $(pwd)/runs:/app/runs \
           -v $(pwd)/data.yaml:/app/data.yaml \
           -v $(pwd)/datasets:/app/datasets \
           basketball-trainer \
           python yolo_cuda_trainer.py -d data.yaml -e 1 -i 640 -b 8

# 5. (Optionnel) Entraînement complet
docker run -v $(pwd)/runs:/app/runs \
           -v $(pwd)/data.yaml:/app/data.yaml \
           -v $(pwd)/datasets:/app/datasets \
           basketball-trainer \
           python yolo_cuda_trainer.py -d data.yaml -e 50 -i 640 -b 8
```

---

### Entraînement sur Linux avec GPU (Alternative rapide)

**⚠️ Option recommandée si vous avez accès à une machine Linux avec GPU NVIDIA.**

Si vous êtes sur Linux avec un GPU NVIDIA, l'entraînement sera **beaucoup plus rapide** :

1. **Installer NVIDIA Container Toolkit** (si pas déjà fait) :
```bash
# Ubuntu/Debian
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list
sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
sudo systemctl restart docker
```

2. **Construire l'image Docker** (même commande que sur Mac) :
```bash
docker build -t basketball-trainer .
```

3. **Lancer l'entraînement avec GPU** :
```bash
docker run --gpus all \
           -v $(pwd)/runs:/app/runs \
           -v $(pwd)/data.yaml:/app/data.yaml \
           -v $(pwd)/datasets:/app/datasets \
           basketball-trainer \
           python yolo_cuda_trainer.py -d data.yaml -e 50 -i 640 -b 16
```

**Différences avec Mac :**
- Ajout de `--gpus all` pour activer le GPU
- Batch size plus élevé (`-b 16` au lieu de `-b 8`) car le GPU a plus de mémoire
- **Temps d'entraînement :** Quelques minutes/heures au lieu de plusieurs heures

---

### Entraînement sur Cloud (Google Colab, etc.)

**⚠️ Option recommandée si vous n'avez pas de GPU local.**

1. Ouvrez le notebook Jupyter : `notebooks/DatasetTraning.ipynb`
2. Uploadez-le sur [Google Colab](https://colab.research.google.com/)
3. Exécutez les cellules dans l'ordre
4. Colab fournit gratuitement un GPU pour l'entraînement

**Avantages :**
- ✅ Accès gratuit à un GPU
- ✅ Pas besoin d'installer Docker
- ✅ Interface Jupyter intuitive

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

Si tu disposes d'une clé API Roboflow, tu peux récupérer les jeux de données officiels en un clic :

```bash
export ROBOFLOW_API_KEY="ta_clef_api"
# Dataset principal (ball/joueur/cerceau)
python scripts/download_datasets.py --dataset basketball --target datasets

# Dataset phases de tir
python scripts/download_datasets.py --dataset shotanalysis --target datasets

# Tout télécharger
python scripts/download_datasets.py --dataset all --target datasets

# Sans clé API ? Utilise le mode direct :
python scripts/download_datasets.py --dataset all --target datasets --method direct
```

Les archives sont extraites dans le dossier fourni (`datasets` ci-dessus). Monte ensuite ce dossier dans Docker :

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
