# Training Parameters Explanation

This document explains all parameters used in the `training_args` dictionary for YOLOv8 model training.

## Core Training Parameters

### `data`

- **Type**: String (path to YAML file)
- **Description**: Path to the dataset configuration file (data.yaml)
- **Example**: `'data.yaml'` or `dataset.location`
- **Purpose**: Defines dataset paths (train/val/test) and class names

### `epochs`

- **Type**: Integer
- **Description**: Number of complete passes through the training dataset
- **Default**: 50
- **Purpose**: Controls how long the model trains. More epochs = longer training but potentially better results

### `imgsz`

- **Type**: Integer
- **Description**: Input image size (square dimensions)
- **Default**: 640
- **Options**: 320, 640, 800, 1280, etc.
- **Purpose**: Larger images = better accuracy but slower training and more memory usage

### `device`

- **Type**: String
- **Description**: Computing device to use
- **Options**: `'cuda'`, `'mps'` (Apple Silicon), `'cpu'`
- **Purpose**: Determines whether to use GPU (faster) or CPU (slower)

### `batch`

- **Type**: Integer
- **Description**: Number of images processed in parallel per training step
- **Default**: Auto-detected based on GPU memory (8-32)
- **Purpose**: Larger batch = faster training but requires more GPU memory

### `workers`

- **Type**: Integer
- **Description**: Number of parallel data loading workers
- **Default**: 8
- **Purpose**: More workers = faster data loading, but uses more CPU/RAM

## Performance Optimizations

### `amp`

- **Type**: Boolean
- **Description**: Automatic Mixed Precision training
- **Default**: `True`
- **Purpose**: Uses FP16 precision for faster training with minimal accuracy loss. Reduces memory usage and speeds up training on modern GPUs

### `cache`

- **Type**: Boolean
- **Description**: Cache images in RAM/disk for faster loading
- **Default**: `True`
- **Purpose**: Speeds up training by avoiding repeated disk reads. Uses more RAM but significantly faster

## Optimizer Settings

### `optimizer`

- **Type**: String
- **Description**: Optimization algorithm
- **Default**: `'AdamW'`
- **Options**: `'SGD'`, `'Adam'`, `'AdamW'`
- **Purpose**: AdamW generally provides better convergence than SGD for object detection tasks

### `cos_lr`

- **Type**: Boolean
- **Description**: Cosine learning rate scheduler
- **Default**: `True`
- **Purpose**: Gradually decreases learning rate following a cosine curve, improving final model accuracy

## Data Augmentation

Data augmentation helps the model generalize better by creating variations of training images.

### `hsv_h`, `hsv_s`, `hsv_v`

- **Type**: Float (0.0 to 1.0)
- **Description**: HSV color space augmentation
  - `hsv_h`: Hue shift (0.015 = ±1.5% hue variation)
  - `hsv_s`: Saturation variation (0.7 = ±70% saturation)
  - `hsv_v`: Value/brightness variation (0.4 = ±40% brightness)
- **Purpose**: Makes model robust to different lighting conditions and colors

### `degrees`

- **Type**: Float
- **Description**: Rotation angle in degrees
- **Default**: `0.0` (no rotation)
- **Purpose**: Rotates images to handle different camera angles

### `translate`

- **Type**: Float (0.0 to 1.0)
- **Description**: Translation factor (0.1 = ±10% shift)
- **Default**: `0.1`
- **Purpose**: Shifts images horizontally/vertically to handle object position variations

### `scale`

- **Type**: Float (0.0 to 1.0)
- **Description**: Scaling factor (0.5 = ±50% size variation)
- **Default**: `0.5`
- **Purpose**: Makes model handle objects at different distances/sizes

### `shear`

- **Type**: Float
- **Description**: Shear angle in degrees
- **Default**: `0.0` (no shearing)
- **Purpose**: Applies geometric distortion

### `flipud`

- **Type**: Float (0.0 to 1.0)
- **Description**: Probability of vertical flip
- **Default**: `0.0` (disabled)
- **Purpose**: Flips images upside down (rarely useful for basketball)

### `fliplr`

- **Type**: Float (0.0 to 1.0)
- **Description**: Probability of horizontal flip
- **Default**: `0.5` (50% chance)
- **Purpose**: Mirrors images left-right, useful for symmetric objects

### `mosaic`

- **Type**: Float (0.0 to 1.0)
- **Description**: Probability of mosaic augmentation
- **Default**: `1.0` (always enabled)
- **Purpose**: Combines 4 images into one, helps model learn to detect objects at different scales

### `close_mosaic`

- **Type**: Integer
- **Description**: Disable mosaic in last N epochs
- **Default**: `10`
- **Purpose**: Disables mosaic in final epochs for more stable training and better final accuracy

## Training Control

### `patience`

- **Type**: Integer
- **Description**: Early stopping patience (epochs to wait without improvement)
- **Default**: `50`
- **Purpose**: Stops training early if validation metrics don't improve, saving time

### `save`

- **Type**: Boolean
- **Description**: Save training checkpoints
- **Default**: `True`
- **Purpose**: Saves model weights during training for recovery/resume

### `save_period`

- **Type**: Integer
- **Description**: Save checkpoint every N epochs
- **Default**: `10`
- **Purpose**: Periodic saves allow resuming from intermediate points

### `plots`

- **Type**: Boolean
- **Description**: Generate training plots and visualizations
- **Default**: `True`
- **Purpose**: Creates graphs showing loss, mAP, etc. for monitoring training progress

### `verbose`

- **Type**: Boolean
- **Description**: Verbose output during training
- **Default**: `True`
- **Purpose**: Shows detailed training information (loss, metrics per epoch)

## Recommended Adjustments

### For Faster Training (Lower Accuracy)

- Reduce `epochs`: 30-50
- Reduce `imgsz`: 640 or 800
- Increase `batch`: Use maximum GPU memory allows
- Reduce `patience`: 20-30

### For Better Accuracy (Slower Training)

- Increase `epochs`: 100-200
- Increase `imgsz`: 1280
- Reduce `batch`: 8-16 (allows larger images)
- Increase `patience`: 50-100

### For Limited GPU Memory

- Reduce `batch`: 4-8
- Reduce `imgsz`: 640
- Set `cache`: `False` (uses less RAM)
- Reduce `workers`: 4

### For Maximum GPU Utilization

- Increase `batch`: 16-32 (if VRAM allows)
- Set `cache`: `True`
- Increase `workers`: 8-16
- Set `amp`: `True`
