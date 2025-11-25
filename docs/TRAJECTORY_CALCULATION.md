# Basketball Trajectory Calculation

## Overview

This document explains how the basketball trajectory is calculated in the basketball trajectory analyzer system. The calculation is based on fundamental physics principles of projectile motion.

## Physics Foundation

### Projectile Motion Equations

The basketball follows a parabolic trajectory under the influence of gravity. The motion can be described by these fundamental equations:

**Position equations:**
```
x(t) = x₀ + vₓ₀ × t
y(t) = y₀ + vᵧ₀ × t + ½ × g × t²
```

**Velocity equations:**
```
vₓ(t) = vₓ₀ (constant horizontal velocity)
vᵧ(t) = vᵧ₀ + g × t (vertical velocity affected by gravity)
```

Where:
- `x₀, y₀` = initial position
- `vₓ₀, vᵧ₀` = initial velocity components
- `g` = gravitational acceleration (9.81 m/s²)
- `t` = time

## Implementation Steps

### 1. Ball Detection and Tracking

```python
def detect_objects(self, frame):
    # Uses YOLO to detect basketball (class 32 in COCO dataset)
    # Returns ball center position and size
```

The system:
- Detects the basketball using YOLO object detection
- Stores ball positions in a deque with maximum 30 positions
- Tracks the ball's movement frame by frame

### 2. Velocity Calculation

```python
def update_tracking(self, ball_detection):
    if ball_detection:
        # Calculate velocity from position difference
        vx_pixels = current_pos[0] - prev_pos[0]
        vy_pixels = current_pos[1] - prev_pos[1]

        # Convert to m/s
        vx = vx_pixels * self.fps / self.px_per_meter
        vy = vy_pixels * self.fps / self.px_per_meter
```

**Velocity calculation process:**
1. **Position difference**: Calculate pixel displacement between consecutive frames
2. **Unit conversion**: Convert pixels to meters using calibration factor
3. **Time normalization**: Account for frame rate to get velocity in m/s
4. **Smoothing**: Average the last 5 velocity measurements to reduce noise

### 3. Trajectory Prediction

```python
def predict_trajectory(self, current_pos, vx, vy, num_points=50):
    dt = 0.033  # ~30fps time step

    for i in range(num_points):
        t = i * dt

        # Physics equations
        x = x0 + vx * t
        y = y0 + vy * t + 0.5 * self.gravity * t**2

        trajectory.append((px, py))
```

**Prediction process:**
1. **Initial conditions**: Use current position and smoothed velocity
2. **Time stepping**: Calculate position at discrete time intervals (0.033s)
3. **Physics application**: Apply projectile motion equations
4. **Coordinate conversion**: Convert back to pixel coordinates
5. **Boundary checking**: Stop prediction when ball goes out of bounds

## Key Parameters

### Physical Constants

| Parameter | Value | Description |
|-----------|-------|-------------|
| `gravity` | 9.81 m/s² | Earth's gravitational acceleration |
| `fps` | 30 | Video frame rate |
| `dt` | 0.033s | Time step between frames |

### Calibration Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `px_per_meter` | 100 | Pixels per meter conversion factor |
| `confidence_threshold` | 0.5 | Minimum detection confidence |
| `maxlen` | 30 | Maximum stored positions |

### Trajectory Settings

| Parameter | Value | Description |
|-----------|-------|-------------|
| `num_points` | 50 | Number of predicted trajectory points |
| `min_speed` | 0.5 m/s | Minimum speed for prediction |

## Shot Probability Calculation

```python
def calculate_shot_probability(self, trajectory):
    # Find closest point to basket
    min_distance = min(distance_to_basket for each trajectory_point)

    # Probability based on distance
    if min_distance < basket_radius * 0.4:
        probability = 0.95
    elif min_distance < basket_radius * 0.7:
        probability = 0.80
    # ... more conditions
```

**Probability calculation:**
1. **Distance measurement**: Calculate minimum distance from trajectory to basket center
2. **Probability mapping**: Use distance-based probability thresholds
3. **Basket geometry**: Consider basket diameter (45cm) for realistic scoring

## Coordinate Systems

### Pixel Coordinates
- Origin: Top-left corner of image
- X-axis: Left to right (positive)
- Y-axis: Top to bottom (positive)

### Physics Coordinates
- Origin: Convertible to real-world measurements
- X-axis: Horizontal motion
- Y-axis: Vertical motion (gravity acts in positive direction)

## Accuracy Factors

### Factors Affecting Accuracy

1. **Camera calibration**: `px_per_meter` must be accurate for the scene
2. **Frame rate**: Higher FPS provides better velocity estimation
3. **Detection quality**: YOLO confidence affects position accuracy
4. **Air resistance**: Not modeled (assumes vacuum conditions)
5. **Ball spin**: Not considered in current model

### Limitations

- **Simplified physics**: Ignores air resistance and spin effects
- **2D projection**: Assumes motion in camera plane
- **Constant velocity assumption**: Uses linear velocity averaging
- **Detection noise**: YOLO detection jitter affects calculations

## Calibration Process

### Setting Pixels per Meter

1. **Measure known distance**: Find a known distance in the video (e.g., free-throw line = 4.57m)
2. **Count pixels**: Measure the same distance in pixels
3. **Calculate ratio**: `px_per_meter = pixel_distance / real_distance`

Example:
```python
# If free-throw line is 400 pixels and represents 4.57m
px_per_meter = 400 / 4.57  # ≈ 87.5 px/m
```

### Basket Position Calibration

1. **Click on basket**: Use mouse callback to set basket position
2. **Visual verification**: Basket visualization should align with actual basket
3. **Fine-tuning**: Adjust position for best probability calculations

## Error Analysis

### Common Sources of Error

1. **Velocity noise**: Rapid detection changes cause velocity spikes
2. **Calibration errors**: Incorrect `px_per_meter` affects all calculations
3. **Frame drops**: Missing frames create velocity calculation gaps
4. **Perspective distortion**: Camera angle affects accuracy

### Mitigation Strategies

1. **Velocity smoothing**: Average multiple velocity measurements
2. **Outlier filtering**: Remove unrealistic velocity values
3. **Minimum data requirements**: Require 5+ positions before prediction
4. **Boundary checking**: Stop predictions at reasonable limits

## Usage Example

```python
# Initialize with proper calibration
analyzer = BasketballTrajectoryAnalyzer()
analyzer.px_per_meter = 150  # Adjust based on your scene
analyzer.set_basket_position(560, 200)  # Click to set

# Process each frame
output, probability = analyzer.process_frame(frame)
```

## Mathematical Details

### Trajectory Formula Derivation

Starting from Newton's second law:
```
F = ma
mg = ma  (only gravity acts)
a = g    (downward acceleration)
```

Integrating for velocity:
```
v(t) = v₀ + at
vᵧ(t) = vᵧ₀ + gt  (vertical component)
vₓ(t) = vₓ₀       (horizontal component, no forces)
```

Integrating for position:
```
s(t) = s₀ + v₀t + ½at²
y(t) = y₀ + vᵧ₀t + ½gt²
x(t) = x₀ + vₓ₀t
```

### Coordinate Transformations

**Pixels to meters:**
```python
x_meters = x_pixels / px_per_meter
y_meters = y_pixels / px_per_meter
```

**Velocity from positions:**
```python
vx = (x_current - x_previous) * fps / px_per_meter
vy = (y_current - y_previous) * fps / px_per_meter
```

This physics-based approach provides realistic trajectory predictions that can be used for shot analysis and probability estimation.
