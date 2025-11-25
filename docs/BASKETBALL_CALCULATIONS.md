# Basketball Trajectory Analyzer - Mathematical Calculations

## Overview

This document outlines the mathematical calculations and algorithms used in the Basketball Trajectory Analyzer system. The system combines computer vision (YOLO detection) with physics-based trajectory prediction to analyze basketball shots.

## 1. Physics-Based Trajectory Calculations

### 1.1 Projectile Motion Equations

The basketball follows projectile motion principles under gravity:

**Position Equations:**
```
x(t) = x₀ + v₀ₓ × t
y(t) = y₀ + v₀ᵧ × t - ½ × g × t²
z(t) = z₀ + v₀ᵤ × t
```

Where:
- `(x₀, y₀, z₀)` = Initial position
- `(v₀ₓ, v₀ᵧ, v₀ᵤ)` = Initial velocity components
- `g` = Gravitational acceleration (9.81 m/s²)
- `t` = Time

**Velocity Equations:**
```
vₓ(t) = v₀ₓ
vᵧ(t) = v₀ᵧ - g × t
vᵤ(t) = v₀ᵤ
```

### 1.2 Launch Angle and Initial Velocity

**Launch Angle (θ):**
```
θ = arctan(v₀ᵧ / √(v₀ₓ² + v₀ᵤ²))
```

**Initial Speed:**
```
v₀ = √(v₀ₓ² + v₀ᵧ² + v₀ᵤ²)
```

### 1.3 Flight Time and Range

**Time to reach maximum height:**
```
t_max = v₀ᵧ / g
```

**Maximum height:**
```
h_max = y₀ + (v₀ᵧ²) / (2 × g)
```

**Total flight time (assuming landing at same height):**
```
t_flight = 2 × v₀ᵧ / g
```

**Range (horizontal distance):**
```
R = v₀ₓ × t_flight
```

## 2. Detection and Tracking Calculations

### 2.1 YOLO Bounding Box Calculations

**Bounding Box Center:**
```
center_x = bbox_x + bbox_width / 2
center_y = bbox_y + bbox_height / 2
```

**Confidence Score Filtering:**
```
valid_detection = confidence_score > threshold
```

### 2.2 Ball Position Estimation

**3D Position from 2D Detection:**
```
# Camera calibration matrix transformation
world_coords = K⁻¹ × pixel_coords × depth
```

Where `K` is the camera intrinsic matrix.

### 2.3 Velocity Calculation

**Finite Difference Method:**
```
vₓ = (x(t+Δt) - x(t-Δt)) / (2 × Δt)
vᵧ = (y(t+Δt) - y(t-Δt)) / (2 × Δt)
vᵤ = (z(t+Δt) - z(t-Δt)) / (2 × Δt)
```

**Smoothed Velocity (Moving Average):**
```
v_smooth(t) = Σ(v(t-i)) / n  for i = 0 to n-1
```

## 3. Trajectory Prediction Algorithms

### 3.1 Least Squares Fitting

For fitting trajectory to detected points:

**Parabolic Fit (2D):**
```
y = ax² + bx + c
```

**Matrix Form:**
```
[y₁]   [x₁² x₁ 1] [a]
[y₂] = [x₂² x₂ 1] [b]
[y₃]   [x₃² x₃ 1] [c]
```

**Solution:**
```
θ = (XᵀX)⁻¹XᵀY
```

### 3.2 Kalman Filter for Tracking

**State Vector:**
```
x = [x, y, z, vₓ, vᵧ, vᵤ]ᵀ
```

**State Transition Matrix:**
```
F = [1 0 0 Δt 0  0 ]
    [0 1 0 0  Δt 0 ]
    [0 0 1 0  0  Δt]
    [0 0 0 1  0  0 ]
    [0 0 0 0  1  0 ]
    [0 0 0 0  0  1 ]
```

**Process Noise (with gravity):**
```
G = [½Δt² 0     0   ]
    [0     ½Δt² 0   ]
    [0     0     ½Δt²]
    [Δt    0     0   ]
    [0     Δt    0   ]
    [0     0     Δt  ]
```

## 4. Shot Analysis Metrics

### 4.1 Arc Calculation

**Arc Height:**
```
arc_height = max(y_trajectory) - y_release
```

**Arc Angle at Release:**
```
arc_angle = arctan((y_peak - y_release) / (x_peak - x_release))
```

### 4.2 Shot Accuracy Metrics

**Distance to Basket:**
```
distance = √((x_basket - x_ball)² + (z_basket - z_ball)²)
```

**Entry Angle:**
```
entry_angle = arctan(vᵧ_final / √(vₓ_final² + vᵤ_final²))
```

**Shot Probability (based on arc and entry angle):**
```
P_make = sigmoid(w₁ × arc_score + w₂ × entry_score + bias)
```

### 4.3 Release Point Analysis

**Release Height:**
```
h_release = y_coordinate_at_release
```

**Release Speed:**
```
v_release = √(vₓ² + vᵧ² + vᵤ²) at release point
```

**Release Angle:**
```
θ_release = arctan(vᵧ / √(vₓ² + vᵤ²))
```

## 5. Error Metrics and Validation

### 5.1 Prediction Error

**Mean Squared Error:**
```
MSE = (1/n) × Σ(y_predicted - y_actual)²
```

**Root Mean Square Error:**
```
RMSE = √MSE
```

### 5.2 Trajectory Fit Quality

**R-squared (Coefficient of Determination):**
```
R² = 1 - (SS_res / SS_tot)
```

Where:
- `SS_res = Σ(y_actual - y_predicted)²`
- `SS_tot = Σ(y_actual - y_mean)²`

### 5.3 Detection Accuracy

**Intersection over Union (IoU):**
```
IoU = Area_intersection / Area_union
```

**Tracking Accuracy:**
```
tracking_error = √((x_true - x_estimated)² + (y_true - y_estimated)²)
```

## 6. Advanced Calculations

### 6.1 Air Resistance Model

For more accurate trajectory with drag:

**Drag Force:**
```
F_drag = ½ × ρ × Cd × A × v²
```

**Modified Equations of Motion:**
```
aₓ = -k × vₓ × |v|
aᵧ = -g - k × vᵧ × |v|
aᵤ = -k × vᵤ × |v|
```

Where `k = ρ × Cd × A / (2 × m)`

### 6.2 Spin Effects (Magnus Force)

**Magnus Force:**
```
F_magnus = ½ × ρ × Cl × A × v² × (ω × v) / |ω × v|
```

### 6.3 Shot Classification

**Make/Miss Prediction:**
```
score = w₁×arc + w₂×entry_angle + w₃×release_speed + w₄×distance
prediction = sigmoid(score) > threshold
```

## Constants Used

- **Gravity:** g = 9.81 m/s²
- **Basketball diameter:** d = 0.24 m
- **Basket height:** h = 3.05 m
- **Basket diameter:** D = 0.46 m
- **Air density (sea level):** ρ = 1.225 kg/m³
- **Basketball mass:** m ≈ 0.62 kg
- **Drag coefficient:** Cd ≈ 0.47

## Usage Notes

1. All calculations assume metric units (meters, seconds)
2. Camera calibration is required for accurate 3D position estimation
3. Frame rate affects velocity calculation accuracy
4. Environmental factors (wind, air pressure) may affect real-world accuracy
5. Detection confidence thresholds should be tuned based on lighting conditions
