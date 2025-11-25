"""
Configuration constants for Basketball Trajectory Analyzer
"""

# Class IDs from data.yaml
BALL_CLASS = 0
PLAYER_CLASS = 1
RIM_CLASS = 2

# Physics constants
GRAVITY = 9.81  # m/s²
DEFAULT_FPS = 30
DEFAULT_PX_PER_METER = 150

# Detection parameters
DEFAULT_CONFIDENCE_THRESHOLD = 0.5
BALL_TRACK_HISTORY = 20
VELOCITY_HISTORY = 10

# Trajectory predictionc
MAX_TRAJECTORY_POINTS = 50
MIN_SPEED_THRESHOLD = 0.5
HEIGHT_THRESHOLD_PIXELS = 50  # pixels above player head

# Basket physics
BASKET_DIAMETER_M = 0.45  # meters
BASKET_EFFECTIVE_RADIUS_FACTOR = 0.8

# Shot probability zones
SHOT_ZONES = {
    'perfect': (0.3, 0.92),
    'excellent': (0.5, 0.85),
    'good': (0.7, 0.75),
    'decent': (1.0, 0.60),
    'poor': (1.2, 0.40),
    'very_poor': (1.5, 0.25),
    'miss': (2.0, 0.12),
    'complete_miss': (float('inf'), 0.02)
}

# Angle factors
OPTIMAL_ANGLE_RANGE = (35, 55)  # degrees
OPTIMAL_ANGLE_BONUS = 1.2
BAD_ANGLE_PENALTY = 0.6
BAD_ANGLE_RANGES = [(0, 20), (70, 90)]

# Colors (BGR format)
COLORS = {
    'ball_trail': (255, 255, 0),      # Cyan - excellent visibility on court
    'ball_current': (0, 165, 255),    # Orange - bright and visible
    'basket': (0, 255, 255),          # Yellow - excellent visibility on dark court
    'rim_detection': (0, 255, 255),   # Yellow - excellent visibility on dark court
    'trajectory': (255, 255, 255),    # White - maximum contrast on any background
    'probability_high': (0, 255, 0),  # Green - positive indicator
    'probability_med': (0, 165, 255), # Orange - neutral indicator
    'probability_low': (0, 0, 255),   # Red - warning indicator
    'pause': (0, 100, 200),           # Orange-red - less intense red
    'text': (255, 255, 255)           # White - best text readability
}

# Display settings
FONT_SCALE = 1.2
FONT_THICKNESS = 3
SMALL_FONT_SCALE = 0.7
SMALL_FONT_THICKNESS = 2
