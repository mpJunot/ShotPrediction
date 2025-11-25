"""
Utility functions for basketball trajectory analyzer
"""

import cv2
import math
import numpy as np


def fps_to_wait_time(fps, speed_multiplier=1.0):
    """
    Convert FPS to OpenCV waitKey time

    Args:
        fps (int): Target frames per second
        speed_multiplier (float): Speed multiplier for playback

    Returns:
        int: Wait time in milliseconds
    """
    return max(1, int((1000 / fps) / speed_multiplier))


def distance_2d(point1, point2):
    """
    Calculate 2D distance between two points

    Args:
        point1 (tuple): (x, y) coordinates
        point2 (tuple): (x, y) coordinates

    Returns:
        float: Distance between points
    """
    return math.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)


def angle_between_points(point1, point2, point3):
    """
    Calculate angle between three points (point2 is the vertex)

    Args:
        point1 (tuple): First point (x, y)
        point2 (tuple): Vertex point (x, y)
        point3 (tuple): Third point (x, y)

    Returns:
        float: Angle in degrees
    """
    # Vectors from point2 to point1 and point3
    v1 = (point1[0] - point2[0], point1[1] - point2[1])
    v2 = (point3[0] - point2[0], point3[1] - point2[1])

    # Calculate angle using dot product
    dot_product = v1[0] * v2[0] + v1[1] * v2[1]
    magnitude1 = math.sqrt(v1[0]**2 + v1[1]**2)
    magnitude2 = math.sqrt(v2[0]**2 + v2[1]**2)

    if magnitude1 == 0 or magnitude2 == 0:
        return 0

    cos_angle = dot_product / (magnitude1 * magnitude2)
    cos_angle = max(-1, min(1, cos_angle))  # Clamp to [-1, 1]

    return math.degrees(math.acos(cos_angle))


def smooth_trajectory(points, window_size=5):
    """
    Smooth trajectory points using moving average

    Args:
        points (list): List of (x, y) points
        window_size (int): Size of smoothing window

    Returns:
        list: Smoothed points
    """
    if len(points) < window_size:
        return points

    smoothed = []
    half_window = window_size // 2

    for i in range(len(points)):
        start_idx = max(0, i - half_window)
        end_idx = min(len(points), i + half_window + 1)

        window_points = points[start_idx:end_idx]

        avg_x = sum(p[0] for p in window_points) / len(window_points)
        avg_y = sum(p[1] for p in window_points) / len(window_points)

        smoothed.append((int(avg_x), int(avg_y)))

    return smoothed


def calculate_velocity_magnitude(velocity_history):
    """
    Calculate velocity magnitude from velocity history

    Args:
        velocity_history (list): List of (vx, vy) velocity tuples

    Returns:
        float: Average velocity magnitude
    """
    if not velocity_history:
        return 0.0

    magnitudes = [math.sqrt(vx**2 + vy**2) for vx, vy in velocity_history]
    return sum(magnitudes) / len(magnitudes)


def pixels_to_meters(pixels, px_per_meter):
    """
    Convert pixels to meters

    Args:
        pixels (float): Distance in pixels
        px_per_meter (float): Pixels per meter conversion factor

    Returns:
        float: Distance in meters
    """
    return pixels / px_per_meter


def meters_to_pixels(meters, px_per_meter):
    """
    Convert meters to pixels

    Args:
        meters (float): Distance in meters
        px_per_meter (float): Pixels per meter conversion factor

    Returns:
        int: Distance in pixels
    """
    return int(meters * px_per_meter)


def clamp(value, min_value, max_value):
    """
    Clamp value between min and max

    Args:
        value (float): Value to clamp
        min_value (float): Minimum value
        max_value (float): Maximum value

    Returns:
        float: Clamped value
    """
    return max(min_value, min(value, max_value))


def normalize_angle(angle_degrees):
    """
    Normalize angle to [0, 360) range

    Args:
        angle_degrees (float): Angle in degrees

    Returns:
        float: Normalized angle
    """
    return angle_degrees % 360


def is_point_in_bounds(point, width, height, margin=0):
    """
    Check if point is within frame bounds

    Args:
        point (tuple): (x, y) coordinates
        width (int): Frame width
        height (int): Frame height
        margin (int): Margin from edges

    Returns:
        bool: True if point is in bounds
    """
    x, y = point
    return (margin <= x < width - margin and
            margin <= y < height - margin)
