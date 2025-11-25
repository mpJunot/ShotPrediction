"""
Basketball Trajectory Analyzer Package

A comprehensive basketball trajectory analysis system using YOLO detection
and physics-based trajectory prediction.
"""

from .analyzer import BasketballTrajectoryAnalyzer
from .detector import BasketballDetector
from .trajectory import TrajectoryPredictor
from .visualizer import TrajectoryVisualizer
from .shot_phase_detector import ShotPhaseDetector
from .utils import *

__version__ = "1.0.0"
__author__ = "Basketball Analyzer Team"

__all__ = [
    'BasketballTrajectoryAnalyzer',
    'BasketballDetector',
    'TrajectoryPredictor',
    'TrajectoryVisualizer',
    'ShotPhaseDetector',
]
