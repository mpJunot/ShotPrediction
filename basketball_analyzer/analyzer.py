"""
Main basketball trajectory analyzer class
"""

from .detector import BasketballDetector
from .trajectory import TrajectoryPredictor
from .visualizer import TrajectoryVisualizer
from .config import *


class BasketballTrajectoryAnalyzer:
    """Main analyzer class that coordinates all components"""

    def __init__(self, model_path='shot.pt', fps=DEFAULT_FPS, px_per_meter=DEFAULT_PX_PER_METER,
                 show_ball_trail=True, show_trajectory=True, show_detection_boxes=True, show_probability=True):
        # Initialize components
        self.detector = BasketballDetector(model_path)
        self.trajectory_predictor = TrajectoryPredictor(fps, px_per_meter)
        self.visualizer = TrajectoryVisualizer(
            px_per_meter,
            show_ball_trail=show_ball_trail,
            show_trajectory=show_trajectory,
            show_detection_boxes=show_detection_boxes,
            show_probability=show_probability
        )

        # Settings
        self.fps = fps
        self.px_per_meter = px_per_meter

    def process_frame(self, frame):
        """
        Process a single frame

        Args:
            frame: Input video frame

        Returns:
            tuple: (output_frame, shot_probability)
        """
        # Detect objects
        detections = self.detector.detect_objects(frame)

        # Update ball tracking
        self.trajectory_predictor.update_tracking(detections['ball'])

        trajectory = []
        probability = 0.0

        # Predict trajectory only if BOTH ball and basket are detected AND we have enough data
        if (self.trajectory_predictor.has_sufficient_tracking() and
            detections['ball'] is not None and
            self.detector.get_basket_position() is not None):

            current_pos = self.trajectory_predictor.get_current_position()

            # Check if ball is shot (above player)
            ball_is_shot = self.detector.is_ball_shot(detections['ball'], detections['player'])

            if ball_is_shot:
                vx, vy = self.trajectory_predictor.get_current_velocity()
                speed = self.trajectory_predictor.get_ball_speed()

                if speed > MIN_SPEED_THRESHOLD:
                    trajectory = self.trajectory_predictor.predict_trajectory(current_pos, vx, vy)
                    probability = self.trajectory_predictor.calculate_shot_probability(
                        trajectory, self.detector.get_basket_position()
                    )

        # Create visualization
        output = self.visualizer.draw_complete_analysis(
            frame, detections, trajectory, probability,
            self.trajectory_predictor.get_ball_positions(),
            self.detector.get_basket_position()
        )

        return output, probability

    def reset_basket_position(self):
        """Reset basket position for new detection"""
        self.detector.reset_basket_position()

    def set_calibration(self, px_per_meter):
        """Set pixels per meter calibration"""
        self.px_per_meter = px_per_meter
        self.trajectory_predictor.px_per_meter = px_per_meter
        self.visualizer.px_per_meter = px_per_meter

    def set_visualization_options(self, show_ball_trail=None, show_trajectory=None,
                                 show_detection_boxes=None, show_probability=None):
        """Update visualization options"""
        self.visualizer.update_visualization_options(
            show_ball_trail=show_ball_trail,
            show_trajectory=show_trajectory,
            show_detection_boxes=show_detection_boxes,
            show_probability=show_probability
        )

    def get_statistics(self):
        """
        Get current analysis statistics

        Returns:
            dict: Statistics dictionary
        """
        return {
            'ball_positions_count': len(self.trajectory_predictor.ball_positions),
            'velocity_history_count': len(self.trajectory_predictor.velocity_history),
            'current_speed': self.trajectory_predictor.get_ball_speed(),
            'basket_position': self.detector.get_basket_position(),
        }
