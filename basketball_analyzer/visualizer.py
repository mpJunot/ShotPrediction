"""
Visualization and drawing functionality for basketball trajectory analysis
"""

import cv2
from .config import COLORS, FONT_SCALE, FONT_THICKNESS, SMALL_FONT_SCALE, SMALL_FONT_THICKNESS, PHASE_COLORS, SHOT_PHASE_NAMES


class TrajectoryVisualizer:
    """Handles visualization of basketball trajectory analysis"""

    def __init__(self, px_per_meter=150, show_ball_trail=True, show_trajectory=True,
                 show_detection_boxes=True, show_probability=True):
        self.px_per_meter = px_per_meter
        self.show_ball_trail = show_ball_trail
        self.show_trajectory = show_trajectory
        self.show_detection_boxes = show_detection_boxes
        self.show_probability = show_probability

    def draw_ball_trail(self, frame, ball_positions):
        """
        Draw ball trail/history

        Args:
            frame: Frame to draw on
            ball_positions: List of ball positions
        """
        if len(ball_positions) > 1:
            points = list(ball_positions)
            for i, point in enumerate(points):
                alpha = (i + 1) / len(points)
                radius = max(2, int(alpha * 5))
                intensity = int(255 * alpha)
                cv2.circle(frame, point, radius, (255, intensity, 0), -1)

    def draw_current_ball(self, frame, ball_detection):
        """
        Draw current ball position

        Args:
            frame: Frame to draw on
            ball_detection: Ball detection tuple (x, y, width)
        """
        if ball_detection:
            x, y, w = ball_detection
            cv2.circle(frame, (x, y), w//2, COLORS['ball_current'], 2)

    def draw_basket(self, frame, basket_pos):
        """
        Draw basket visualization

        Args:
            frame: Frame to draw on
            basket_pos: Basket position (x, y)
        """
        if basket_pos:
            basket_x, basket_y = basket_pos
            basket_radius = int(0.45 * self.px_per_meter / 2)

            # Draw basket rim as thick yellow line
            ##cv2.line(frame, (basket_x - basket_radius, basket_y), (basket_x + basket_radius, basket_y), COLORS['basket'], 4)

            # Draw center point
            cv2.circle(frame, (basket_x, basket_y), 4, COLORS['basket'], -1)

            # Draw basket circle outline
            ## cv2.circle(frame, (basket_x, basket_y), basket_radius, COLORS['basket'], 2)

    def draw_rim_detection(self, frame, rim_detection):
        """
        Draw rim detection box

        Args:
            frame: Frame to draw on
            rim_detection: Rim detection tuple
        """
        if rim_detection:
            x, y, w, h = rim_detection
            cv2.rectangle(frame, (x-w//2, y-h//2), (x+w//2, y+h//2), COLORS['rim_detection'], 2)

    def draw_trajectory(self, frame, trajectory):
        """
        Draw predicted trajectory

        Args:
            frame: Frame to draw on
            trajectory: List of trajectory points
        """
        for i, point in enumerate(trajectory):
            alpha = 1.0 - (i / len(trajectory)) if len(trajectory) > 0 else 1.0
            point_size = max(2, int(alpha * 5.3))
            cv2.circle(frame, point, point_size, COLORS['trajectory'], -1)

    def draw_shot_probability(self, frame, probability):
        """
        Draw shot probability text

        Args:
            frame: Frame to draw on
            probability: Shot probability (0-1)
        """
        if probability > 0:
            prob_text = f"Shot: {probability*100:.0f}%"

            # Choose color based on probability
            if probability > 0.6:
                color = COLORS['probability_high']
            elif probability > 0.3:
                color = COLORS['probability_med']
            else:
                color = COLORS['probability_low']

            cv2.putText(frame, prob_text, (20, 50), cv2.FONT_HERSHEY_SIMPLEX,
                       FONT_SCALE, color, FONT_THICKNESS)

    def draw_pause_indicator(self, frame):
        """
        Draw pause indicator

        Args:
            frame: Frame to draw on
        """
        cv2.putText(frame, "PAUSED", (20, frame.shape[0]//2),
                   cv2.FONT_HERSHEY_SIMPLEX, 2, COLORS['pause'], 3)

    def draw_phase_boxes(self, frame, phase_detections):
        """
        Draw bounding boxes for detected shot phases

        Args:
            frame: Frame to draw on
            phase_detections: Dictionary of detected phases {class_id: phase_info}
        """
        if not phase_detections:
            return

        for class_id, phase_info in phase_detections.items():
            bbox = phase_info.get('bbox')
            confidence = phase_info.get('confidence', 0.0)
            position = phase_info.get('position')

            if bbox:
                x1, y1, x2, y2 = bbox
                color = PHASE_COLORS.get(class_id, COLORS['trajectory'])

                # Draw bounding box
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

                # Draw label with phase name and confidence
                phase_name = SHOT_PHASE_NAMES.get(class_id, f"Phase {class_id}")
                label = f"{phase_name}: {confidence:.2f}"

                # Calculate text size for background
                (text_width, text_height), baseline = cv2.getTextSize(
                    label, cv2.FONT_HERSHEY_SIMPLEX, SMALL_FONT_SCALE, SMALL_FONT_THICKNESS
                )

                # Draw background rectangle for text
                cv2.rectangle(
                    frame,
                    (x1, y1 - text_height - baseline - 5),
                    (x1 + text_width, y1),
                    color,
                    -1
                )

                # Draw text
                cv2.putText(
                    frame,
                    label,
                    (x1, y1 - baseline - 2),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    SMALL_FONT_SCALE,
                    (255, 255, 255),  # White text
                    SMALL_FONT_THICKNESS
                )

    def draw_detection_status(self, frame, detections):
        """
        Draw detection status information

        Args:
            frame: Frame to draw on
            detections: Detection results
        """
        y_offset = 80

        # Ball detection status
        ball_status = "Ball: DETECTED" if detections.get('ball') else "Ball: NOT DETECTED"
        ball_color = COLORS['probability_high'] if detections.get('ball') else COLORS['probability_low']
        cv2.putText(frame, ball_status, (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX,
                   SMALL_FONT_SCALE, ball_color, SMALL_FONT_THICKNESS)

        # Rim detection status
        rim_status = "Rim: DETECTED" if detections.get('rim') else "Rim: NOT DETECTED"
        rim_color = COLORS['probability_high'] if detections.get('rim') else COLORS['probability_low']
        cv2.putText(frame, rim_status, (20, y_offset + 25), cv2.FONT_HERSHEY_SIMPLEX,
                   SMALL_FONT_SCALE, rim_color, SMALL_FONT_THICKNESS)

    def draw_complete_analysis(self, frame, detections, trajectory, probability,
                             ball_positions, basket_pos, phase_detections=None):
        """
        Draw complete trajectory analysis

        Args:
            frame: Frame to draw on
            detections: Detection results
            trajectory: Predicted trajectory
            probability: Shot probability
            ball_positions: Ball position history
            basket_pos: Basket position
            phase_detections: Dictionary of detected shot phases (optional)
        Returns:
            numpy.ndarray: Frame with analysis drawn
        """
        output = frame.copy()

        # Draw elements based on configuration
        if self.show_ball_trail:
            self.draw_ball_trail(output, ball_positions)

        self.draw_current_ball(output, detections.get('ball'))
        self.draw_basket(output, basket_pos)

        if self.show_detection_boxes:
            self.draw_rim_detection(output, detections.get('rim'))
            # Draw phase detection boxes if available
            if phase_detections:
                self.draw_phase_boxes(output, phase_detections)

        if self.show_trajectory:
            self.draw_trajectory(output, trajectory)

        if self.show_probability:
            self.draw_shot_probability(output, probability)

        self.draw_detection_status(output, detections)

        return output

    def update_visualization_options(self, show_ball_trail=None, show_trajectory=None,
                                   show_detection_boxes=None, show_probability=None):
        """
        Update visualization options

        Args:
            show_ball_trail: Show ball trail (None to keep current)
            show_trajectory: Show predicted trajectory (None to keep current)
            show_detection_boxes: Show detection boxes (None to keep current)
            show_probability: Show shot probability (None to keep current)
        """
        if show_ball_trail is not None:
            self.show_ball_trail = show_ball_trail
        if show_trajectory is not None:
            self.show_trajectory = show_trajectory
        if show_detection_boxes is not None:
            self.show_detection_boxes = show_detection_boxes
        if show_probability is not None:
            self.show_probability = show_probability
