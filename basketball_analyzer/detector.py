"""
Basketball object detection using YOLO
"""

from ultralytics import YOLO
from .config import BALL_CLASS, PLAYER_CLASS, RIM_CLASS


class BasketballDetector:
    """Handles basketball-related object detection using YOLO"""

    def __init__(self, model_path='shot.pt', confidence_threshold=0.3):
        self.model = YOLO(model_path)
        self.confidence_threshold = confidence_threshold

        # Basket tracking
        self.basket_pos = None

    def detect_objects(self, frame):
        """
        Detect basketball objects in frame

        Args:
            frame: Input video frame

        Returns:
            dict: Detection results for ball, player, and rim
        """
        results = self.model(frame, verbose=False)
        detections = {'ball': None, 'player': None, 'rim': None}

        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    cls = int(box.cls[0])
                    conf = float(box.conf[0])

                    if conf > self.confidence_threshold:
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        center_x = (x1 + x2) // 2
                        center_y = (y1 + y2) // 2

                        if cls == BALL_CLASS:
                            detections['ball'] = (center_x, center_y, x2-x1)
                        elif cls == PLAYER_CLASS:
                            player_top = y1
                            detections['player'] = (center_x, center_y, x2-x1, y2-y1, player_top)
                        elif cls == RIM_CLASS:
                            detections['rim'] = (center_x, center_y, x2-x1, y2-y1)
                            self.basket_pos = (center_x, center_y)

        return detections

    def is_ball_shot(self, ball_pos, player_detection, height_threshold=50):
        """
        Check if ball is above player height (indicating a shot)

        Args:
            ball_pos: Ball position tuple (x, y, width)
            player_detection: Player detection tuple
            height_threshold: Pixels above player head

        Returns:
            bool: True if ball is shot (above player)
        """
        if not ball_pos or not player_detection:
            return False

        ball_y = ball_pos[1]
        player_top_y = player_detection[4]  # Top of player bounding box

        return ball_y < (player_top_y - height_threshold)

    def get_basket_position(self):
        """Get current basket position"""
        return self.basket_pos

    def reset_basket_position(self):
        """Reset basket position for new detection"""
        self.basket_pos = None
