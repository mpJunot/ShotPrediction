"""
Shot phase detection using YOLO model for detecting different phases of a basketball shot
"""

from ultralytics import YOLO
from collections import deque
from .config import (SHOT_PHASE_NAMES, SHOT_RELEASE_CLASS, SHOT_FOLLOWTHROUGH_CLASS,
                     PHASE_DETECTION_CONFIDENCE_THRESHOLD)


class ShotPhaseDetector:
    """Detects different phases of a basketball shot using specialized YOLO model"""

    def __init__(self, model_path='copyme.pt', confidence_threshold=None):
        """
        Initialize shot phase detector

        Args:
            model_path: Path to the shot phase detection model
            confidence_threshold: Minimum confidence for detections (default: PHASE_DETECTION_CONFIDENCE_THRESHOLD)
        """
        self.model = YOLO(model_path)
        self.confidence_threshold = confidence_threshold if confidence_threshold is not None else PHASE_DETECTION_CONFIDENCE_THRESHOLD

        # Phase history for tracking
        self.phase_history = deque(maxlen=30)
        self.current_phase = None
        # Track when followthrough was last detected (to stop prediction after)
        self.last_followthrough_frame = None
        self.frame_count = 0

    def detect_phases(self, frame):
        """
        Detect shot phases in frame

        Args:
            frame: Input video frame

        Returns:
            dict: Detected phases with their positions and confidences
        """
        self.frame_count += 1
        results = self.model(frame, verbose=False)
        detected_phases = {}

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
                        width = x2 - x1
                        height = y2 - y1

                        # Store phase detection
                        phase_info = {
                            'class': cls,
                            'confidence': conf,
                            'position': (center_x, center_y),
                            'bbox': (x1, y1, x2, y2),
                            'size': (width, height)
                        }

                        # Store by class (phase type)
                        if cls not in detected_phases or detected_phases[cls]['confidence'] < conf:
                            detected_phases[cls] = phase_info

        if detected_phases:
            # Get the phase with highest confidence
            best_phase = max(detected_phases.values(), key=lambda x: x['confidence'])

            if best_phase['confidence'] >= self.confidence_threshold:
                self.current_phase = best_phase['class']
                self.phase_history.append({
                    'phase': self.current_phase,
                    'confidence': best_phase['confidence'],
                    'position': best_phase['position']
                })

                if self.current_phase == SHOT_FOLLOWTHROUGH_CLASS:
                    self.last_followthrough_frame = self.frame_count
        else:
            if self.phase_history:
                self.current_phase = self.phase_history[-1]['phase']
            else:
                self.current_phase = None

        return detected_phases

    def get_current_phase(self):
        """
        Get current detected phase

        Returns:
            int or None: Current phase class ID
        """
        return self.current_phase

    def get_current_phase_name(self):
        """
        Get current detected phase name

        Returns:
            str or None: Current phase name (e.g., "Shot Release")
        """
        if self.current_phase is not None:
            return SHOT_PHASE_NAMES.get(self.current_phase, f"Unknown Phase {self.current_phase}")
        return None

    def get_phase_history(self):
        """
        Get phase detection history

        Returns:
            list: List of recent phase detections
        """
        return list(self.phase_history)

    def get_phase_sequence(self):
        """
        Get sequence of phases detected over time

        Returns:
            list: Sequence of phase class IDs
        """
        return [p['phase'] for p in self.phase_history]

    def has_required_phases(self, min_confidence=None):
        """
        Check if both shot_release and shot_followthrough have been detected with high confidence

        Args:
            min_confidence: Minimum confidence threshold (default: uses confidence_threshold)

        Returns:
            bool: True if both required phases have been detected with high confidence
        """
        if not self.phase_history:
            return False

        min_conf = min_confidence if min_confidence is not None else self.confidence_threshold

        has_release = False
        has_followthrough = False

        for phase_entry in self.phase_history:
            phase = phase_entry['phase']
            confidence = phase_entry['confidence']

            if phase == SHOT_RELEASE_CLASS and confidence >= min_conf:
                has_release = True
            elif phase == SHOT_FOLLOWTHROUGH_CLASS and confidence >= min_conf:
                has_followthrough = True

        return has_release and has_followthrough

    def is_in_active_shot_phase(self, frames_after_followthrough=10, min_confidence=None):
        """
        Check if we are still in an active shot phase (release or followthrough)
        or within a short window after followthrough, with high confidence

        Args:
            frames_after_followthrough: Number of frames to continue prediction after followthrough
            min_confidence: Minimum confidence threshold (default: uses confidence_threshold)

        Returns:
            bool: True if still in active phase with high confidence or recently finished followthrough
        """
        if not self.phase_history:
            return False

        min_conf = min_confidence if min_confidence is not None else self.confidence_threshold

        # Check if currently detecting release or followthrough with high confidence
        if self.current_phase in [SHOT_RELEASE_CLASS, SHOT_FOLLOWTHROUGH_CLASS]:
            # Check if the current phase detection has high confidence
            if self.phase_history:
                last_phase = self.phase_history[-1]
                if last_phase['phase'] == self.current_phase and last_phase['confidence'] >= min_conf:
                    return True

        # Check if we recently detected followthrough with high confidence (within window)
        if self.last_followthrough_frame is not None:
            frames_since_followthrough = self.frame_count - self.last_followthrough_frame
            if frames_since_followthrough <= frames_after_followthrough:
                # Verify the followthrough detection had high confidence
                for phase_entry in reversed(self.phase_history):
                    if phase_entry['phase'] == SHOT_FOLLOWTHROUGH_CLASS:
                        if phase_entry['confidence'] >= min_conf:
                            return True
                        break

        return False

    def reset(self):
        """Reset phase history"""
        self.phase_history.clear()
        self.current_phase = None
        self.last_followthrough_frame = None
        self.frame_count = 0

