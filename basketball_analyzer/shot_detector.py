"""
Advanced shot detection logic for basketball trajectory analysis
"""

import math
from collections import deque
from .config import *


class ShotDetector:
    """Detects if the ball movement constitutes a genuine basketball shot"""

    def __init__(self, fps=DEFAULT_FPS, px_per_meter=DEFAULT_PX_PER_METER):
        self.fps = fps
        self.px_per_meter = px_per_meter

        # History for analysis
        self.velocity_history = deque(maxlen=10)
        self.acceleration_history = deque(maxlen=5)
        self.height_history = deque(maxlen=15)

        # Detection thresholds
        self.min_shot_speed = 3.0  # m/s
        self.max_shot_speed = 15.0  # m/s
        self.min_upward_velocity = 1.5  # m/s upward
        self.min_height_gain = 0.3  # meters of height gain
        self.shot_angle_range = (15, 75)  # degrees of acceptable shot angle

    def update_ball_data(self, ball_pos, basket_pos):
        """
        Updates ball data for analysis

        Args:
            ball_pos: Ball position (x, y, width)
            basket_pos: Basket position (x, y)
        """
        if not ball_pos:
            return

        x, y, _ = ball_pos
        current_time = len(self.height_history)

        # Store height (y inverted since 0,0 is top-left)
        height_meters = y / self.px_per_meter
        self.height_history.append((current_time, height_meters))

        # Calculate velocity if we have enough data
        if len(self.height_history) >= 2:
            self._calculate_velocity()

        # Calculate acceleration if we have enough velocities
        if len(self.velocity_history) >= 2:
            self._calculate_acceleration()

    def _calculate_velocity(self):
        """Calculate current ball velocity"""
        if len(self.height_history) < 2:
            return

        current = self.height_history[-1]
        previous = self.height_history[-2]

        dt = (current[0] - previous[0]) / self.fps
        if dt > 0:
            # Vertical velocity (positive = upward)
            vy = -(current[1] - previous[1]) / dt
            self.velocity_history.append((current[0], vy))

    def _calculate_acceleration(self):
        """Calculate current ball acceleration"""
        if len(self.velocity_history) < 2:
            return

        current_v = self.velocity_history[-1]
        previous_v = self.velocity_history[-2]

        dt = (current_v[0] - previous_v[0]) / self.fps
        if dt > 0:
            acceleration = (current_v[1] - previous_v[1]) / dt
            self.acceleration_history.append(acceleration)

    def is_genuine_shot(self, ball_pos, player_pos, basket_pos):
        """
        Determines if the ball movement constitutes a genuine shot

        Args:
            ball_pos: Ball position (x, y, width)
            player_pos: Player position (x, y, width, height, top)
            basket_pos: Basket position (x, y)

        Returns:
            dict: Analysis results with confidence score
        """
        if not all([ball_pos, basket_pos]):
            return {'is_shot': False, 'confidence': 0.0, 'reasons': ['Missing detections']}

        analysis = {
            'is_shot': False,
            'confidence': 0.0,
            'reasons': [],
            'details': {}
        }

        # 1. Check minimum speed
        speed_check = self._check_ball_speed()
        analysis['details']['speed'] = speed_check

        # 2. Check direction (upward initially)
        direction_check = self._check_upward_motion()
        analysis['details']['direction'] = direction_check

        # 3. Check shot angle
        angle_check = self._check_shot_angle(ball_pos, basket_pos)
        analysis['details']['angle'] = angle_check

        # 4. Check position relative to player
        player_check = self._check_player_release(ball_pos, player_pos)
        analysis['details']['player_release'] = player_check

        # 5. Check height gain
        height_check = self._check_height_gain()
        analysis['details']['height_gain'] = height_check

        # 6. Check direction towards basket
        basket_direction_check = self._check_basket_direction(ball_pos, basket_pos)
        analysis['details']['basket_direction'] = basket_direction_check

        # Calculate confidence score
        confidence_factors = [
            speed_check['valid'] * 0.25,
            direction_check['valid'] * 0.20,
            angle_check['valid'] * 0.15,
            player_check['valid'] * 0.15,
            height_check['valid'] * 0.15,
            basket_direction_check['valid'] * 0.10
        ]

        analysis['confidence'] = sum(confidence_factors)
        analysis['is_shot'] = analysis['confidence'] > 0.6  # 60% threshold

        # Compile reasons
        if not analysis['is_shot']:
            if not speed_check['valid']:
                analysis['reasons'].append(f"Insufficient speed ({speed_check['value']:.1f} m/s)")
            if not direction_check['valid']:
                analysis['reasons'].append("No initial upward movement")
            if not angle_check['valid']:
                analysis['reasons'].append(f"Inappropriate shot angle ({angle_check['value']:.1f}°)")
            if not player_check['valid']:
                analysis['reasons'].append("Ball not above player")
            if not height_check['valid']:
                analysis['reasons'].append("Insufficient height gain")
            if not basket_direction_check['valid']:
                analysis['reasons'].append("Wrong direction towards basket")

        return analysis

    def _check_ball_speed(self):
        """Check if ball speed is within shot range"""
        if len(self.velocity_history) < 3:
            return {'valid': False, 'value': 0.0, 'reason': 'Insufficient data'}

        # Take initial speed (first measurements)
        initial_velocities = list(self.velocity_history)[:3]
        avg_speed = sum(abs(v[1]) for v in initial_velocities) / len(initial_velocities)

        valid = self.min_shot_speed <= avg_speed <= self.max_shot_speed
        return {
            'valid': valid,
            'value': avg_speed,
            'reason': 'Appropriate speed' if valid else f'Speed out of range ({self.min_shot_speed}-{self.max_shot_speed} m/s)'
        }

    def _check_upward_motion(self):
        """Check if ball has initial upward movement"""
        if len(self.velocity_history) < 2:
            return {'valid': False, 'value': 0.0, 'reason': 'Insufficient data'}

        # Check first 2-3 velocities
        initial_velocities = list(self.velocity_history)[:3]
        upward_count = sum(1 for v in initial_velocities if v[1] > self.min_upward_velocity)

        valid = upward_count >= len(initial_velocities) * 0.6  # 60% of initial measurements
        avg_upward = sum(max(0, v[1]) for v in initial_velocities) / len(initial_velocities)

        return {
            'valid': valid,
            'value': avg_upward,
            'reason': 'Initial upward movement' if valid else 'No sufficient upward motion'
        }

    def _check_shot_angle(self, ball_pos, basket_pos):
        """Check if angle towards basket is appropriate for a shot"""
        if not self.velocity_history or len(self.velocity_history) < 2:
            return {'valid': False, 'value': 0.0, 'reason': 'Insufficient data'}

        ball_x, ball_y, _ = ball_pos
        basket_x, basket_y = basket_pos

        # Calculate angle towards basket
        dx = basket_x - ball_x
        dy = basket_y - ball_y  # y positive downward

        if abs(dx) < 1:  # Avoid division by zero
            return {'valid': False, 'value': 0.0, 'reason': 'Same horizontal position'}

        angle_to_basket = math.degrees(math.atan(-dy / abs(dx)))  # Angle upward

        valid = self.shot_angle_range[0] <= angle_to_basket <= self.shot_angle_range[1]

        return {
            'valid': valid,
            'value': angle_to_basket,
            'reason': 'Correct shot angle' if valid else f'Angle out of range ({self.shot_angle_range[0]}-{self.shot_angle_range[1]}°)'
        }

    def _check_player_release(self, ball_pos, player_pos):
        """Check if ball is above player (shot released)"""
        if not player_pos:
            return {'valid': True, 'value': 0.0, 'reason': 'Player not detected - assumed valid'}

        ball_x, ball_y, _ = ball_pos
        player_x, player_y, _, _, player_top = player_pos

        # Horizontal distance between ball and player
        horizontal_distance = abs(ball_x - player_x)

        # Relative height (ball above player head)
        height_above_player = player_top - ball_y  # Positive if ball above

        # Ball should be reasonably above player
        valid = height_above_player > 20 and horizontal_distance < 100  # pixels

        return {
            'valid': valid,
            'value': height_above_player,
            'reason': 'Ball released above player' if valid else 'Ball not in shooting position'
        }

    def _check_height_gain(self):
        """Check if ball has gained sufficient height"""
        if len(self.height_history) < 5:
            return {'valid': False, 'value': 0.0, 'reason': 'Insufficient data'}

        # Compare current height vs initial height
        initial_height = self.height_history[0][1]
        recent_heights = [h[1] for h in list(self.height_history)[-3:]]
        max_height = min(recent_heights)  # Min because y inverted

        height_gain = initial_height - max_height  # Positive gain

        valid = height_gain > self.min_height_gain

        return {
            'valid': valid,
            'value': height_gain,
            'reason': f'Height gain: {height_gain:.2f}m' if valid else f'Insufficient gain: {height_gain:.2f}m'
        }

    def _check_basket_direction(self, ball_pos, basket_pos):
        """Check if ball is heading towards basket"""
        if len(self.height_history) < 3:
            return {'valid': False, 'value': 0.0, 'reason': 'Insufficient data'}

        ball_x, ball_y, _ = ball_pos
        basket_x, basket_y = basket_pos

        # Calculate recent movement direction
        recent_positions = list(self.height_history)[-3:]
        if len(recent_positions) < 2:
            return {'valid': False, 'value': 0.0, 'reason': 'Insufficient positions'}

        # Rough estimation of direction (simplified)
        distance_to_basket = math.sqrt((ball_x - basket_x)**2 + (ball_y - basket_y)**2)

        # If ball is reasonably close horizontally to basket
        horizontal_distance = abs(ball_x - basket_x)
        valid = horizontal_distance < 200  # pixels - reasonable shooting zone

        return {
            'valid': valid,
            'value': distance_to_basket,
            'reason': 'Direction towards basket' if valid else 'Too far horizontally from basket'
        }

    def reset(self):
        """Reset history for new analysis"""
        self.velocity_history.clear()
        self.acceleration_history.clear()
        self.height_history.clear()

    def get_shot_quality_score(self, analysis_result):
        """
        Evaluate shot quality based on analysis

        Args:
            analysis_result: Result from is_genuine_shot()

        Returns:
            str: Shot quality ('Excellent', 'Good', 'Average', 'Poor')
        """
        if not analysis_result['is_shot']:
            return 'Not a shot'

        confidence = analysis_result['confidence']

        if confidence >= 0.9:
            return 'Excellent'
        elif confidence >= 0.8:
            return 'Good'
        elif confidence >= 0.7:
            return 'Average'
        else:
            return 'Poor'
