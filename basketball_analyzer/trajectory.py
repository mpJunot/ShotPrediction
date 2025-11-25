"""
Basketball trajectory prediction and shot probability calculation
"""

import math
from collections import deque
from .config import *


class TrajectoryPredictor:
    """Handles basketball trajectory prediction and shot analysis"""

    def __init__(self, fps=DEFAULT_FPS, px_per_meter=DEFAULT_PX_PER_METER):
        self.fps = fps
        self.px_per_meter = px_per_meter
        self.gravity = GRAVITY

        # Ball tracking
        self.ball_positions = deque(maxlen=BALL_TRACK_HISTORY)
        self.velocity_history = deque(maxlen=VELOCITY_HISTORY)

    def update_tracking(self, ball_detection):
        """
        Update ball position tracking

        Args:
            ball_detection: Ball detection tuple (x, y, width)
        """
        if ball_detection:
            x, y, _ = ball_detection
            self.ball_positions.append((x, y))

            if len(self.ball_positions) >= 2:
                prev_pos = self.ball_positions[-2]
                current_pos = self.ball_positions[-1]

                # Calculate velocity
                vx_pixels = current_pos[0] - prev_pos[0]
                vy_pixels = current_pos[1] - prev_pos[1]

                vx = vx_pixels * self.fps / self.px_per_meter
                vy = vy_pixels * self.fps / self.px_per_meter

                self.velocity_history.append((vx, vy))

    def get_current_velocity(self):
        """
        Get current velocity by averaging recent measurements

        Returns:
            tuple: (vx, vy) velocity in m/s
        """
        if len(self.velocity_history) == 0:
            return 0, 0

        recent_velocities = list(self.velocity_history)[-5:]
        avg_vx = sum(v[0] for v in recent_velocities) / len(recent_velocities)
        avg_vy = sum(v[1] for v in recent_velocities) / len(recent_velocities)

        return avg_vx, avg_vy

    def get_current_position(self):
        """Get current ball position"""
        if len(self.ball_positions) == 0:
            return None
        return self.ball_positions[-1]

    def predict_trajectory(self, current_pos, vx, vy, num_points=MAX_TRAJECTORY_POINTS):
        """
        Predict parabolic trajectory using physics

        Args:
            current_pos: Current ball position (x, y)
            vx, vy: Velocity components in m/s
            num_points: Number of trajectory points to calculate

        Returns:
            list: Trajectory points [(x, y), ...]
        """
        if vx is None or vy is None or current_pos is None:
            return []

        trajectory = []
        dt = 0.033  # ~30fps

        # Convert to meters
        x0 = current_pos[0] / self.px_per_meter
        y0 = current_pos[1] / self.px_per_meter

        for i in range(num_points):
            t = i * dt
            # Parabolic motion equations
            x = x0 + vx * t
            y = y0 + vy * t + 0.5 * self.gravity * t**2

            # Convert back to pixels
            px = int(x * self.px_per_meter)
            py = int(y * self.px_per_meter)

            trajectory.append((px, py))

            # Stop if out of bounds
            if py > 1500 or px > 2500 or px < 0:
                break

        return trajectory

    def calculate_shot_probability(self, trajectory, basket_pos):
        """
        Calculate shot success probability

        Args:
            trajectory: List of trajectory points
            basket_pos: Basket position (x, y)

        Returns:
            float: Probability between 0 and 1
        """
        # Return 0 if no basket detected or no trajectory
        if basket_pos is None or not trajectory:
            return 0.0

        basket_x, basket_y = basket_pos
        basket_radius = BASKET_DIAMETER_M * self.px_per_meter

        # Find trajectory points near basket height
        basket_crossing_points = []
        min_distance = float('inf')

        for i, point in enumerate(trajectory):
            height_diff = abs(point[1] - basket_y)
            if height_diff <= 10:  # Within 10 pixels of basket height
                distance = math.sqrt((point[0] - basket_x)**2 + (point[1] - basket_y)**2)
                basket_crossing_points.append((distance, i))
                if distance < min_distance:
                    min_distance = distance

        if not basket_crossing_points:
            return 0.0

        # Calculate approach angle factor
        approach_angle_factor = self._calculate_angle_factor(trajectory, basket_y)

        # Calculate base probability from distance
        base_probability = self._calculate_distance_probability(min_distance, basket_radius)

        # Apply angle factor
        final_probability = min(0.98, base_probability * approach_angle_factor)

        return final_probability

    def _calculate_angle_factor(self, trajectory, basket_y):
        """Calculate approach angle factor"""
        approach_angle_factor = 1.0

        if len(trajectory) > 5:
            basket_area_points = [p for p in trajectory if abs(p[1] - basket_y) <= 20]
            if len(basket_area_points) >= 2:
                p1, p2 = basket_area_points[0], basket_area_points[-1]
                if p2[0] != p1[0]:
                    slope = (p2[1] - p1[1]) / (p2[0] - p1[0])
                    angle_deg = math.degrees(math.atan(abs(slope)))

                    # Check if angle is optimal
                    if OPTIMAL_ANGLE_RANGE[0] <= angle_deg <= OPTIMAL_ANGLE_RANGE[1]:
                        approach_angle_factor = OPTIMAL_ANGLE_BONUS
                    elif any(bad_range[0] <= angle_deg <= bad_range[1] for bad_range in BAD_ANGLE_RANGES):
                        approach_angle_factor = BAD_ANGLE_PENALTY

        return approach_angle_factor

    def _calculate_distance_probability(self, min_distance, basket_radius):
        """Calculate base probability from distance to basket"""
        effective_radius = basket_radius * BASKET_EFFECTIVE_RADIUS_FACTOR

        for zone_name, (factor, probability) in SHOT_ZONES.items():
            if min_distance <= effective_radius * factor:
                return probability

        return SHOT_ZONES['complete_miss'][1]

    def get_ball_positions(self):
        """Get ball position history"""
        return list(self.ball_positions)

    def has_sufficient_tracking(self):
        """Check if we have enough tracking data"""
        return len(self.ball_positions) >= 5

    def get_ball_speed(self):
        """Get current ball speed"""
        vx, vy = self.get_current_velocity()
        return math.sqrt(vx**2 + vy**2)
