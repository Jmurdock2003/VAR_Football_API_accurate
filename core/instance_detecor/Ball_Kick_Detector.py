import numpy as np

class BallKickDetector:
    def __init__(self, velocity_threshold=8.0, acceleration_threshold=5.0):
        self.velocity_threshold = velocity_threshold
        self.acceleration_threshold = acceleration_threshold
        self.last_position = None
        self.last_velocity = None
        self.last_kick_frame = -30

    def update(self, ball_position, current_frame):
        if self.last_position is None:
            self.last_position = ball_position
            self.last_velocity = [0.0, 0.0]
            return False

        dx = ball_position[0] - self.last_position[0]
        dy = ball_position[1] - self.last_position[1]
        velocity = [dx, dy]
        speed = np.linalg.norm(velocity)

        ddx = velocity[0] - self.last_velocity[0]
        ddy = velocity[1] - self.last_velocity[1]
        acceleration = np.linalg.norm([ddx, ddy])

        self.last_position = ball_position
        self.last_velocity = velocity

        if current_frame - self.last_kick_frame < 5:
            return False

        if speed > self.velocity_threshold and acceleration > self.acceleration_threshold:
            self.last_kick_frame = current_frame
            print(f"[BallKickDetector] Kick detected at frame {current_frame}")
            return True

        return False
