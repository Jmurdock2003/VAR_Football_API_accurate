import numpy as np  # Numerical operations for distance calculation

class BallKickDetector:
    """
    Detects when the ball is kicked by monitoring exit from player bounding box.
    It triggers once the ball leaves the proximity of a player after contact.
    """
    def __init__(self, distance_threshold=5.0):
        # Maximum pixel distance between ball and player box edges to consider "touching"
        self.distance_threshold = float(distance_threshold)
        # Flag indicating we are waiting for the ball to exit player bbox after touching
        self.awaiting_exit = False

    def update(self, ball, player, current_frame):
        """
        Determine if a kick event occurred based on ball and player bounding boxes.
        Returns True if the ball was inside the player's box and then moved out.
        """
        # Validate inputs
        if not isinstance(ball, dict) or not isinstance(player, dict):
            return False
        if 'bbox' not in ball or 'bbox' not in player:
            return False

        # Unpack bounding box coordinates
        px1, py1, px2, py2 = player['bbox']  # player box
        bx1, by1, bx2, by2 = ball['bbox']    # ball box

        # Compute horizontal and vertical edge-to-edge distances
        horizontal_dist = max(0, max(px1 - bx2, bx1 - px2))
        vertical_dist   = max(0, max(py1 - by2, by1 - py2))
        # Euclidean distance between nearest edges
        dist = np.hypot(horizontal_dist, vertical_dist)

        # If ball is within threshold (i.e. "touching" the player), arm exit trigger
        if dist <= self.distance_threshold:
            self.awaiting_exit = True
            return False

        # If previously touching and now outside threshold → kick detected
        if self.awaiting_exit and dist > self.distance_threshold:
            self.awaiting_exit = False
            print(f"[KICK DETECTED] Frame {current_frame} – Ball exited player {player.get('id')} bbox")
            return True

        # No kick detected
        return False
