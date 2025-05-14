import numpy as np  # numerical operations on arrays
import cv2  # OpenCV for image processing and optical flow
from collections import deque  # efficient queue for fixed-length history
from utils.bbox_utils import get_centre  # helper to compute bounding box center

class BallTracker:
    """
    Tracks the ball across frames using detection or optical flow.
    Maintains a short history of positions to estimate velocity.
    """
    def __init__(self, max_history=5):
        # Last known ball state (dict with id, bbox, cls, velocity)
        self.last_ball = None
        # Deque to hold recent centres for velocity estimation
        self.ball_history = deque(maxlen=max_history)
        # Last estimated velocity vector [dx, dy]
        self.last_velocity = [0.0, 0.0]
        # Unique identifier for the ball track
        self.ball_id = 1
        # Previous grayscale frame for optical flow
        self.prev_gray = None
        # Previous bounding box for optical flow lane
        self.prev_bbox = None

    def update(self, frame, detections):
        """
        Update ball position and velocity for the current frame.
        If a detection is available, use it; otherwise, fallback to LK optical flow.
        Returns a list containing the current ball track or empty if not found.
        """
        # Validate frame input
        if frame is None or not isinstance(frame, np.ndarray):
            return []

        # Convert current frame to grayscale once
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Choose the best ball detection among provided detections
        best_ball = self._select_best_ball(detections)

        if best_ball:
            # If a reliable detection exists, reset optical flow reference
            self.prev_bbox = best_ball['bbox']
            self.prev_gray = gray.copy()

            # Compute centre of detected bounding box
            cx, cy = get_centre(best_ball['bbox'])
            # Append new centre to history
            self.ball_history.append((cx, cy))
            # Estimate velocity from history
            velocity = self._estimate_velocity(self.ball_history)

            # Update last_ball with detection results
            self.last_ball = {
                'id': self.ball_id,
                'bbox': [float(x) for x in best_ball['bbox']],
                'cls': '0',
                'velocity': [float(v) for v in velocity]
            }

        elif self.last_ball and self.prev_gray is not None and self.prev_bbox is not None:
            # If no detection, attempt optical flow from last known position
            prev_point = np.array([get_centre(self.prev_bbox)], dtype=np.float32).reshape(-1,1,2)
            next_point, status, _ = cv2.calcOpticalFlowPyrLK(
                self.prev_gray, gray, prev_point, None
            )
            # Check if flow found a valid new point
            if status[0][0] == 1:
                new_cx, new_cy = next_point[0][0]
                # Keep bounding box size constant
                w = self.prev_bbox[2] - self.prev_bbox[0]
                h = self.prev_bbox[3] - self.prev_bbox[1]
                new_bbox = [new_cx - w/2, new_cy - h/2, new_cx + w/2, new_cy + h/2]

                # Update previous references for next frame
                self.prev_bbox = new_bbox
                self.prev_gray = gray.copy()

                # Append new optical-flow centre to history
                self.ball_history.append((new_cx, new_cy))
                velocity = self._estimate_velocity(self.ball_history)

                # Update last_ball with optical flow results
                self.last_ball = {
                    'id': self.ball_id,
                    'bbox': [float(x) for x in new_bbox],
                    'cls': '0',
                    'velocity': [float(v) for v in velocity]
                }

        # Return the most recent ball track, if one exists
        return [self.last_ball] if self.last_ball else []

    def _select_best_ball(self, detections):
        """
        From multiple ball detections (cls '0'), choose the most likely.
        Combines distance from last known position and detection confidence.
        Returns the selected detection dict or None if no valid match.
        """
        # Filter detections to only those labeled as ball
        balls = [d for d in detections if d.get('cls') == '0']
        if not balls:
            return None

        best_ball = None
        best_score = float('inf')

        # Weights to balance spatial and confidence terms
        dist_weight = 1.0
        conf_weight = 100.0
        max_dist = 100      # pixels considered reasonable
        reject_threshold = 150

        for det in balls:
            conf = float(det.get('conf', 0))
            cx, cy = get_centre(det['bbox'])

            # Compute distance from last ball location if available
            if self.last_ball:
                lx, ly = get_centre(self.last_ball['bbox'])
                dist = np.linalg.norm([cx - lx, cy - ly])
            else:
                dist = 0

            # Combine into a scoring function
            score = dist * dist_weight + (1 - conf) * conf_weight
            # Allow confident large jumps to override
            if self.last_ball and dist > max_dist and conf > 0.6:
                score -= 50

            # Keep the detection with lowest score
            if score < best_score:
                best_score = score
                best_ball = det

        # Reject match if score too large
        if best_score > reject_threshold:
            return None

        return best_ball

    def _estimate_velocity(self, history):
        """
        Simple velocity: difference between last two centre points.
        Returns [dx, dy]; zero if insufficient history.
        """
        if len(history) < 2:
            return [0.0, 0.0]
        x_prev, y_prev = history[-2]
        x_curr, y_curr = history[-1]
        return [x_curr - x_prev, y_curr - y_prev]
