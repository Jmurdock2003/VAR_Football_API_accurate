import numpy as np
import cv2
from collections import deque
from utils.bbox_utils import get_centre

class BallTracker:
    def __init__(self, max_history=5):
        self.last_ball = None
        self.ball_history = deque(maxlen=max_history)
        self.last_velocity = [0.0, 0.0]
        self.ball_id = 1
        self.prev_gray = None
        self.prev_bbox = None

    def update(self, frame, detections):
        if frame is None or not isinstance(frame, np.ndarray):
            return []

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        best_ball = self._select_best_ball(detections)

        if best_ball:
            self.prev_bbox = best_ball['bbox']
            self.prev_gray = gray.copy()
            cx, cy = get_centre(best_ball['bbox'])
            self.ball_history.append((cx, cy))
            velocity = self._estimate_velocity(self.ball_history)
            self.last_ball = {
                'id': int(self.ball_id),
                'bbox': [float(x) for x in best_ball['bbox']],
                'cls': '0',
                'velocity': [float(v) for v in velocity]
            }
        elif self.last_ball and self.prev_gray is not None and self.prev_bbox is not None:
            prev_points = np.array([get_centre(self.prev_bbox)], dtype=np.float32).reshape(-1, 1, 2)
            next_points, status, _ = cv2.calcOpticalFlowPyrLK(self.prev_gray, gray, prev_points, None)
            if status[0][0] == 1:
                new_cx, new_cy = next_points[0][0]
                w = self.prev_bbox[2] - self.prev_bbox[0]
                h = self.prev_bbox[3] - self.prev_bbox[1]
                new_bbox = [new_cx - w / 2, new_cy - h / 2, new_cx + w / 2, new_cy + h / 2]
                self.prev_bbox = new_bbox
                self.prev_gray = gray.copy()
                self.ball_history.append((new_cx, new_cy))
                velocity = self._estimate_velocity(self.ball_history)
                self.last_ball = {
                    'id': int(self.ball_id),
                    'bbox': [float(x) for x in new_bbox],
                    'cls': '0',
                    'velocity': [float(v) for v in velocity]
                }

        return [self.last_ball] if self.last_ball else []

    def _select_best_ball(self, detections):
        balls = [t for t in detections if t['cls'] == '0']
        if not balls:
            return None

        best_ball = None
        best_score = float('inf')
        distance_weight = 1.0
        confidence_weight = 100.0
        max_reasonable_dist = 100  # pixels
        score_threshold = 150      # reject if worse than this

        for b in balls:
            conf = float(b.get('conf', 0))
            centre = get_centre(b['bbox'])

            if self.last_ball:
                last_centre = get_centre(self.last_ball['bbox'])
                dist = np.linalg.norm(np.array(centre) - np.array(last_centre))
            else:
                dist = 0

            score = dist * distance_weight + (1 - conf) * confidence_weight

            # Prefer fast switching when confident and far
            if self.last_ball and dist > max_reasonable_dist and conf > 0.6:
                score -= 50

            if score < best_score:
                best_score = score
                best_ball = b

        # Reject if even best match is too suspicious
        if best_score > score_threshold:
            return None

        return best_ball


    def _estimate_velocity(self, history):
        if len(history) < 2:
            return [0.0, 0.0]
        dx = history[-1][0] - history[-2][0]
        dy = history[-1][1] - history[-2][1]
        return [dx, dy]
