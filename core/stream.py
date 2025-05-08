import cv2
import numpy as np
from .detector import Detector
from .tracker import Tracker
from .team_assign import TeamAssigner

class LiveProcessor:
    def __init__(self, source=0, detect_every: int = 1):
        self.cap            = cv2.VideoCapture(source)
        self.detector       = Detector()
        self.tracker        = Tracker(max_age=15, n_init=2, max_cosine_distance=0.25)
        self.team_assigner  = TeamAssigner()
        self.detect_every   = detect_every
        self.frame_count    = 0
        self.last_dets      = []
        self.last_ball      = None  # Track ball for continuity

    def __iter__(self):
        while self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                break
            yield self.process(frame)

    def process(self, frame):
        self.frame_count += 1

        # 1) Detection
        if self.frame_count % self.detect_every == 0:
            raw_dets = self.detector(frame)
            formatted_dets = []
            for det in raw_dets:
                x1, y1, x2, y2, cls, conf = det
                formatted_dets.append({
                    "bbox": [x1, y1, x2, y2],
                    "conf": conf,
                    "cls": str(int(cls)),
                    "id": None
                })
            self.last_dets = formatted_dets

        dets = self.last_dets

        # 2) Tracking
        tracks = self.tracker.update(dets, frame)

        # 3) Ensure format
        for track in tracks:
            track['cls'] = str(track.get('cls', '2'))

        # 4) Team assignment
        team_tracks = self.team_assigner.assign(frame, tracks)

        # 5) Keep only one ball
        team_tracks, self.last_ball = self._suppress_duplicate_balls(team_tracks, self.last_ball)

        # 6) Output payload
        fid = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))
        return {"frame_id": fid, "tracks": team_tracks}

    def _suppress_duplicate_balls(self, tracks, prev_ball):
        balls = [t for t in tracks if t['cls'] == '0']
        if not balls:
            return tracks, None

        # Prefer highest confidence
        balls.sort(key=lambda b: float(b.get('conf', 0)), reverse=True)

        if prev_ball:
            px, py = self._get_center(prev_ball['bbox'])
            balls.sort(key=lambda b: np.linalg.norm(
                np.array(self._get_center(b['bbox'])) - np.array([px, py])
            ))

        best_ball = balls[0]
        filtered = [t for t in tracks if t['cls'] != '0'] + [best_ball]
        return filtered, best_ball

    def _get_center(self, bbox):
        x1, y1, x2, y2 = bbox
        return [(x1 + x2) / 2, (y1 + y2) / 2]
