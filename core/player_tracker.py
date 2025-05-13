import numpy as np
import supervision as sv
from utils.bbox_utils import get_centre

class PlayerTracker:
    def __init__(self):
        self.tracker = sv.ByteTrack()
        self.previous_positions = {}  # track_id -> list of past centres

    def update(self, detections, frame):
        # Filter player-related detections (exclude ball)
        xyxy = []
        class_ids = []
        confidences = []

        for det in detections:
            if det.get("cls") != "0":  # only player or referee
                x1, y1, x2, y2 = det["bbox"]
                xyxy.append([x1, y1, x2, y2])
                class_ids.append(int(det["cls"]))
                confidences.append(float(det["conf"]))

        xyxy = np.array(xyxy, dtype=np.float32)
        class_ids = np.array(class_ids, dtype=int)
        confidences = np.array(confidences, dtype=np.float32)

        sv_detections = sv.Detections(
            xyxy=xyxy,
            class_id=class_ids,
            confidence=confidences
        )

        results = self.tracker.update_with_detections(sv_detections)

        tracks_out = []
        for r in results:
            x1, y1, x2, y2 = r[0]
            cls = r[3]
            tid = r[4]

            cx, cy = get_centre([x1, y1, x2, y2])
            if tid not in self.previous_positions:
                self.previous_positions[tid] = []
            self.previous_positions[tid].append((cx, cy))
            if len(self.previous_positions[tid]) > 5:
                self.previous_positions[tid] = self.previous_positions[tid][-5:]

            velocity = self._estimate_velocity(self.previous_positions[tid])

            tracks_out.append({
                'id': int(tid),
                'bbox': [float(x1), float(y1), float(x2), float(y2)],
                'cls': str(int(cls)),
                'velocity': [float(v) for v in velocity]
            })

        return tracks_out

    def _estimate_velocity(self, history):
        if len(history) < 2:
            return [0.0, 0.0]
        dx = history[-1][0] - history[-2][0]
        dy = history[-1][1] - history[-2][1]
        return [dx, dy]
