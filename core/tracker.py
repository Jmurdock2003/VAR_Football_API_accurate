import numpy as np
from deep_sort_realtime.deepsort_tracker import DeepSort

class Tracker:
    def __init__(self, max_age=15, n_init=2, max_cosine_distance=0.25):
        self.deepsort = DeepSort(
            max_age=max_age,
            n_init=n_init,
            max_cosine_distance=max_cosine_distance,
            embedder="mobilenet",
        )
        self.previous_positions = {}  # track_id -> list of past centres

    def update(self, detections, frame):
        ds_inputs = []

        for det in detections:
            try:
                x1, y1, x2, y2 = det['bbox']
                cls = int(float(det.get('cls', 2)))
                conf = float(det.get('conf', 1.0))
                w, h = x2 - x1, y2 - y1
                ds_inputs.append(([x1, y1, w, h], conf, cls))
            except Exception as e:
                print(f"[WARN] Skipping invalid detection: {det} → {e}")

        tracks_out = []
        tracks = self.deepsort.update_tracks(ds_inputs, frame=frame)

        for i, track in enumerate(tracks):
            if not track.is_confirmed():
                continue

            x1, y1, x2, y2 = track.to_ltrb()
            cls = getattr(track, 'det_class', ds_inputs[i][2] if i < len(ds_inputs) else 2)
            tid = track.track_id

            # Estimate trajectory (centre history)
            cx, cy = self._get_center([x1, y1, x2, y2])
            if tid not in self.previous_positions:
                self.previous_positions[tid] = []
            self.previous_positions[tid].append((cx, cy))

            # Keep last 5 positions
            if len(self.previous_positions[tid]) > 5:
                self.previous_positions[tid] = self.previous_positions[tid][-5:]

            velocity = self._estimate_velocity(self.previous_positions[tid])

            tracks_out.append({
                'id': tid,
                'bbox': [x1, y1, x2, y2],
                'cls': str(int(cls)),
                'velocity': velocity,
            })

        return tracks_out

    def _get_center(self, bbox):
        x1, y1, x2, y2 = bbox
        return ((x1 + x2) / 2, (y1 + y2) / 2)

    def _estimate_velocity(self, history):
        if len(history) < 2:
            return [0.0, 0.0]
        dx = history[-1][0] - history[-2][0]
        dy = history[-1][1] - history[-2][1]
        return [dx, dy]
