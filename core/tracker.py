# core/tracker.py
import numpy as np
from deep_sort_realtime.deepsort_tracker import DeepSort

class Tracker:
    def __init__(self, max_age=15, n_init=2, max_cosine_distance=0.25):
        self.deepsort = DeepSort(
            max_age=5,
            n_init=1,
            max_cosine_distance=0.3,
            embedder="mobilenet",
        )

    def update(self, detections, frame):
        """
        detections: list of dicts like {'bbox': [...], 'cls': ..., 'conf': ...}
        frame: current BGR image
        """
        ds_inputs = []

        for det in detections:
            try:
                x1, y1, x2, y2 = det['bbox']
                cls = int(float(det.get('cls', 2)))
                score = float(det.get('conf', 1.0))
                w, h = x2 - x1, y2 - y1
                ds_inputs.append(([x1, y1, w, h], score, cls))
            except Exception as e:
                print(f"[WARN] Skipping invalid detection: {det} → {e}")

        tracks_out = []
        tracks = self.deepsort.update_tracks(ds_inputs, frame=frame)

        for i, track in enumerate(tracks):
            if not track.is_confirmed():
                continue

            x1, y1, x2, y2 = track.to_ltrb()

            # Use saved det_class or fallback to original input if aligned
            cls = getattr(track, 'det_class', ds_inputs[i][2] if i < len(ds_inputs) else 2)

            tracks_out.append({
                'id': track.track_id,
                'bbox': [x1, y1, x2, y2],
                'cls': str(int(cls)),
            })

        return tracks_out
