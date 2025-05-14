import numpy as np  # array operations for numerical data
import supervision as sv  # supervision library for ByteTrack
from utils.bbox_utils import get_centre  # helper to compute bounding box center

class PlayerTracker:
    """
    Tracks player objects across video frames using ByteTrack,
    maintains history for velocity estimation, and ensures class stability.
    """
    def __init__(self):
        # ByteTrack tracker instance
        self.tracker = sv.ByteTrack()
        # Dictionary mapping track ID -> list of past center coordinates
        self.previous_positions = {}
        # Dictionary mapping track ID -> last known object class
        self.class_history = {}
        # Dictionary mapping track ID -> previous team assignment (optional)
        self.prev_assignments = {}

    def update(self, detections, frame):
        """
        Update tracks based on new detections in the current frame.
        Filters out non-player classes, runs tracker, stabilizes class labels,
        estimates velocity, and returns formatted track output.
        """
        # Extract bounding boxes, class IDs, and confidences for players/referees
        xyxy, class_ids, confidences = [], [], []
        for det in detections:
            # Skip the ball (class '0'), only track players/referees
            if det.get("cls") != "0":
                x1, y1, x2, y2 = det["bbox"]
                xyxy.append([x1, y1, x2, y2])
                class_ids.append(int(det["cls"]))
                confidences.append(float(det["conf"]))

        # If no players detected, return empty list
        if not xyxy:
            return []

        # Convert detection lists to NumPy arrays for supervision
        xyxy = np.array(xyxy, dtype=np.float32)
        class_ids = np.array(class_ids, dtype=int)
        confidences = np.array(confidences, dtype=np.float32)

        # Create a Detections object for ByteTrack
        sv_detections = sv.Detections(
            xyxy=xyxy,
            class_id=class_ids,
            confidence=confidences
        )

        # Run the tracker update with current detections
        results = self.tracker.update_with_detections(sv_detections)

        tracks_out = []  # final output list
        seen_ids = set()  # to avoid duplicate IDs in one frame

        for r in results:
            x1, y1, x2, y2 = r[0]       # bounding box coordinates
            cls = int(r[3])             # class ID from tracker
            tid = int(r[4])             # unique track ID

            # Ignore duplicates from tracker output
            if tid in seen_ids:
                continue
            seen_ids.add(tid)

            # If class has changed, retain previous class to avoid jitter
            if tid in self.class_history:
                if self.class_history[tid] != cls:
                    print(f"[SWITCH] Track {tid} class changed from {self.class_history[tid]} to {cls}, reverting.")
                    cls = self.class_history[tid]
                    # Remove previous team assignment to force reassign if needed
                    self.prev_assignments.pop(tid, None)
            # Store class if first time seeing this track
            self.class_history[tid] = cls

            # Compute center of the bounding box for velocity estimation
            cx, cy = get_centre([x1, y1, x2, y2])
            # Initialize history if needed, then append new center
            if tid not in self.previous_positions:
                self.previous_positions[tid] = []
            self.previous_positions[tid].append((cx, cy))
            # Keep only the last 5 positions to limit memory
            if len(self.previous_positions[tid]) > 5:
                self.previous_positions[tid] = self.previous_positions[tid][-5:]

            # Estimate velocity vector based on position history
            velocity = self._estimate_velocity(self.previous_positions[tid])

            # Format track dictionary for output
            tracks_out.append({
                'id': tid,
                'bbox': [float(x1), float(y1), float(x2), float(y2)],
                'cls': str(cls),
                'velocity': [float(v) for v in velocity]
            })

        return tracks_out

    def _estimate_velocity(self, history):
        """
        Estimate simple frame-to-frame velocity from history of centers.
        Returns zero velocity if insufficient data.
        """
        if len(history) < 2:
            return [0.0, 0.0]
        # Difference between the last two center positions
        dx = history[-1][0] - history[-2][0]
        dy = history[-1][1] - history[-2][1]
        return [dx, dy]
