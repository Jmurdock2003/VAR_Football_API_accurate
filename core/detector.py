# core/detector.py
from ultralytics import YOLO

class Detector:
    def __init__(self, model_path: str = "best.pt"):
        print(">>> Using ultralytics YOLO from core/detector.py")  # debug
        self.model = YOLO(model_path)
        try:
            self.model.fuse()
        except Exception:
            pass
        self.model.conf = 0.35  # better filtering of weak overlapping boxes

    def __call__(self, frame):
        results = self.model(frame)[0]
        detections = []
        for x1, y1, x2, y2, conf, cls in results.boxes.data.tolist():
            detections.append([
                int(x1), int(y1), int(x2), int(y2),
                int(cls), float(conf)
            ])
        return detections
