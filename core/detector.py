from ultralytics import YOLO

class Detector:
    def __init__(self, model_path="best.pt"):
        self.model = YOLO(model_path)

    def __call__(self, frame):
        results = self.model(frame)
        detections = results[0].boxes.data.cpu().numpy()  # xyxy + conf + cls
        return [[float(x1), float(y1), float(x2), float(y2), int(cls), float(conf)]
                for x1, y1, x2, y2, conf, cls in detections]

    def detect_ball_only(self, frame, conf_thresh=0.25):
        results = self.model(frame, conf=conf_thresh)
        detections = results[0].boxes.data.cpu().numpy()

        ball_dets = []
        for x1, y1, x2, y2, conf, cls in detections:
            if int(cls) == 0:  # class 0 = ball
                ball_dets.append({
                    "bbox": [float(x1), float(y1), float(x2), float(y2)],
                    "conf": float(conf),
                    "cls": "0",
                    "id": None
                })

        return ball_dets
