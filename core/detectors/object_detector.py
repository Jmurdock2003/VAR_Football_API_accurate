from ultralytics import YOLO  # Ultralytics YOLO model for object detection

class Detector:
    """
    Wrapper around a YOLO model for object detection on video frames.
    Provides general detection and a specialized method for ball-only detection.
    """
    def __init__(self, model_path="models/best.pt"):
        # Load the YOLO model from the given path
        try:
            self.model = YOLO(model_path)
        except Exception as e:
            raise RuntimeError(f"Failed to load YOLO model from {model_path}: {e}")

    def __call__(self, frame):
        """
        Run the YOLO model on a frame and return detections as a list of
        [x1, y1, x2, y2, class_id, confidence].
        """
        try:
            results = self.model(frame)
        except Exception as e:
            # If model inference fails, log and return empty list
            print(f"Detection inference error: {e}")
            return []

        # Extract bounding boxes, confidences, and class IDs from the results
        detections = results[0].boxes.data.cpu().numpy()
        output = []
        for x1, y1, x2, y2, conf, cls in detections:
            # Convert values to native Python types
            output.append([
                float(x1),
                float(y1),
                float(x2),
                float(y2),
                int(cls),
                float(conf)
            ])
        return output

    def detect_ball_only(self, frame, conf_thresh=0.25):
        """
        Run YOLO on the frame with a confidence threshold and return only
        ball detections (class 0) formatted as dicts with bbox, conf, cls, id.
        """
        try:
            # Perform detection limiting by confidence
            results = self.model(frame, conf=conf_thresh)
        except Exception as e:
            print(f"Ball-only detection error: {e}")
            return []

        detections = results[0].boxes.data.cpu().numpy()
        ball_detections = []
        for x1, y1, x2, y2, conf, cls in detections:
            if int(cls) == 0:
                # Build a standardized dict for the ball
                ball_detections.append({
                    "bbox": [float(x1), float(y1), float(x2), float(y2)],
                    "conf": float(conf),
                    "cls": "0",
                    "id": None
                })
        return ball_detections