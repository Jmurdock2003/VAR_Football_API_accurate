
import cv2
import numpy as np

class CameraMotionEstimator:
    def __init__(self, min_distance: float = 5.0):
        self.min_distance = min_distance
        # Lucas-Kanade optical flow params
        self.lk_params = dict(
            winSize  = (15, 15),
            maxLevel = 2,
            criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
        )
        # Shi-Tomasi corner detection params
        self.feature_params = dict(
            maxCorners    = 100,
            qualityLevel  = 0.3,
            minDistance   = 3,
            blockSize     = 7,
            mask           = None  # set later based on frame dimensions
        )
        self.prev_gray     = None
        self.prev_features = None

    def initialize(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape
        mask = np.zeros_like(gray)
        # focus on edges (left 20px and right 20px)
        mask[:, :20] = 1
        mask[:, w-20:] = 1
        self.feature_params['mask'] = mask
        self.prev_gray = gray
        self.prev_features = cv2.goodFeaturesToTrack(gray, **self.feature_params)

    def estimate(self, frame) -> tuple[float, float]:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if self.prev_gray is None:
            self.initialize(frame)
            return 0.0, 0.0

        new_features, st, err = cv2.calcOpticalFlowPyrLK(
            self.prev_gray, gray, self.prev_features, None, **self.lk_params
        )

        if new_features is None or self.prev_features is None or st is None:
            self.initialize(frame)
            return 0.0, 0.0

        good_new = new_features[st == 1]
        good_old = self.prev_features[st == 1]

        if len(good_new) == 0:
            self.initialize(frame)
            return 0.0, 0.0

        displacements = good_new - good_old
        dx, dy = np.median(displacements, axis=0)  # smoother than max

        # Refresh features if needed
        if np.linalg.norm([dx, dy]) > self.min_distance:
            self.prev_features = cv2.goodFeaturesToTrack(gray, **self.feature_params)
        self.prev_gray = gray
        return float(dx), float(dy)
