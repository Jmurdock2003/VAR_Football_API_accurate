import cv2  # OpenCV for image processing and optical flow
import numpy as np  # Numerical operations for arrays

class CameraMotionEstimator:
    """
    Estimates camera motion (dx, dy) between frames using sparse optical flow.
    Looks for Shi-Tomasi corners in masked regions and uses Lucas-Kanade flow.
    """
    def __init__(self, min_distance: float = 5.0):
        # Minimum pixel movement threshold to trigger feature refresh
        self.min_distance = float(min_distance)

        # Parameters for Lucas-Kanade optical flow
        self.lk_params = {
            'winSize': (15, 15),
            'maxLevel': 2,
            'criteria': (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
        }

        # Parameters for Shi-Tomasi corner detection; mask set in initialize()
        self.feature_params = {
            'maxCorners': 100,
            'qualityLevel': 0.3,
            'minDistance': 3,
            'blockSize': 7,
            'mask': None  # Will be replaced by a binary mask focusing on frame edges
        }

        # Storage for previous frame (grayscale) and feature points
        self.prev_gray = None
        self.prev_features = None

    def initialize(self, frame):
        """
        Prepare for flow estimation by:
          - Converting the frame to grayscale
          - Creating a binary mask focusing on the leftmost and rightmost 20px
          - Detecting initial feature points within the mask
        """
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape

        # Create mask: only non-zero mask regions are considered for corner detection
        mask = np.zeros_like(gray)
        # BROKEN: Focusing only 20px at edges may yield too few features, causing unstable motion estimates
        mask[:, :20] = 1
        mask[:, w-20:] = 1
        self.feature_params['mask'] = mask

        # Store this frame and detect corners
        self.prev_gray = gray
        # BROKEN: If no features found here, prev_features will be None, causing later errors
        self.prev_features = cv2.goodFeaturesToTrack(gray, **self.feature_params)

    def estimate(self, frame) -> tuple[float, float]:
        """
        Calculate median camera motion (dx, dy) between the stored previous frame
        and the current frame. Refreshes features if movement exceeds threshold.
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # On first call (or after reset), initialize features
        if self.prev_gray is None or self.prev_features is None:
            self.initialize(frame)
            return 0.0, 0.0

        # Calculate optical flow for previous features
        new_features, status, _ = cv2.calcOpticalFlowPyrLK(
            self.prev_gray, gray, self.prev_features, None, **self.lk_params
        )

        # Validate flow results
        if new_features is None or status is None:
            # BROKEN: Flow computation failed, re-initialize
            self.initialize(frame)
            return 0.0, 0.0

        # Select only successfully tracked points
        good_new = new_features[status.flatten() == 1]
        good_old = self.prev_features[status.flatten() == 1]
        if len(good_new) == 0:
            # BROKEN: No good points left, re-initialize
            self.initialize(frame)
            return 0.0, 0.0

        # Compute displacement vectors and take median for robustness
        displacements = good_new - good_old
        dx, dy = np.median(displacements, axis=0)

        # If movement is small, do not refresh feature points
        if np.hypot(dx, dy) < self.min_distance:
            # Update previous frame only
            self.prev_gray = gray
            return float(dx), float(dy)

        # Significant movement: refresh features for next iteration
        self.prev_features = cv2.goodFeaturesToTrack(gray, **self.feature_params)
        self.prev_gray = gray
        return float(dx), float(dy)

    # BROKEN: I cant understand what is wrong with this code, when i add it into the system it causes tracks to inccorrectly