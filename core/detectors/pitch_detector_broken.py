import cv2
import numpy as np
from skimage.morphology import skeletonize

class PitchDetector:
    def __init__(self):
        self.frame_count = 0

    def detect(self, frame):
        self.frame_count += 1

        # STEP 1: Green mask
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        lower_green = np.array([35, 40, 40])
        upper_green = np.array([85, 255, 255])
        green_mask = cv2.inRange(hsv, lower_green, upper_green)
        contours, _ = cv2.findContours(green_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        pitch_mask = np.zeros_like(green_mask)
        if contours:
            largest = max(contours, key=cv2.contourArea)
            cv2.drawContours(pitch_mask, [largest], -1, 255, thickness=cv2.FILLED)

        # STEP 2: TopHat transform
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
        tophat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, kernel)

        # STEP 3: Apply pitch mask
        masked = cv2.bitwise_and(tophat, tophat, mask=pitch_mask)

        # STEP 4: Threshold
        _, binary = cv2.threshold(masked, 30, 255, cv2.THRESH_BINARY)

        # STEP 5: Skeletonize the binary image
        binary_normalized = (binary // 255).astype(np.uint8)
        skeleton = skeletonize(binary_normalized).astype(np.uint8) * 255

        # STEP 6: Hough transform on skeleton
        lines = cv2.HoughLinesP(skeleton, 1, np.pi / 180, threshold=80, minLineLength=60, maxLineGap=10)
        fitted_lines = []
        if lines is not None:
            for x1, y1, x2, y2 in lines[:, 0]:
                fitted_lines.append((x1, y1, x2, y2))

        # ==== DEBUG VIEWS ====
        cv2.imshow("Pitch Mask", pitch_mask)
        cv2.imshow("TopHat", tophat)
        cv2.imshow("Binary", binary)

        # Draw lines directly on skeleton
        skeleton_colored = cv2.cvtColor(skeleton, cv2.COLOR_GRAY2BGR)
        for x1, y1, x2, y2 in fitted_lines:
            cv2.line(skeleton_colored, (x1, y1), (x2, y2), (0, 0, 255), 1)
        cv2.imshow("Skeleton with Lines", skeleton_colored)

        debug = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        debug = cv2.cvtColor(debug, cv2.COLOR_GRAY2BGR)
        for x1, y1, x2, y2 in fitted_lines:
            cv2.line(debug, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.imshow("Detected Pitch Lines (Skeleton Hough)", debug)
        cv2.waitKey(1)

        return fitted_lines