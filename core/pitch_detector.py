import cv2
import numpy as np

class PitchDetector:
    def __init__(self):
        self.last_lines = {}
        self.cache_interval = 30
        self.frame_count = 0
        self.previous_right = None

    def detect(self, frame):
        self.frame_count += 1

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blur, 50, 150, apertureSize=3)

        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=80, minLineLength=60, maxLineGap=10)
        horizontal_lines = []
        vertical_lines = []

        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
                if abs(angle) < 30:
                    horizontal_lines.append((x1, y1, x2, y2))
                elif abs(angle) > 60:
                    vertical_lines.append((x1, y1, x2, y2))

        h, w = frame.shape[:2]

        def midpoint(p): return (p[0] + p[2]) / 2
        def line_height(p): return abs(p[1] - p[3])

        top, bottom = None, None
        if horizontal_lines:
            y_mids = [((y1 + y2) / 2) for x1, y1, x2, y2 in horizontal_lines]
            top = int(min(y_mids))
            bottom = int(max(y_mids))

        valid_verticals = []
        if vertical_lines and top is not None and bottom is not None:
            pitch_height = bottom - top
            for x1, y1, x2, y2 in vertical_lines:
                height = line_height((x1, y1, x2, y2))
                if abs(height - pitch_height) < 0.15 * pitch_height:  # within 15% of full pitch height
                    valid_verticals.append((x1, y1, x2, y2))

        x_mids = [int(midpoint(l)) for l in valid_verticals]
        left, halfway, right = None, None, None

        if x_mids:
            if self.previous_right is None:
                left = min(x_mids)
                halfway = max(x_mids)
                self.previous_right = halfway
            else:
                current_max = max(x_mids)
                if current_max > self.previous_right:
                    left = self.previous_right
                    right = current_max
                    halfway = None
                    self.previous_right = current_max
                else:
                    left = min(x_mids)
                    halfway = max(x_mids)

        pitch_boundaries = {
            "left": left,
            "right": right,
            "top": top,
            "bottom": bottom,
            "halfway": halfway
        }

        self.last_lines = pitch_boundaries
        return pitch_boundaries
