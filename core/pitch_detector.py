import cv2
import numpy as np

class PitchDetector:
    def __init__(self):
        self.last_lines = {}
        self.previous_right = None
        self.frame_count = 0

    def detect(self, frame):
        self.frame_count += 1
        h, w = frame.shape[:2]

        # STEP 1: Isolate green pitch
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        lower_green = np.array([35, 40, 40])
        upper_green = np.array([85, 255, 255])
        green_mask = cv2.inRange(hsv, lower_green, upper_green)

        # Find largest green area as pitch
        contours, _ = cv2.findContours(green_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        pitch_mask = np.zeros_like(green_mask)

        if contours:
            largest = max(contours, key=cv2.contourArea)
            cv2.drawContours(pitch_mask, [largest], -1, 255, thickness=cv2.FILLED)
            margin = 10
            pitch_mask[:margin, :] = 0
            pitch_mask[-margin:, :] = 0 
            pitch_mask[:, :margin] = 0
            pitch_mask[:, -margin:] = 0
            blurred_mask = cv2.GaussianBlur(pitch_mask, (5, 5), 0)
            kernel = np.ones((3, 3), np.uint8)
            eroded_mask = cv2.erode(blurred_mask, kernel, iterations=1)

        # STEP 1.5: Improve visibility before white line detection
        # Apply CLAHE to luminance to boost contrast (works well for stadium footage)
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(4,4))
        cl = clahe.apply(l)
        enhanced_lab = cv2.merge((cl, a, b))
        contrast_frame = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)

        # Convert enhanced image to HSV for white mask
        hsv_contrast = cv2.cvtColor(contrast_frame, cv2.COLOR_BGR2HSV)

        # STEP 2: Detect white lines inside the pitch
        lower_white = np.array([0, 0, 140])
        upper_white = np.array([180, 80, 255])
        white_mask = cv2.inRange(hsv_contrast, lower_white, upper_white)

        # Keep only white inside the detected pitch region
        focused_white = cv2.bitwise_and(white_mask, white_mask, mask=eroded_mask)


        # STEP 3: Morphological ops
        kernel = np.ones((3, 3), np.uint8)
        enhanced = cv2.dilate(focused_white, kernel, iterations=2)
        enhanced = cv2.erode(enhanced, kernel, iterations=1)

        # STEP 4: Edge + Hough
        edges = cv2.Canny(enhanced, 50, 150)
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 100, minLineLength=100, maxLineGap=20)

        horizontal_lines, vertical_lines = [], []

        if lines is not None:
            for x1, y1, x2, y2 in lines[:, 0]:
                angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
                length = np.hypot(x2 - x1, y2 - y1)
                if length < 100:
                    continue
                if abs(angle) < 30:
                    horizontal_lines.append((x1, y1, x2, y2))
                elif abs(angle) > 60:
                    vertical_lines.append((x1, y1, x2, y2))

        def midpoint(p): return (p[0] + p[2]) / 2
        def line_height(p): return abs(p[1] - p[3])

        # Horizontal line logic
        top, bottom = None, None
        if horizontal_lines:
            y_mids = [((y1 + y2) / 2) for x1, y1, x2, y2 in horizontal_lines]
            top = int(min(y_mids))
            bottom = int(max(y_mids))

        # Vertical line logic
        valid_verticals = []
        if vertical_lines and top is not None and bottom is not None:
            pitch_height = bottom - top
            for x1, y1, x2, y2 in vertical_lines:
                height = line_height((x1, y1, x2, y2))
                if abs(height - pitch_height) < 0.2 * pitch_height:
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

        # ==== DEBUG VIEWS ====
        cv2.imshow("Pitch Mask", pitch_mask)
        cv2.imshow("White Line Mask", focused_white)
        debug = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)
        if lines is not None:
            for x1, y1, x2, y2 in lines[:, 0]:
                cv2.line(debug, (x1, y1), (x2, y2), (0, 255, 255), 2)
        cv2.imshow("Final Line Detection", debug)
        cv2.waitKey(1)

        return pitch_boundaries
