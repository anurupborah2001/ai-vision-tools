from .base import AIVisionComponent
import cv2

class MotionDetector(AIVisionComponent):
    """Detects motion by comparing the current frame to the previous frame."""
    
    def __init__(self):
        super().__init__()
        self.prev_gray = None

    def _execute(self, data, config):
        frame = data["frame"] if isinstance(data, dict) else data
        output = frame.copy()
        
        min_area = config.get('min_area', 800)
        draw = config.get('draw_motion', True)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (21, 21), 0)
        motion_boxes = []

        if self.prev_gray is not None:
            diff = cv2.absdiff(self.prev_gray, gray)
            thresh = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)[1]
            thresh = cv2.dilate(thresh, None, iterations=2)

            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            for c in contours:
                if cv2.contourArea(c) < min_area:
                    continue

                x, y, w, h = cv2.boundingRect(c)
                motion_boxes.append((x, y, w, h))

                if draw:
                    cv2.rectangle(output, (x, y), (x + w, y + h), (0, 255, 255), 2)

        self.prev_gray = gray
        
        return {"frame": output, "motion_boxes": motion_boxes}