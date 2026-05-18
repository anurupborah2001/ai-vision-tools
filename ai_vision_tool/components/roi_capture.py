from .picture_taker import PictureTaker
import cv2  

class ROICapture(PictureTaker):
    def __init__(self, roi=(100, 100, 400, 400), draw_roi=True, **kwargs):
        super().__init__(**kwargs)
        self.name = "roi_capture"
        self.roi = roi
        self.draw_roi = draw_roi

    def extract_roi(self, frame):
        x, y, w, h = self.roi
        return frame[y:y + h, x:x + w]

    def process(self, frame, **kwargs):
        output = frame.copy()

        if self.draw_roi:
            x, y, w, h = self.roi
            cv2.rectangle(output, (x, y), (x + w, y + h), (0, 255, 0), 2)

        if kwargs.get("capture_roi", False):
            roi_frame = self.extract_roi(frame)
            self.save_frame(roi_frame)

        return output
      
