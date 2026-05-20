from .picture_taker import PictureTaker
import cv2


class ROICapture(PictureTaker):
    """Captures still images cropped to a fixed region of interest (ROI).

    Extends PictureTaker to overlay and optionally crop a rectangular ROI
    before saving. The ROI rectangle is drawn on the preview frame.

    Args:
        roi (tuple[int, int, int, int]): Region of interest as (x, y, w, h).
            Default is (100, 100, 400, 400).
        draw_roi (bool): If True, draws the ROI rectangle on the preview.
            Default is True.
    """

    def __init__(self, roi=(100, 100, 400, 400), draw_roi=True, **kwargs):
        """Initializes ROICapture with ROI bounds and display settings.

        Args:
            roi (tuple[int, int, int, int]): Bounding box (x, y, w, h) for the ROI.
                Default is (100, 100, 400, 400).
            draw_roi (bool): Whether to overlay the ROI rectangle on preview.
                Default is True.
            **kwargs: Forwarded to PictureTaker.
        """
        super().__init__(**kwargs)
        self.name = "roi_capture"
        self.roi = roi
        self.draw_roi = draw_roi

    def extract_roi(self, frame):
        """Crops the frame to the configured region of interest.

        Args:
            frame (numpy.ndarray): Input BGR image.

        Returns:
            numpy.ndarray: Cropped sub-image corresponding to the ROI.
        """
        x, y, w, h = self.roi
        return frame[y:y + h, x:x + w]

    def process(self, frame, **kwargs):
        """Draws the ROI overlay and optionally saves the cropped region.

        Args:
            frame (numpy.ndarray): Current camera frame.
            **kwargs: Supports:
                - 'capture_roi' (bool): If True, crops and saves the ROI region.
                - 'cap': Unused; present for interface compatibility.

        Returns:
            numpy.ndarray: Frame with the ROI rectangle drawn (if draw_roi is True).
        """
        output = frame.copy()

        if self.draw_roi:
            x, y, w, h = self.roi
            cv2.rectangle(output, (x, y), (x + w, y + h), (0, 255, 0), 2)

        if kwargs.get("capture_roi", False):
            roi_frame = self.extract_roi(frame)
            self.save_frame(roi_frame)

        return output
