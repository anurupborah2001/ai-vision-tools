from .image_capture import PictureTaker
import time


class BurstPictureTaker(PictureTaker):
    """Captures a rapid sequence of still images from a webcam feed.

    Extends PictureTaker to support burst mode: when triggered, reads
    ``burst_count`` frames in quick succession separated by ``interval_seconds``.

    Args:
        burst_count (int): Number of frames to capture per burst. Default is 5.
        interval_seconds (float): Delay between consecutive captures in seconds.
            Default is 0.2.
    """

    def __init__(self, burst_count=5, interval_seconds=0.2, **kwargs):
        """Initializes BurstPictureTaker with burst parameters.

        Args:
            burst_count (int): Frames per burst. Default is 5.
            interval_seconds (float): Sleep time between captures. Default is 0.2.
            **kwargs: Forwarded to PictureTaker.
        """
        super().__init__(**kwargs)
        self.name = "burst_picture_taker"
        self.burst_count = burst_count
        self.interval_seconds = interval_seconds

    def process(self, frame, **kwargs):
        """Optionally triggers a burst capture on the current frame.

        Args:
            frame (numpy.ndarray): Current camera frame (returned unchanged).
            **kwargs: Supports 'burst_capture' (bool) and 'cap' (cv2.VideoCapture).
                If 'burst_capture' is True, ``capture_burst`` is called with 'cap'.

        Returns:
            numpy.ndarray: The original frame, unchanged.
        """
        if kwargs.get("burst_capture", False):
            self.capture_burst(kwargs.get("cap"))
        return frame

    def capture_burst(self, cap):
        """Reads and saves a sequence of frames from an open VideoCapture.

        Args:
            cap (cv2.VideoCapture): Open camera capture object to read frames from.

        Returns:
            list[str]: Paths of saved frame images. May be shorter than
                ``burst_count`` if the camera stops delivering frames early.
        """
        saved = []

        for i in range(self.burst_count):
            ok, frame = cap.read()
            if not ok:
                break

            path = self.save_frame(frame)
            saved.append(path)

            time.sleep(self.interval_seconds)

        print(f"Burst saved: {len(saved)} images")
        return saved
