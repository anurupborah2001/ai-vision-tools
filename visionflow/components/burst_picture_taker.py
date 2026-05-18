from .picture_taker import PictureTaker
import time

class BurstPictureTaker(PictureTaker):
    def __init__(self, burst_count=5, interval_seconds=0.2, **kwargs):
        super().__init__(**kwargs)
        self.name = "burst_picture_taker"
        self.burst_count = burst_count
        self.interval_seconds = interval_seconds

    def process(self, frame, **kwargs):
        if kwargs.get("burst_capture", False):
            self.capture_burst(kwargs.get("cap"))

        return frame

    def capture_burst(self, cap):
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