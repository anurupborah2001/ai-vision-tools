import numpy as np
import cv2
import mss


class ScreenCapture:
    """Captures frames from a monitor using mss screen grabbing.

    Args:
        monitor_index (int): Index of the monitor to capture from. Default is 1.
    """

    def __init__(self, monitor_index=1):
        """Initializes ScreenCapture and selects the target monitor.

        Args:
            monitor_index (int): mss monitor list index. 0 is the combined virtual
                screen; 1 is the primary monitor. Default is 1.
        """
        self.sct = mss.mss()
        self.monitor = self.sct.monitors[monitor_index]

    def grab(self):
        """Captures a single frame from the configured monitor.

        Returns:
            numpy.ndarray: BGR image of the full monitor screen.
        """
        frame = np.array(self.sct.grab(self.monitor))
        return cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
