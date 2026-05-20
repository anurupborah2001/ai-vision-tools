import cv2
import numpy as np
import os
import time
import imageio
from datetime import datetime
from dataclasses import dataclass
from typing import Optional


@dataclass
class VideoRecorder:
    """Records video to MP4 or GIF with pause/resume and elapsed-time tracking.

    Attributes:
        output_path (str): Directory where recordings are saved. Default is 'recordings'.
        fps (int): Output frame rate. Default is 20.
        codec (str): FourCC codec string for MP4 encoding. Default is 'mp4v'.
        output_format (str): Output container — 'mp4' or 'gif'. Default is 'mp4'.
    """

    output_path: str = "recordings"
    fps: int = 20
    codec: str = "mp4v"
    output_format: str = "mp4"

    def __post_init__(self):
        """Creates the output directory and initializes internal recorder state."""
        os.makedirs(self.output_path, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        self.file_path = os.path.join(
            self.output_path,
            f"record_{timestamp}.{self.output_format}"
        )

        self.writer = None
        self.frames = []

        self.is_recording = False
        self.is_paused = False

        self.start_time = None
        self.elapsed_time = 0

    def start(self, frame_shape=None):
        """Begins recording. Creates a VideoWriter for MP4 or prepares the GIF frame buffer.

        Args:
            frame_shape (tuple or None): Shape of the frames to be written (H, W[, C]).
                Required for MP4 mode to set the output dimensions. Unused for GIF mode.
        """
        self.is_recording = True
        self.is_paused = False
        self.start_time = time.time()

        self.frames = []

        if self.output_format == "mp4" and frame_shape is not None:
            h, w = frame_shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*self.codec)
            self.writer = cv2.VideoWriter(self.file_path, fourcc, self.fps, (w, h))

        print(f"Recording started → {self.file_path}")

    def write(self, frame):
        """Writes a single frame to the recording buffer.

        For MP4 output, the frame is written directly via the VideoWriter.
        For GIF output, the frame is converted to RGB and appended to the buffer.
        Frames are silently dropped when paused or not recording.

        Args:
            frame (numpy.ndarray): BGR image frame to record.
        """
        if not self.is_recording or self.is_paused:
            return

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        if self.output_format == "mp4" and self.writer:
            self.writer.write(frame)

        elif self.output_format == "gif":
            self.frames.append(rgb_frame)

    def get_elapsed_time(self):
        """Returns the number of seconds elapsed since recording started.

        Returns:
            float: Elapsed seconds rounded to 2 decimal places, or 0 if not started.
        """
        if self.start_time:
            return round(time.time() - self.start_time, 2)
        return 0

    def pause(self):
        """Pauses recording. Subsequent ``write`` calls are silently dropped."""
        self.is_paused = True
        print("Recording paused")

    def resume(self):
        """Resumes recording after a pause."""
        self.is_paused = False
        print("Recording resumed")

    def stop(self):
        """Stops recording and flushes the output file.

        For MP4 mode, releases the VideoWriter. For GIF mode, encodes and saves
        the accumulated frame buffer using imageio.
        """
        self.is_recording = False

        if self.writer:
            self.writer.release()
            print(f"MP4 saved → {self.file_path}")

        if self.output_format == "gif" and self.frames:
            imageio.mimsave(self.file_path, self.frames, fps=self.fps)
            print(f"GIF saved → {self.file_path}")

    def attach_audio(self, audio_path: str):
        """Placeholder for ffmpeg-based audio synchronization.

        Args:
            audio_path (str): Path to the audio file to attach (not yet implemented).
        """
        print(f"Audio sync not implemented yet: {audio_path}")
