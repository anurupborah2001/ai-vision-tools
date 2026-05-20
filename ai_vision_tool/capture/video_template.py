import time
import cv2
import numpy as np
from PIL import ImageGrab
from typing import Callable, Optional, Union

import pyautogui
from pathlib import Path
from datetime import datetime

from ai_vision_tool.capture.video_recorder import VideoRecorder


def save_screenshot(frame, output_dir="screenshots", prefix="capture"):
    """Saves a single frame as a timestamped PNG file.

    Args:
        frame (numpy.ndarray): BGR image to save.
        output_dir (str): Directory where the file is written. Created if absent.
            Default is 'screenshots'.
        prefix (str): Filename prefix before the timestamp. Default is 'capture'.

    Returns:
        str: Absolute path of the saved PNG file.
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    filename = Path(output_dir) / f"{prefix}_{timestamp}.png"
    cv2.imwrite(str(filename), frame)
    print(f"Screenshot saved: {filename}")
    return str(filename)


def video_capture_template(
    video_source: Union[int, str] = 0,
    loop_forever: bool = True,
    custom_logic: Optional[Callable[[cv2.typing.MatLike], cv2.typing.MatLike]] = None,
    window_name: str = "Demo",
    show_window: bool = True,
    resolution: tuple[int, int] = (1280, 720),
    center_window: bool = True,
    draw_fps: bool = True,
    fps=15,
    mouse_callback: Optional[Callable] = None,
    mouse_callback_params: Optional[dict] = None,
    enable_recording: bool = False,
    record_format="mp4",
    enable_screenshot: bool = False,
    screenshot_output_dir: str = "screenshots",
    screenshot_prefix: str = "capture",
    auto_screenshot_after_seconds: Optional[float] = None,
    auto_screenshot_repeat: bool = False,
):
    """Reusable OpenCV video capture loop with optional recording and screenshot support.

    Opens a camera or video file, applies optional per-frame custom logic, and
    displays the result in a named window. Press ESC to exit. Press ``s`` or ``S``
    to take a manual screenshot when ``enable_screenshot`` is True.

    Args:
        video_source (int or str): Camera index (int) or path to a video file (str).
            Default is 0 (primary webcam).
        loop_forever (bool): If True, loops a video file back to the start when it ends.
            Default is True.
        custom_logic (callable or None): Function that accepts a BGR frame and returns
            a processed BGR frame. Applied every iteration. Default is None.
        window_name (str): Title of the OpenCV display window. Default is 'Demo'.
        show_window (bool): If True, renders frames in an OpenCV window. Default is True.
        resolution (tuple[int, int]): Camera capture resolution as (width, height).
            Default is (1280, 720).
        center_window (bool): If True, repositions the window to the screen center once
            using pyautogui. Default is True.
        draw_fps (bool): If True, overlays a real-time FPS counter on each frame.
            Default is True.
        fps (int or float): Frame rate for video recording (used only when
            ``enable_recording`` is True). Default is 15.
        mouse_callback (callable or None): OpenCV mouse event callback registered on
            the window. Default is None.
        mouse_callback_params (dict or None): Extra parameters forwarded to the mouse
            callback. Default is None.
        enable_recording (bool): If True, records all displayed frames to a file
            via VideoRecorder. Default is False.
        record_format (str): Recording container format — 'mp4' or 'gif'. Default is 'mp4'.
        enable_screenshot (bool): If True, activates screenshot functionality. Default is False.
        screenshot_output_dir (str): Directory for saved screenshots. Default is 'screenshots'.
        screenshot_prefix (str): Filename prefix for screenshots. Default is 'capture'.
        auto_screenshot_after_seconds (float or None): If set, automatically saves a
            screenshot after this many seconds. Default is None (disabled).
        auto_screenshot_repeat (bool): If True, repeats auto-screenshots at the specified
            interval. Default is False.
    """
    cap = cv2.VideoCapture(video_source)
    recorder = (
        VideoRecorder(output_format=record_format, fps=fps)
        if enable_recording
        else None
    )

    if not cap.isOpened():
        print(f"Error: Could not open video source '{video_source}'")
        return

    frame_width, frame_height = resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)

    window_centered = False
    recorder_started = False
    if draw_fps:
        _fps_tick = cv2.getTickCount()
        _fps_value = 0.0

    start_time = time.time()
    last_auto_screenshot_time = start_time
    auto_screenshot_done = False

    if show_window:
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, frame_width, frame_height)

        if mouse_callback is not None:
            cv2.setMouseCallback(window_name, mouse_callback, mouse_callback_params)

    while True:
        if loop_forever and cap.get(cv2.CAP_PROP_POS_FRAMES) == cap.get(cv2.CAP_PROP_FRAME_COUNT):
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

        ret, frame = cap.read()
        if not ret:
            print("End of video stream or failed to read frame.")
            break

        if custom_logic is not None:
            frame = custom_logic(frame)

        if draw_fps:
            _tick_now = cv2.getTickCount()
            _fps_value = cv2.getTickFrequency() / max(_tick_now - _fps_tick, 1)
            _fps_tick = _tick_now
            cv2.putText(frame, f"FPS: {_fps_value:.1f}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            fps = _fps_value

        if recorder:
            if not recorder_started:
                recorder.start(frame.shape)
                recorder_started = True

            recorder.write(frame)

            cv2.putText(
                frame,
                f"REC: {recorder.get_elapsed_time()}s",
                (20, 80),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 255),
                2,
            )

        if enable_screenshot and auto_screenshot_after_seconds is not None:
            now = time.time()

            if auto_screenshot_repeat:
                if now - last_auto_screenshot_time >= auto_screenshot_after_seconds:
                    save_screenshot(
                        frame,
                        output_dir=screenshot_output_dir,
                        prefix=screenshot_prefix,
                    )
                    last_auto_screenshot_time = now
            else:
                if not auto_screenshot_done and now - start_time >= auto_screenshot_after_seconds:
                    save_screenshot(
                        frame,
                        output_dir=screenshot_output_dir,
                        prefix=screenshot_prefix,
                    )
                    auto_screenshot_done = True

        if show_window:
            cv2.imshow(window_name, frame)

        if center_window and not window_centered:
            screen_width, screen_height = pyautogui.size()
            x = int((screen_width - frame_width) / 2)
            y = int((screen_height - frame_height) / 2)
            cv2.moveWindow(window_name, x, y)
            window_centered = True

        key = cv2.waitKey(1) & 0xFF

        if key == 27:
            break

        if enable_screenshot and key in [ord("s"), ord("S")]:
            save_screenshot(
                frame,
                output_dir=screenshot_output_dir,
                prefix=screenshot_prefix,
            )

    cap.release()
    cv2.destroyAllWindows()
    if recorder:
        recorder.stop()
