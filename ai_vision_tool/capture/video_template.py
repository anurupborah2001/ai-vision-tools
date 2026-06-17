import time
import cv2
import numpy as np
from typing import Callable, Optional, Union
from pathlib import Path
from datetime import datetime

class KeyEventManager:
    def __init__(self):
        self.handlers = {}

    def register(self, key, callback):
        """
        key: ord('r'), ord('p'), etc.
        callback(frame, state)
        """
        self.handlers[key] = callback

    def handle(self, key, frame, state):
        if key in self.handlers:
            self.handlers[key](frame, state)
            
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


def _make_recorder(output_format="mp4", fps=15):
    from ai_vision_tool.capture.video_recorder import VideoRecorder
    return VideoRecorder(output_format=output_format, fps=fps)


def video_capture_template(
    video_source: Union[int, str] = 0,
    loop_forever: bool = True,
    custom_logic: Optional[Callable[[cv2.typing.MatLike], cv2.typing.MatLike]] = None,
<<<<<<< Updated upstream
=======
    state: dict | None = None,
    key_manager: KeyEventManager | None = None,
>>>>>>> Stashed changes
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

<<<<<<< Updated upstream
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
=======
    How to use:
    1. Define your own logic as a function that takes a frame and returns the processed frame.
    2. Call this template with the video source and your logic function.
    3. FPS counter, ESC exit, resolution control, and window centering are already handled.

    Parameters:
        video_source (int or str): 
            - int (e.g. 0, 1, 2...) → camera index
            - str → path to video file (mp4, avi, etc.)
        loop_forever (bool): If True, loops the video file when it ends. Default = True
        screen_capture (bool): If True, captures a portion of the screen instead of webcam/video. Default = False
        screen_capture_bbox (tuple): Bounding box for screen capture (left, top, right, bottom). Default = (300, 300, 1500, 1000)
        custom_logic (callable, optional): 
            Function that receives the frame and returns the modified frame.
            This is where you put ALL your own logic (blink detection, face detection, etc.).
        state (dict, optional):
            A dictionary that is passed to key handlers and can be used to store game state, scores, or any other information you need to persist across frames and key events.
             Default is None, but you can initialize it with your own dictionary before passing to the template. For example:
             state = {'score': [0, 0], 'game_over': False}
        key_manager (KeyEventManager, optional): An instance of KeyEventManager to handle key events. Default = None
        show_window (bool): If True, displays the video window. Default = True
        window_name (str): Name of the OpenCV window.
        resolution (tuple[int, int]): Desired camera resolution (width, height). Default = (1280, 720)
        center_window (bool): If True, automatically centers the window on screen. Default = True
        draw_fps (bool): If True, calculates and displays FPS on the video feed. Default = True
        fps: Frame rate for recording (only applies if enable_auto_recording is True). Default = 15  
        
        # MOUSE CALLBACK OPTION
        mouse_callback (callable, optional): Function to handle mouse events. Default = None
        mouse_callback_params (dict, optional): Additional parameters to pass to the mouse callback function. Default = None  
        
        # VIDEO RECORDING OPTIONS
        enable_auto_recording (bool): If True, records the video feed to an output file automatically. Default = False
        enable_manual_recording (bool): If True, allows starting/stopping recording with a key press (e.g. 'r' or 'R'). Default = False
        record_format (str): Format for recording output ("mp4" or "gif"). Default = "mp4"
      
         # SCREENSHOT OPTIONS
        enable_screenshot (bool): If True, allows taking screenshots by pressing 's'. Default = True
        screenshot_output_dir (str): Directory where screenshots will be saved. Default = "screenshots"
        screenshot_prefix (str): Prefix for screenshot filenames. Default = "capture"
        auto_screenshot_after_seconds (float, optional): If set, automatically takes a screenshot after this many seconds. Default = None (disabled)
        auto_screenshot_repeat (bool): If True and auto_screenshot_after_seconds is set, continues to take screenshots at the specified interval. Default = False
        
        
        Usasge:
        
        1. Screenshot:
          For repeated auto screenshots every 5 seconds:
          video_capture_template(
            video_source=0,
              custom_logic=my_logic,
              enable_screenshot=True,
              auto_screenshot_after_seconds=5,
              auto_screenshot_repeat=False,
          )
          For manual screenshots with 's' key:
          video_capture_template(
            video_source=0,
              custom_logic=my_logic,
              enable_screenshot=True,
              auto_screenshot_after_seconds=None,
              auto_screenshot_repeat=False,
          ) 
    """
  cap = cv2.VideoCapture(video_source)
 
  if not cap.isOpened():
      print(f"Error: Could not open video source '{video_source}'")
      return
    
  if state is None:
    state = {}
>>>>>>> Stashed changes

    if not cap.isOpened():
        print(f"Error: Could not open video source '{video_source}'")
        return

    frame_width, frame_height = resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)

<<<<<<< Updated upstream
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
=======
  # ── auto recording state ──────────────────────────────────────────────
  auto_recorder: Optional[object] = None
  auto_recorder_started = False

  # ── manual recording state ────────────────────────────────────────────
  manual_recording = False          # True while the user is recording
  manual_recorder: Optional[object] = None

  if draw_fps:
      prev_tick = cv2.getTickCount()
>>>>>>> Stashed changes

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

<<<<<<< Updated upstream
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
=======
      if draw_fps:
          tick = cv2.getTickCount()
          elapsed = (tick - prev_tick) / cv2.getTickFrequency()
          current_fps = 1.0 / elapsed if elapsed > 0 else fps
          prev_tick = tick  # noqa: F841
          cv2.putText(frame, f"FPS: {current_fps:.1f}", (10, 30),
                      cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

      # ── AUTO RECORDING ────────────────────────────────────────────────
      if enable_auto_recording:
          if auto_recorder is None:
              safe_fps = current_fps if current_fps and current_fps > 0 else 10
              print("Initializing auto-recorder with FPS:", safe_fps)
              auto_recorder = _make_recorder(output_format=record_format, fps=safe_fps)
>>>>>>> Stashed changes

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

<<<<<<< Updated upstream
        if center_window and not window_centered:
            screen_width, screen_height = pyautogui.size()
            x = int((screen_width - frame_width) / 2)
            y = int((screen_height - frame_height) / 2)
            cv2.moveWindow(window_name, x, y)
            window_centered = True
=======
      # ── MANUAL RECORDING ─────────────────────────────────────────────
      if enable_manual_recording and manual_recording:
          if manual_recorder is None:
              safe_fps = current_fps if current_fps and current_fps > 0 else 10
              print("Initializing manual recorder with FPS:", safe_fps)
              manual_recorder = _make_recorder(output_format=record_format, fps=safe_fps)
              manual_recorder.start(frame.shape)
>>>>>>> Stashed changes

        key = cv2.waitKey(1) & 0xFF

        if key == 27:
            break

        if enable_screenshot and key in [ord("s"), ord("S")]:
            save_screenshot(
                frame,
                output_dir=screenshot_output_dir,
                prefix=screenshot_prefix,
            )

<<<<<<< Updated upstream
    cap.release()
    cv2.destroyAllWindows()
    if recorder:
        recorder.stop()
=======
      if show_window:
          cv2.imshow(window_name, frame)

          if center_window and not window_centered and first_frame_rendered:
              import pyautogui  # optional dep — only needed for center_window
              screen_width, screen_height = pyautogui.size()
              x = (screen_width - frame_width) // 2
              y = (screen_height - frame_height) // 2
              cv2.moveWindow(window_name, x, y)
              window_centered = True

          first_frame_rendered = True

      key = cv2.waitKey(1) & 0xFF

      # ESC → exit
      if key == 27:
          print("Exiting cleanly...")
          break
        
      # Custom key handlers
      if key_manager:
          key_manager.handle(key, frame, state)

      # S → screenshot
      if enable_screenshot and key in [ord("s"), ord("S")]:
          save_screenshot(frame, output_dir=screenshot_output_dir, prefix=screenshot_prefix)

      # R → toggle manual recording on/off
      if enable_manual_recording and key in [ord("r"), ord("R")]:
          manual_recording = not manual_recording

          if manual_recording:
              # ── START ──────────────────────────────────────────────
              print("🎥 Manual recording: ON")
              # Recorder is created fresh each time so a new file is opened
              manual_recorder = None  # will be lazily created above on next frame

          else:
              # ── STOP ───────────────────────────────────────────────
              print("⏹️  Manual recording: OFF — saving…")
              if manual_recorder is not None:
                  manual_recorder.stop()
                  manual_recorder = None

  # ── CLEANUP ───────────────────────────────────────────────────────────
  cap.release()
  cv2.destroyAllWindows()

  if auto_recorder:
      print("Stopping auto-recorder…")
      auto_recorder.stop()

  if manual_recorder:
      print("Stopping manual recorder (cleanup)…")
      manual_recorder.stop()
>>>>>>> Stashed changes
