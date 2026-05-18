import time
import cv2
import numpy as np
from PIL import ImageGrab
from typing import Callable, Optional, Union

from module.fps_counter import FPSCounter

import pyautogui
from pathlib import Path
from datetime import datetime

from template.video_recorder import VideoRecorder


def save_screenshot(frame, output_dir="screenshots", prefix="capture"):
  Path(output_dir).mkdir(parents=True, exist_ok=True)
  timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
  filename = Path(output_dir) / f"{prefix}_{timestamp}.png"
  cv2.imwrite(str(filename), frame)
  print(f"📸 Screenshot saved: {filename}")

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
    
    # MOUSE CALLBACK OPTION
    mouse_callback: Optional[Callable] = None,
    mouse_callback_params: Optional[dict] = None,
    
    # VIDEO RECORDING OPTIONS
    enable_recording: bool = False,
    record_format="mp4",   # "mp4" | "gif"

    # SCREENSHOT OPTIONS
    enable_screenshot: bool = False,
    screenshot_output_dir: str = "screenshots",
    screenshot_prefix: str = "capture",
    auto_screenshot_after_seconds: Optional[float] = None,
    auto_screenshot_repeat: bool = False,
):
    """
    REUSABLE TEMPLATE for all OpenCV video demos (UPDATED).

    New configurable features:
        - resolution: Set camera resolution (e.g. 1280x720, 1920x1080)
        - center_window: Automatically centers the OpenCV window on your screen using pyautogui

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
        show_window (bool): If True, displays the video window. Default = True
        window_name (str): Name of the OpenCV window.
        resolution (tuple[int, int]): Desired camera resolution (width, height). Default = (1280, 720)
        center_window (bool): If True, automatically centers the window on screen. Default = True
        draw_fps (bool): If True, calculates and displays FPS on the video feed. Default = True
        fps: Frame rate for recording (only applies if enable_recording is True). Default = 15  
        
        # MOUSE CALLBACK OPTION
        mouse_callback (callable, optional): Function to handle mouse events. Default = None
        mouse_callback_params (dict, optional): Additional parameters to pass to the mouse callback function. Default = None  
        
        # VIDEO RECORDING OPTIONS
        enable_recording (bool): If True, records the video feed to an output file. Default = False
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
          
        2. 
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

    # ======================
    # NEW: Set custom resolution (frame_width, frame_height)
    # ======================
    frame_width, frame_height = resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)
    
    window_centered = False   # Used to center window only once
    recorder_started = False
    if draw_fps:
      fps_counter = FPSCounter()
      
    start_time = time.time()
    last_auto_screenshot_time = start_time
    auto_screenshot_done = False
    
    if show_window:
      cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
      cv2.resizeWindow(window_name, frame_width, frame_height)

      if mouse_callback is not None:
          cv2.setMouseCallback(window_name, mouse_callback, mouse_callback_params)
        
    while True:
        ## 1. Start the timer
        # timer = cv2.getTickCount()
        if loop_forever and cap.get(cv2.CAP_PROP_POS_FRAMES) == cap.get(cv2.CAP_PROP_FRAME_COUNT):
          cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
          
        ## 2. Read and process the frame
        ret, frame = cap.read()
        if not ret:
            print("End of video stream or failed to read frame.")
            break
            
        if custom_logic is not None:
            frame = custom_logic(frame) 

        if draw_fps:  
          # fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer)
          # FPS calculation
          frame, fps = fps_counter.update(frame)
          
        # =========================
        # RECORDER ENGINE
        # =========================
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
        
        # AUTO SCREENSHOT AFTER N SECONDS
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
      
        # ESC exits
        if key == 27:
            break

        # Press S to screenshot
        if enable_screenshot and key in [ord("s"), ord("S")]:
            save_screenshot(
                frame,
                output_dir=screenshot_output_dir,
                prefix=screenshot_prefix,
            )

    #CLEANUP
    cap.release()
    cv2.destroyAllWindows()
    if recorder:
        recorder.stop()