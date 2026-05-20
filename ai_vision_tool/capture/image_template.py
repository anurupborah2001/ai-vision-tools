import cv2
import numpy as np
from typing import Callable, Optional
import pyautogui

window_centered = False


def image_template(
      image_path: str,
      custom_logic: Optional[Callable[[cv2.typing.MatLike], cv2.typing.MatLike]] = None,
      window_name: str = "Demo",
      center_window: bool = True,
      show_window: bool = True,
      resolution: tuple[int, int] = (1280, 720),
):
    """Displays a still image with optional custom processing in an OpenCV window.

    Loads the image from disk, optionally applies a user-supplied transform,
    resizes to the target resolution, and shows it in a named window. Blocks
    until any key is pressed.

    Args:
        image_path (str): Path to the image file to load.
        custom_logic (callable or None): Function that receives a BGR NumPy array
            and returns a processed BGR array. Applied before display. Default is None.
        window_name (str): Title of the OpenCV display window. Default is 'Demo'.
        center_window (bool): If True, repositions the window to the center of the
            screen using pyautogui. Default is True.
        show_window (bool): If True, renders the image in a window. Default is True.
        resolution (tuple[int, int]): Target display resolution as (width, height).
            Default is (1280, 720).

    Raises:
        ValueError: If the image is None after loading or after custom_logic.
        TypeError: If the image is not a NumPy array.
        ValueError: If the image array is empty.
    """
    image = cv2.imread(image_path)
    print(image_path)
    if custom_logic is not None:
        image = custom_logic(image)

    if image is None:
        raise ValueError("Image is None. Check file path or loading logic.")

    if not isinstance(image, np.ndarray):
        raise TypeError(f"Invalid image type: {type(image)}")

    if image.size == 0:
        raise ValueError("Empty image array")

    hWIDTH, hHEIGHT = resolution
    resized_img = cv2.resize(image, (hWIDTH, hHEIGHT))

    if center_window:
        screen_width, screen_height = pyautogui.size()
        x = int((screen_width - hWIDTH) / 2)
        y = int((screen_height - hHEIGHT) / 2)
        cv2.moveWindow(window_name, x, y)

    if show_window:
        cv2.imshow(window_name, resized_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
