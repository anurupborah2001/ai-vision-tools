import os

import cv2

from ai_vision_tool.core.base import AIVisionComponent


class FrameGrabber(AIVisionComponent):
    """Extracts still frames from a video file at a set interval."""

    def _execute(self, data, config):
        """Reads a video file and saves sampled frames to disk.

        Args:
            data (str): Absolute or relative path to the video file.
            config (dict): Runtime parameters. Supports:
                - 'output_folder' (str): Directory name for saved frames.
                  Default is 'extracted_pics'.
                - 'skip_frames' (int): Number of frames to skip between captures.
                  Default is 90.
                - 'resize_factor' (float): Scale factor applied to each saved frame.
                  Default is 0.5.

        Returns:
            list[str]: Paths of all saved frame images. Empty list if the video
                file is not found or contains no frames.
        """
        if not data or not os.path.exists(data):
            print(f"Error: Video file not found: {data}")
            return []

        output_folder_name = config.get("output_folder", "extracted_pics")
        skip_frames = config.get("skip_frames", 90)
        resize_factor = config.get("resize_factor", 0.5)

        folder_path = os.path.join(os.getcwd(), output_folder_name)
        if not os.path.isdir(folder_path):
            os.makedirs(folder_path)

        basefn = os.path.basename(data).split(".")[0]
        video = cv2.VideoCapture(data)

        im_count = 0
        frame_count = 0
        extracted_images = []

        print(f"[{self.__class__.__name__}] Processing video: {data}")

        try:
            while video.isOpened():
                hasFrame, frame = video.read()
                if not hasFrame:
                    print(f"Reached end of {data}!")
                    break

                frame_count += 1

                if frame_count >= skip_frames:
                    frame = cv2.resize(frame, None, fx=resize_factor, fy=resize_factor)

                    im_count += 1
                    im_name = f"{basefn}-{im_count}.jpg"
                    savepath = os.path.join(folder_path, im_name)

                    cv2.imshow("Extracted image (Press any key to close early)", frame)
                    cv2.waitKey(10)

                    cv2.imwrite(savepath, frame)
                    extracted_images.append(savepath)

                    frame_count = 0
        finally:
            video.release()
            cv2.destroyAllWindows()

        return extracted_images
