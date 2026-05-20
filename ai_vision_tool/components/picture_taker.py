import cv2
import os
from .base import AIVisionComponent


class PictureTaker(AIVisionComponent):
    """Captures still images from a webcam feed."""

    def _execute(self, data, config):
        """Opens a webcam window and saves frames on keypress.

        Displays a live preview window. Press ``p`` to save the current frame
        as a JPEG and ``q`` to quit. Images are numbered sequentially to
        avoid overwriting existing files.

        Args:
            data: Unused. Present for interface compatibility.
            config (dict): Runtime parameters. Supports:
                - 'imgdir' (str): Directory name for saved images. Default is 'Pics'.
                - 'resolution' (str): Camera resolution as 'WxH'. Default is '1280x720'.
                - 'camera_id' (int): OpenCV camera index. Default is 0.

        Returns:
            list[str]: Paths of all images saved during the session.
        """
        imgdir = config.get('imgdir', 'Pics')
        resolution = config.get('resolution', '1280x720')
        camera_id = config.get('camera_id', 0)

        imW, imH = map(int, resolution.split('x'))
        dirpath = os.path.join(os.getcwd(), imgdir)

        if not os.path.exists(dirpath):
            os.makedirs(dirpath)

        imnum = 1
        while os.path.exists(os.path.join(dirpath, f"{imgdir}-{imnum}.jpg")):
            imnum += 1

        cap = cv2.VideoCapture(camera_id)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, imW)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, imH)

        winname = 'Press "p" to take a picture! "q" to quit.'
        cv2.namedWindow(winname)
        cv2.moveWindow(winname, 50, 30)

        print(f"[{self.__class__.__name__}] Ready. Saving to {dirpath}")
        saved_images = []

        try:
            while True:
                hasFrame, frame = cap.read()
                if not hasFrame:
                    break

                cv2.imshow(winname, frame)
                key = cv2.waitKey(1) & 0xFF

                if key == ord('q'):
                    break
                elif key == ord('p'):
                    filename = f"{imgdir}-{imnum}.jpg"
                    savepath = os.path.join(dirpath, filename)
                    cv2.imwrite(savepath, frame)
                    print(f"Picture taken and saved as {filename}")
                    saved_images.append(savepath)
                    imnum += 1
        finally:
            cap.release()
            cv2.destroyAllWindows()

        return saved_images
