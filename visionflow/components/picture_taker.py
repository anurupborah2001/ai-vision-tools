import cv2
import os
from .base import AIVisionComponent

class PictureTaker(AIVisionComponent):
    """Captures still images from a webcam feed."""
    
    def _execute(self, data, config):
        # Extract configuration
        imgdir = config.get('imgdir', 'Pics')
        resolution = config.get('resolution', '1280x720')
        camera_id = config.get('camera_id', 0)
        
        imW, imH = map(int, resolution.split('x'))
        dirpath = os.path.join(os.getcwd(), imgdir)
        
        if not os.path.exists(dirpath):
            os.makedirs(dirpath)

        # Increment image number to prevent overwriting
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
                if not hasFrame: break
                
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
            
        return saved_images # Yields a list of saved image paths