import cv2
import os

from .base import AIVisionComponent


class FrameGrabber(AIVisionComponent):
    """Extracts still frames from a video file at a set interval."""
    
    def _execute(self, data, config):
        # 'data' in this context is expected to be a single video filepath string
        if not data or not os.path.exists(data):
            print(f"Error: Video file not found: {data}")
            return []

        # Extract configuration
        output_folder_name = config.get('output_folder', 'extracted_pics')
        skip_frames = config.get('skip_frames', 90)
        resize_factor = config.get('resize_factor', 0.5)
        
        folder_path = os.path.join(os.getcwd(), output_folder_name)
        if not os.path.isdir(folder_path):
            os.makedirs(folder_path)

        basefn = os.path.basename(data).split('.')[0]
        video = cv2.VideoCapture(data)
        
        im_count = 0
        frame_count = 0
        extracted_images = []

        print(f"[{self.__class__.__name__}] Processing video: {data}")

        try:
            while video.isOpened():
                hasFrame, frame = video.read()
                if not hasFrame:
                    print(f'Reached end of {data}!')
                    break

                frame_count += 1
                
                if frame_count >= skip_frames:
                    # Resize the frame
                    frame = cv2.resize(frame, None, fx=resize_factor, fy=resize_factor)
                    
                    im_count += 1
                    im_name = f"{basefn}-{im_count}.jpg"
                    savepath = os.path.join(folder_path, im_name)
                    
                    cv2.imshow('Extracted image (Press any key to close early)', frame)
                    cv2.waitKey(10) # Brief pause to display
                    
                    cv2.imwrite(savepath, frame)
                    extracted_images.append(savepath)
                    
                    frame_count = 0 # Reset counter
        finally:
            video.release()
            cv2.destroyAllWindows()

        # Returns the list of extracted images from THIS specific video
        return extracted_images