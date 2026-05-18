
from .base import AIVisionComponent
import cv2
import os

class VideoTaker(AIVisionComponent):
    """Records video clips from a webcam feed."""
    
    def _execute(self, data, config):
        vid_dir = config.get('viddir', 'Videos')
        resolution = config.get('resolution', '1280x720')
        camera_id = config.get('camera_id', 0)
        fps = config.get('fps', 30.0)
        
        imW, imH = map(int, resolution.split('x'))
        dirpath = os.path.join(os.getcwd(), vid_dir)
        
        if not os.path.exists(dirpath):
            os.makedirs(dirpath)

        vidnum = 1
        while os.path.exists(os.path.join(dirpath, f"recording-{vidnum}.avi")):
            vidnum += 1

        cap = cv2.VideoCapture(camera_id)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, imW)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, imH)
        
        # Define the codec and create VideoWriter object
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = None
        recording = False
        
        winname = 'Press "r" to start/stop recording. "q" to quit.'
        cv2.namedWindow(winname)
        saved_videos = []

        print(f"[{self.__class__.__name__}] Ready. Saving to {dirpath}")

        try:
            while True:
                hasFrame, frame = cap.read()
                if not hasFrame: break
                
                # Add a visual indicator if recording
                display_frame = frame.copy()
                if recording:
                    cv2.circle(display_frame, (30, 30), 10, (0, 0, 255), -1) # Red recording dot
                    out.write(frame)
                    
                cv2.imshow(winname, display_frame)
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord('q'):
                    break
                elif key == ord('r'):
                    recording = not recording
                    if recording:
                        filename = f"recording-{vidnum}.avi"
                        savepath = os.path.join(dirpath, filename)
                        out = cv2.VideoWriter(savepath, fourcc, fps, (imW, imH))
                        print(f"Started recording: {filename}")
                    else:
                        out.release()
                        saved_videos.append(savepath)
                        print(f"Stopped recording. Saved to {filename}")
                        vidnum += 1
        finally:
            cap.release()
            if out is not None:
                out.release()
            cv2.destroyAllWindows()
            
        return saved_videos # Yields a list of saved video paths