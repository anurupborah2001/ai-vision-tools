from .base import AIVisionComponent
import cv2

class FrameAnnotator(AIVisionComponent):
    """Draws shapes and text onto a frame based on provided annotations."""
    
    def _execute(self, data, config):
        frame = data["frame"] if isinstance(data, dict) else data
        output = frame.copy()
        
        # Annotations can be passed dynamically in the data payload or statically in config
        annotations = data.get("annotations", []) if isinstance(data, dict) else config.get("annotations", [])

        for item in annotations:
            ann_type = item.get("type")
            if ann_type == "text":
                self._draw_text(output, item["text"], item.get("pos", (30, 30)))
            elif ann_type == "box":
                self._draw_box(output, item["box"])
            elif ann_type == "point":
                self._draw_point(output, item["point"])

        if isinstance(data, dict):
            data["frame"] = output
            return data
        return output

    def _draw_text(self, frame, text, pos=(30, 30), scale=0.8, color=(255, 255, 255), thickness=2):
        cv2.putText(frame, str(text), pos, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness)

    def _draw_box(self, frame, box, color=(0, 255, 0), thickness=2):
        x, y, w, h = box
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, thickness)

    def _draw_point(self, frame, point, color=(0, 0, 255), radius=6):
        cv2.circle(frame, point, radius, color, cv2.FILLED)
