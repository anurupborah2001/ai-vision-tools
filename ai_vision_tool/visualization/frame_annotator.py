import cv2

from ai_vision_tool.core.base import AIVisionComponent


class FrameAnnotator(AIVisionComponent):
    """Draws shapes and text onto a frame based on provided annotations."""

    def _execute(self, data, config):
        """Renders annotation overlays onto the input frame.

        Annotations are read from ``data['annotations']`` when input is a dict,
        or from ``config['annotations']`` otherwise. Each annotation item is a
        dict with a ``'type'`` key and type-specific fields.

        Supported types:
            - ``'text'``: requires ``'text'`` (str) and optional ``'pos'`` (tuple[int,int]).
            - ``'box'``: requires ``'box'`` (tuple[int,int,int,int]) as (x, y, w, h).
            - ``'point'``: requires ``'point'`` (tuple[int,int]).

        Args:
            data: Input image as NumPy array or payload dict with 'frame' and
                optional 'annotations' keys.
            config (dict): Runtime parameters. Supports 'annotations' (list[dict])
                when data is not a dict.

        Returns:
            numpy.ndarray or dict: Annotated image in the same format as input.
        """
        frame = data["frame"] if isinstance(data, dict) else data
        output = frame.copy()

        annotations = (
            data.get("annotations", [])
            if isinstance(data, dict)
            else config.get("annotations", [])
        )

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

    def _draw_text(
        self, frame, text, pos=(30, 30), scale=0.8, color=(255, 255, 255), thickness=2
    ):
        """Renders a text string onto the frame at the given position.

        Args:
            frame (numpy.ndarray): Image to draw on (modified in place).
            text (str): Text content to render.
            pos (tuple[int, int]): Bottom-left origin of the text. Default is (30, 30).
            scale (float): Font scale factor. Default is 0.8.
            color (tuple[int, int, int]): BGR text color. Default is (255, 255, 255).
            thickness (int): Line thickness in pixels. Default is 2.
        """
        cv2.putText(
            frame, str(text), pos, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness
        )

    def _draw_box(self, frame, box, color=(0, 255, 0), thickness=2):
        """Draws a rectangle on the frame.

        Args:
            frame (numpy.ndarray): Image to draw on (modified in place).
            box (tuple[int, int, int, int]): Bounding box as (x, y, w, h).
            color (tuple[int, int, int]): BGR rectangle color. Default is (0, 255, 0).
            thickness (int): Line thickness in pixels. Default is 2.
        """
        x, y, w, h = box
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, thickness)

    def _draw_point(self, frame, point, color=(0, 0, 255), radius=6):
        """Draws a filled circle at the given point on the frame.

        Args:
            frame (numpy.ndarray): Image to draw on (modified in place).
            point (tuple[int, int]): Center (x, y) of the circle.
            color (tuple[int, int, int]): BGR circle color. Default is (0, 0, 255).
            radius (int): Circle radius in pixels. Default is 6.
        """
        cv2.circle(frame, point, radius, color, cv2.FILLED)
