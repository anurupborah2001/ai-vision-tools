from __future__ import annotations

import http.server
import threading
import time

import cv2
import numpy as np

from ai_vision_tool.core.base import AIVisionComponent
from ai_vision_tool.utils.image_utils import extract_frame


class DashboardSink(AIVisionComponent):
    """Serves a live web dashboard via MJPEG HTTP stream (or Gradio if available).

    Args:
        host: Bind host.
        port: HTTP server port.
        quality: JPEG stream quality.
        title: Dashboard page title.
    """

    def __init__(self, host: str = "0.0.0.0", port: int = 7860, quality: int = 80,
                 title: str = "AI Vision Dashboard"):
        super().__init__()
        self.host = host
        self.port = port
        self.quality = quality
        self.title = title
        self._latest_frame: bytes | None = None
        self._lock = threading.Lock()
        self._server_thread: threading.Thread | None = None

    def setup(self, config: dict):
        self._server_thread = threading.Thread(target=self._start_server, daemon=True)
        self._server_thread.start()
        time.sleep(0.2)
        print(f"[DashboardSink] Live stream at http://{self.host}:{self.port}/")
        super().setup(config)

    def _start_server(self):
        try:
            import gradio as gr
            self._start_gradio()
        except ImportError:
            self._start_mjpeg()

    def _start_gradio(self):
        import gradio as gr
        sink = self

        def get_frame():
            with sink._lock:
                data = sink._latest_frame
            if data:
                import numpy as np
                buf = np.frombuffer(data, np.uint8)
                frame = cv2.imdecode(buf, cv2.IMREAD_COLOR)
                return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            return np.zeros((480, 640, 3), np.uint8)

        with gr.Blocks(title=self.title) as demo:
            gr.Markdown(f"# {self.title}")
            img = gr.Image(label="Live Feed")
            demo.load(get_frame, outputs=img, every=0.1)
        demo.launch(server_name=self.host, server_port=self.port, quiet=True)

    def _start_mjpeg(self):
        sink = self
        HTML = f"""<!DOCTYPE html>
<html><head><title>{self.title}</title></head>
<body style="background:#111;color:#eee;font-family:monospace">
<h2>{self.title}</h2>
<img src="/stream" style="max-width:100%;border:1px solid #444"/>
</body></html>"""

        class Handler(http.server.BaseHTTPRequestHandler):
            def log_message(self, *a): pass

            def do_GET(self):
                if self.path == "/stream":
                    self.send_response(200)
                    self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
                    self.end_headers()
                    try:
                        while True:
                            with sink._lock:
                                data = sink._latest_frame
                            if data:
                                self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + data + b"\r\n")
                            time.sleep(0.033)
                    except Exception:
                        pass
                else:
                    self.send_response(200)
                    self.send_header("Content-Type", "text/html")
                    self.end_headers()
                    self.wfile.write(HTML.encode())

        srv = http.server.HTTPServer((self.host, self.port), Handler)
        srv.serve_forever()

    def _execute(self, data, config):
        frame = extract_frame(data)
        _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, self.quality])
        with self._lock:
            self._latest_frame = buf.tobytes()
        return data
