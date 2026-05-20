from __future__ import annotations

import asyncio
import base64
import http.server
import io
import threading
import time

import cv2
import numpy as np

from ai_vision_tool.core.base import AIVisionComponent
from ai_vision_tool.utils.image_utils import extract_frame


class WebSocketSink(AIVisionComponent):
    """Pushes encoded frames to WebSocket clients. Falls back to MJPEG HTTP if websockets unavailable.

    Args:
        host: Host to bind server.
        port: Server port.
        quality: JPEG encoding quality.
        format: "jpeg" or "png".
    """

    def __init__(self, host: str = "0.0.0.0", port: int = 8765, quality: int = 80, format: str = "jpeg"):
        super().__init__()
        self.host = host
        self.port = port
        self.quality = quality
        self.format = format
        self._latest_frame: bytes | None = None
        self._lock = threading.Lock()
        self._loop: asyncio.AbstractEventLoop | None = None
        self._server_thread: threading.Thread | None = None
        self._clients: set = set()
        self._use_mjpeg = False

    def setup(self, config: dict):
        try:
            import websockets
            self._loop = asyncio.new_event_loop()
            self._server_thread = threading.Thread(target=self._run_ws_server, daemon=True)
            self._server_thread.start()
        except ImportError:
            print(f"[WebSocketSink] websockets not available, using MJPEG at http://{self.host}:{self.port}/stream")
            self._use_mjpeg = True
            self._server_thread = threading.Thread(target=self._run_mjpeg_server, daemon=True)
            self._server_thread.start()
        time.sleep(0.2)
        super().setup(config)

    def _encode_frame(self, frame: np.ndarray) -> bytes:
        ext = ".jpg" if self.format == "jpeg" else ".png"
        params = [cv2.IMWRITE_JPEG_QUALITY, self.quality] if self.format == "jpeg" else []
        _, buf = cv2.imencode(ext, frame, params)
        return buf.tobytes()

    def _run_ws_server(self):
        import websockets

        async def handler(ws, path=None):
            self._clients.add(ws)
            try:
                await ws.wait_closed()
            finally:
                self._clients.discard(ws)

        async def serve():
            async with websockets.serve(handler, self.host, self.port):
                await asyncio.Future()

        asyncio.set_event_loop(self._loop)
        self._loop.run_until_complete(serve())

    def _run_mjpeg_server(self):
        sink = self

        class Handler(http.server.BaseHTTPRequestHandler):
            def log_message(self, *a): pass

            def do_GET(self):
                if self.path == "/stream":
                    self.send_response(200)
                    self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
                    self.end_headers()
                    while True:
                        with sink._lock:
                            data = sink._latest_frame
                        if data:
                            self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\n\r\n")
                            self.wfile.write(data)
                            self.wfile.write(b"\r\n")
                        time.sleep(0.033)
                else:
                    self.send_response(200)
                    self.send_header("Content-Type", "text/html")
                    self.end_headers()
                    self.wfile.write(f'<html><body><img src="/stream"/></body></html>'.encode())

        srv = http.server.HTTPServer((self.host, self.port), Handler)
        srv.serve_forever()

    def _execute(self, data, config):
        frame = extract_frame(data)
        encoded = self._encode_frame(frame)
        with self._lock:
            self._latest_frame = encoded

        if not self._use_mjpeg and self._loop and self._clients:
            b64 = base64.b64encode(encoded).decode()
            msg = f'{{"frame": "{b64}"}}'

            async def broadcast():
                for ws in list(self._clients):
                    try:
                        await ws.send(msg)
                    except Exception:
                        self._clients.discard(ws)

            asyncio.run_coroutine_threadsafe(broadcast(), self._loop)
        return data

    def cleanup(self):
        if self._loop:
            self._loop.stop()


class WebSocketSource(AIVisionComponent):
    """Reads frames from a WebSocket server.

    Args:
        url: WebSocket server URL (e.g. ws://localhost:8765).
    """

    def __init__(self, url: str):
        super().__init__()
        self.url = url
        self._ws = None
        self._loop: asyncio.AbstractEventLoop | None = None

    def setup(self, config: dict):
        try:
            import websockets
        except ImportError:
            raise ImportError("Install with: pip install websockets")
        self._loop = asyncio.new_event_loop()
        super().setup(config)

    def _execute(self, data, config):
        import websockets

        async def recv():
            async with websockets.connect(self.url) as ws:
                return await ws.recv()

        msg = self._loop.run_until_complete(recv())
        if isinstance(msg, str):
            import json
            d = json.loads(msg)
            raw = base64.b64decode(d["frame"])
        else:
            raw = msg
        buf = np.frombuffer(raw, dtype=np.uint8)
        frame = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        return {"frame": frame, "source": self.url}
