from __future__ import annotations

import cv2
import numpy as np

from ai_vision_tool.core.base import AIVisionComponent


class GCSSource(AIVisionComponent):
    """Reads images from a Google Cloud Storage bucket.

    Args:
        bucket: GCS bucket name.
        prefix: Blob prefix to filter.
        extensions: Image file extensions to include.
        project: GCP project ID (None = infer from environment).
        credentials_path: Path to service account JSON (None = use ADC).
    """

    def __init__(
        self,
        bucket: str,
        prefix: str = "",
        extensions: tuple = (".jpg", ".png"),
        project: str | None = None,
        credentials_path: str | None = None,
    ):
        super().__init__()
        self.bucket = bucket
        self.prefix = prefix
        self.extensions = tuple(e.lower() for e in extensions)
        self.project = project
        self.credentials_path = credentials_path
        self._client = None
        self._blobs: list = []
        self._index = 0

    def setup(self, config: dict):
        try:
            from google.cloud import storage
        except ImportError as exc:
            raise ImportError(
                "GCS source requires: pip install ai-vision-tool[cloud]"
            ) from exc
        if self.credentials_path:
            self._client = storage.Client.from_service_account_json(
                self.credentials_path, project=self.project
            )
        else:
            self._client = storage.Client(project=self.project)
        bucket = self._client.bucket(self.bucket)
        self._blobs = [
            b
            for b in self._client.list_blobs(bucket, prefix=self.prefix)
            if any(b.name.lower().endswith(ext) for ext in self.extensions)
        ]
        print(
            f"[GCSSource] Found {len(self._blobs)} images in gs://{self.bucket}/{self.prefix}"
        )
        super().setup(config)

    def _execute(self, data, config):
        if isinstance(data, str):
            blob = self._client.bucket(self.bucket).blob(data)
        elif self._blobs:
            blob = self._blobs[self._index % len(self._blobs)]
            self._index += 1
        else:
            raise ValueError("GCSSource: no blob to read")

        buf = np.frombuffer(blob.download_as_bytes(), dtype=np.uint8)
        frame = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        if frame is None:
            raise OSError(f"GCSSource: could not decode {blob.name!r}")

        payload = data if isinstance(data, dict) else {}
        payload["frame"] = frame
        payload["blob_name"] = blob.name
        payload["bucket"] = self.bucket
        payload["source"] = "gcs"
        return payload
