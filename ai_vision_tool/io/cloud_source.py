from __future__ import annotations

import io

import cv2
import numpy as np

from ai_vision_tool.components.base import AIVisionComponent
from ai_vision_tool.components._image_utils import extract_frame, replace_frame


class S3Source(AIVisionComponent):
    """Reads images from an AWS S3 bucket.

    Args:
        bucket: S3 bucket name.
        prefix: Key prefix to filter objects.
        extensions: Image file extensions to include.
        region_name: AWS region (None = use default from env/config).
    """

    def __init__(self, bucket: str, prefix: str = "", extensions: tuple = (".jpg", ".png", ".jpeg"),
                 region_name: str | None = None):
        super().__init__()
        self.bucket = bucket
        self.prefix = prefix
        self.extensions = tuple(e.lower() for e in extensions)
        self.region_name = region_name
        self._client = None
        self._keys: list[str] = []
        self._index = 0

    def setup(self, config: dict):
        try:
            import boto3
        except ImportError:
            raise ImportError("Install with: pip install boto3")
        kwargs = {}
        if self.region_name:
            kwargs["region_name"] = self.region_name
        self._client = boto3.client("s3", **kwargs)
        paginator = self._client.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=self.bucket, Prefix=self.prefix):
            for obj in page.get("Contents", []):
                key = obj["Key"]
                if any(key.lower().endswith(ext) for ext in self.extensions):
                    self._keys.append(key)
        print(f"[S3Source] Found {len(self._keys)} images in s3://{self.bucket}/{self.prefix}")
        super().setup(config)

    def _execute(self, data, config):
        if isinstance(data, str):
            key = data
        elif isinstance(data, dict) and "key" in data:
            key = data["key"]
        elif self._keys:
            key = self._keys[self._index % len(self._keys)]
            self._index += 1
        else:
            raise ValueError("S3Source: no key to read")

        obj = self._client.get_object(Bucket=self.bucket, Key=key)
        buf = np.frombuffer(obj["Body"].read(), dtype=np.uint8)
        frame = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        if frame is None:
            raise IOError(f"S3Source: could not decode {key!r}")

        payload = data if isinstance(data, dict) else {}
        payload["frame"] = frame
        payload["key"] = key
        payload["bucket"] = self.bucket
        payload["source"] = "s3"
        return payload


class GCSSource(AIVisionComponent):
    """Reads images from a Google Cloud Storage bucket.

    Args:
        bucket: GCS bucket name.
        prefix: Blob prefix to filter.
        extensions: Image file extensions to include.
        project: GCP project ID (None = infer from environment).
        credentials_path: Path to service account JSON (None = use ADC).
    """

    def __init__(self, bucket: str, prefix: str = "", extensions: tuple = (".jpg", ".png"),
                 project: str | None = None, credentials_path: str | None = None):
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
        except ImportError:
            raise ImportError("Install with: pip install google-cloud-storage")
        if self.credentials_path:
            self._client = storage.Client.from_service_account_json(
                self.credentials_path, project=self.project
            )
        else:
            self._client = storage.Client(project=self.project)
        bucket = self._client.bucket(self.bucket)
        self._blobs = [
            b for b in self._client.list_blobs(bucket, prefix=self.prefix)
            if any(b.name.lower().endswith(ext) for ext in self.extensions)
        ]
        print(f"[GCSSource] Found {len(self._blobs)} images in gs://{self.bucket}/{self.prefix}")
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
            raise IOError(f"GCSSource: could not decode {blob.name!r}")

        payload = data if isinstance(data, dict) else {}
        payload["frame"] = frame
        payload["blob_name"] = blob.name
        payload["bucket"] = self.bucket
        payload["source"] = "gcs"
        return payload
