from __future__ import annotations

import cv2
import numpy as np

from ai_vision_tool.core.base import AIVisionComponent


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
        except ImportError as exc:
            raise ImportError(
                "S3 source requires: pip install ai-vision-tool[cloud]"
            ) from exc
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
