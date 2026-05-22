from __future__ import annotations

import hashlib
import os
import urllib.request
from pathlib import Path


class ModelDownloader:
    """Downloads model weights to a local cache directory.

    Args:
        cache_dir: Local cache directory. Defaults to ~/.cache/ai_vision_tool/models/.
    """

    def __init__(self, cache_dir: str | None = None):
        self.cache_dir = Path(cache_dir or Path.home() / ".cache" / "ai_vision_tool" / "models")
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def download(self, url: str, filename: str | None = None, checksum: str | None = None) -> str:
        name = filename or Path(url).name
        dest = self.cache_dir / name

        if dest.exists() and checksum:
            actual = self.checksum_sha256(str(dest))
            if actual == checksum:
                print(f"[ModelDownloader] Cache hit: {dest}")
                return str(dest)
        elif dest.exists() and not checksum:
            print(f"[ModelDownloader] Cache hit: {dest}")
            return str(dest)

        print(f"[ModelDownloader] Downloading {url} → {dest}")
        downloaded_mb = [0.0]

        def _progress(count, block, total):
            mb = count * block / 1024 ** 2
            if mb - downloaded_mb[0] >= 10:
                downloaded_mb[0] = mb
                total_mb = total / 1024 ** 2 if total > 0 else "?"
                print(f"[ModelDownloader] {mb:.0f} MB / {total_mb} MB")

        urllib.request.urlretrieve(url, str(dest), reporthook=_progress)
        print(f"[ModelDownloader] Done: {dest}")

        if checksum:
            actual = self.checksum_sha256(str(dest))
            if actual != checksum:
                dest.unlink()
                raise ValueError(f"Checksum mismatch for {name}: expected {checksum}, got {actual}")

        return str(dest)

    def from_huggingface(self, repo_id: str, filename: str, revision: str = "main") -> str:
        url = f"https://huggingface.co/{repo_id}/resolve/{revision}/{filename}"
        cache_name = repo_id.replace("/", "__") + "_" + filename
        return self.download(url, filename=cache_name)

    @staticmethod
    def checksum_md5(path: str) -> str:
        h = hashlib.md5(usedforsecurity=False)
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                h.update(chunk)
        return h.hexdigest()

    @staticmethod
    def checksum_sha256(path: str) -> str:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                h.update(chunk)
        return h.hexdigest()
