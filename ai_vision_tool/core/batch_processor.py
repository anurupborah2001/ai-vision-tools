from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import cv2


class BatchProcessor:
    """Processes images through a pipeline in configurable batch sizes.

    Args:
        pipeline: Object with .execute(data) or list of components with .run(data).
        batch_size: Number of images per batch.
        num_workers: Thread workers. 0 = single-threaded.
    """

    def __init__(self, pipeline, batch_size: int = 8, num_workers: int = 0):
        self.pipeline = pipeline
        self.batch_size = batch_size
        self.num_workers = num_workers

    def _run_one(self, item):
        if hasattr(self.pipeline, "execute"):
            return self.pipeline.execute(item)
        if isinstance(self.pipeline, list):
            data = item
            for comp in self.pipeline:
                data = comp.run(data)
            return data
        return self.pipeline(item)

    def process(self, images: list) -> list:
        results = [None] * len(images)
        if self.num_workers > 0:
            with ThreadPoolExecutor(max_workers=self.num_workers) as ex:
                futures = {
                    ex.submit(self._run_one, img): i for i, img in enumerate(images)
                }
                done = 0
                for fut in as_completed(futures):
                    results[futures[fut]] = fut.result()
                    done += 1
                    if done % self.batch_size == 0:
                        print(f"[BatchProcessor] {done}/{len(images)} processed")
        else:
            for i, img in enumerate(images):
                results[i] = self._run_one(img)
                if (i + 1) % self.batch_size == 0:
                    print(f"[BatchProcessor] {i+1}/{len(images)} processed")
        return results

    def process_directory(
        self,
        path: str,
        extensions: tuple[str, ...] = (".jpg", ".png", ".jpeg", ".bmp", ".webp"),
    ) -> list:
        paths = sorted(
            p for p in Path(path).iterdir() if p.suffix.lower() in extensions
        )
        images = []
        for p in paths:
            img = cv2.imread(str(p))
            if img is not None:
                images.append({"frame": img, "path": str(p)})
        print(f"[BatchProcessor] Found {len(images)} images in {path}")
        return self.process(images)
