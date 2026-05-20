from __future__ import annotations

import copy
from concurrent.futures import ThreadPoolExecutor, as_completed


class ParallelPipeline:
    """Runs multiple component branches in parallel and merges results.

    Args:
        branches: List of component lists (each list is one branch).
        merge_fn: Callable that merges list of branch outputs. Default: return list.
        global_config: Config passed to all components.
    """

    def __init__(self, branches: list[list], merge_fn=None, global_config: dict | None = None):
        self._branches = branches
        self.merge_fn = merge_fn or (lambda results: results)
        self.global_config = global_config or {}

    def _run_branch(self, branch: list, data, config: dict):
        branch_data = copy.deepcopy(data) if isinstance(data, dict) else data
        for comp in branch:
            branch_data = comp.run(branch_data, config)
        return branch_data

    def execute(self, data) -> object:
        config = self.global_config
        results = [None] * len(self._branches)
        with ThreadPoolExecutor(max_workers=len(self._branches)) as ex:
            futures = {ex.submit(self._run_branch, branch, data, config): i
                       for i, branch in enumerate(self._branches)}
            for fut in as_completed(futures):
                results[futures[fut]] = fut.result()
        return self.merge_fn(results)

    def cleanup(self):
        for branch in self._branches:
            for comp in branch:
                if hasattr(comp, "cleanup"):
                    comp.cleanup()

    @staticmethod
    def merge_bboxes(results: list) -> dict:
        merged = dict(results[0]) if isinstance(results[0], dict) else {"result": results[0]}
        all_bboxes = []
        for r in results:
            if isinstance(r, dict):
                all_bboxes.extend(r.get("bboxes", []))
        merged["bboxes"] = all_bboxes
        return merged

    @staticmethod
    def merge_masks(results: list) -> dict:
        merged = dict(results[0]) if isinstance(results[0], dict) else {}
        all_masks = []
        for r in results:
            if isinstance(r, dict):
                all_masks.extend(r.get("masks", []))
        merged["masks"] = all_masks
        return merged

    @staticmethod
    def merge_first(results: list):
        return results[0]

    @staticmethod
    def merge_vote(results: list) -> dict:
        all_boxes = []
        all_scores = []
        for r in results:
            if isinstance(r, dict):
                for bb in r.get("bboxes", []):
                    all_boxes.append([bb["x1"], bb["y1"], bb["x2"], bb["y2"]])
                    all_scores.append(bb.get("conf", 1.0))
        if not all_boxes:
            return results[0] if results else {}
        import numpy as np
        from ai_vision_tool.detection.object_detector import ObjectDetector
        boxes = np.array(all_boxes)
        scores = np.array(all_scores)
        keep = ObjectDetector._nms(boxes, scores, 0.5)
        merged = dict(results[0]) if isinstance(results[0], dict) else {}
        merged["bboxes"] = [results[0]["bboxes"][k] for k in keep if k < len(results[0].get("bboxes", []))]
        return merged


class FanOutPipeline:
    """Sequential shared processing followed by parallel branch processing.

    Args:
        shared: Components run sequentially on all data first.
        branches: List of component lists run in parallel after shared.
    """

    def __init__(self, shared: list, branches: list[list], global_config: dict | None = None):
        self._shared = shared
        self._parallel = ParallelPipeline(branches, global_config=global_config)

    def execute(self, data) -> object:
        config = self._parallel.global_config
        for comp in self._shared:
            data = comp.run(data, config)
        return self._parallel.execute(data)

    def cleanup(self):
        for comp in self._shared:
            if hasattr(comp, "cleanup"):
                comp.cleanup()
        self._parallel.cleanup()
