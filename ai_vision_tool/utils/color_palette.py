from __future__ import annotations

import numpy as np


class ColorPalette:
    """Deterministic BGR color palette with max visual separation via golden-ratio hue spacing."""

    def __init__(self, n_colors: int = 80, seed: int = 42):
        self._n = n_colors
        self._label_map: dict[str, int] = {}
        self._next_id = 0
        golden = 0.618033988749895
        rng = np.random.default_rng(seed)
        hues = np.mod(np.arange(n_colors) * golden, 1.0)
        saturations = rng.uniform(0.6, 1.0, n_colors)
        values = rng.uniform(0.7, 1.0, n_colors)
        self._colors: list[tuple[int, int, int]] = [
            self._hsv_to_bgr(h, s, v) for h, s, v in zip(hues, saturations, values)
        ]

    @staticmethod
    def _hsv_to_bgr(h: float, s: float, v: float) -> tuple[int, int, int]:
        i = int(h * 6)
        f = h * 6 - i
        p, q, t = v * (1 - s), v * (1 - f * s), v * (1 - (1 - f) * s)
        i %= 6
        rgb = [(v, t, p), (q, v, p), (p, v, t), (p, q, v), (t, p, v), (v, p, q)][i]
        return (int(rgb[2] * 255), int(rgb[1] * 255), int(rgb[0] * 255))

    def __getitem__(self, class_id: int) -> tuple[int, int, int]:
        return self._colors[class_id % self._n]

    def get(self, label: str) -> tuple[int, int, int]:
        if label not in self._label_map:
            self._label_map[label] = self._next_id
            self._next_id += 1
        return self[self._label_map[label]]

    def as_dict(self) -> dict[int, tuple[int, int, int]]:
        return {i: self._colors[i] for i in range(self._n)}
