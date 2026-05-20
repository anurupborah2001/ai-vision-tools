from __future__ import annotations

import time
import tracemalloc

import numpy as np


class ModelBenchmark:
    """Profiles latency and memory of any AIVisionComponent or callable.

    Args:
        component: Component with .run(data) or any callable.
        warmup_runs: Warmup iterations before benchmarking.
        benchmark_runs: Timed iterations.
    """

    def __init__(self, component, warmup_runs: int = 5, benchmark_runs: int = 100):
        self.component = component
        self.warmup_runs = warmup_runs
        self.benchmark_runs = benchmark_runs

    def _call(self, sample_input):
        if hasattr(self.component, "run"):
            return self.component.run(sample_input)
        return self.component(sample_input)

    def run(self, sample_input) -> dict:
        for _ in range(self.warmup_runs):
            self._call(sample_input)

        latencies = []
        for _ in range(self.benchmark_runs):
            t0 = time.perf_counter()
            self._call(sample_input)
            latencies.append((time.perf_counter() - t0) * 1000.0)

        lats = np.array(latencies)
        mean_ms = float(lats.mean())
        return {
            "mean_ms": round(mean_ms, 3),
            "std_ms": round(float(lats.std()), 3),
            "min_ms": round(float(lats.min()), 3),
            "max_ms": round(float(lats.max()), 3),
            "p50_ms": round(float(np.percentile(lats, 50)), 3),
            "p95_ms": round(float(np.percentile(lats, 95)), 3),
            "p99_ms": round(float(np.percentile(lats, 99)), 3),
            "fps": round(1000.0 / mean_ms if mean_ms > 0 else 0.0, 2),
            "runs": self.benchmark_runs,
        }

    def run_memory(self, sample_input) -> dict:
        tracemalloc.start()
        self._call(sample_input)
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        result = self.run(sample_input)
        result["peak_memory_mb"] = round(peak / 1024 ** 2, 3)
        result["current_memory_mb"] = round(current / 1024 ** 2, 3)
        return result

    @staticmethod
    def print_report(results: dict) -> None:
        width = 42
        print("+" + "-" * width + "+")
        print(f"| {'Benchmark Report':^{width-2}} |")
        print("+" + "-" * width + "+")
        rows = [
            ("Runs", str(results.get("runs", "?"))),
            ("Mean latency", f"{results.get('mean_ms', 0):.3f} ms"),
            ("Std latency", f"{results.get('std_ms', 0):.3f} ms"),
            ("Min latency", f"{results.get('min_ms', 0):.3f} ms"),
            ("Max latency", f"{results.get('max_ms', 0):.3f} ms"),
            ("p50 latency", f"{results.get('p50_ms', 0):.3f} ms"),
            ("p95 latency", f"{results.get('p95_ms', 0):.3f} ms"),
            ("p99 latency", f"{results.get('p99_ms', 0):.3f} ms"),
            ("Throughput", f"{results.get('fps', 0):.2f} FPS"),
        ]
        if "peak_memory_mb" in results:
            rows.append(("Peak memory", f"{results['peak_memory_mb']:.3f} MB"))
        for k, v in rows:
            print(f"| {k:<20} {v:>{width-22}} |")
        print("+" + "-" * width + "+")
