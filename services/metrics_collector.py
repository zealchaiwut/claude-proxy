"""In-memory rolling metrics aggregator (issue #43).

Injectable via app.state.metrics_collector so tests can substitute a fresh
instance without touching any global state.
"""
import os
import time
from collections import deque
from threading import Lock

_COST_PER_INPUT_TOKEN = 3.0 / 1_000_000
_COST_PER_OUTPUT_TOKEN = 15.0 / 1_000_000


def _percentile(data: list[float], p: float) -> float:
    """Linear-interpolation percentile on a sorted list. p is 0–100."""
    if not data:
        return 0.0
    n = len(data)
    if n == 1:
        return data[0]
    idx = (p / 100.0) * (n - 1)
    lo = int(idx)
    hi = min(lo + 1, n - 1)
    return data[lo] + (idx - lo) * (data[hi] - data[lo])


class MetricsCollector:
    """Accumulates per-profile request metrics in a bounded in-memory ring.

    Each sample is a tuple:
        (ts, profile, status, latency_ms, input_tokens, output_tokens, cost_usd)
    """

    def __init__(self) -> None:
        self._lock = Lock()
        self._samples: deque = deque()

    def _window_seconds(self) -> float | None:
        val = os.getenv("METRICS_WINDOW_SECONDS")
        if val:
            try:
                return float(val)
            except ValueError:
                pass
        return None

    def record(
        self,
        *,
        profile: str,
        status: int,
        latency_ms: float,
        input_tokens: int = 0,
        output_tokens: int = 0,
        cost_usd: float = 0.0,
        token_drift_input: int | None = None,
        token_drift_output: int | None = None,
    ) -> None:
        ts = time.monotonic()
        with self._lock:
            self._samples.append((ts, profile, status, latency_ms, input_tokens, output_tokens, cost_usd, token_drift_input, token_drift_output))

    def snapshot(self) -> dict:
        """Return a dict keyed by profile name with aggregated stats.

        Samples older than METRICS_WINDOW_SECONDS (if set) are excluded.
        Never reads or replays any log file.
        """
        now = time.monotonic()
        window = self._window_seconds()
        with self._lock:
            samples = list(self._samples)

        if window is not None:
            cutoff = now - window
            samples = [s for s in samples if s[0] >= cutoff]

        accum: dict[str, dict] = {}
        for sample in samples:
            _ts, profile, status, latency_ms, input_tokens, output_tokens, cost_usd = sample[:7]
            drift_in = sample[7] if len(sample) > 7 else None
            drift_out = sample[8] if len(sample) > 8 else None
            if profile not in accum:
                accum[profile] = {
                    "request_count": 0,
                    "error_count": 0,
                    "total_input_tokens": 0,
                    "total_output_tokens": 0,
                    "total_est_cost_usd": 0.0,
                    "_latencies": [],
                    "_drift_inputs": [],
                    "_drift_outputs": [],
                }
            entry = accum[profile]
            entry["request_count"] += 1
            entry["_latencies"].append(latency_ms)
            is_error = not (200 <= status < 300)
            if is_error:
                entry["error_count"] += 1
            else:
                entry["total_input_tokens"] += input_tokens
                entry["total_output_tokens"] += output_tokens
                entry["total_est_cost_usd"] += cost_usd
            if drift_in is not None:
                entry["_drift_inputs"].append(drift_in)
            if drift_out is not None:
                entry["_drift_outputs"].append(drift_out)

        result = {}
        for profile, entry in accum.items():
            lats = sorted(entry["_latencies"])
            drift_ins = entry["_drift_inputs"]
            drift_outs = entry["_drift_outputs"]
            result[profile] = {
                "request_count": entry["request_count"],
                "error_count": entry["error_count"],
                "total_input_tokens": entry["total_input_tokens"],
                "total_output_tokens": entry["total_output_tokens"],
                "total_est_cost_usd": entry["total_est_cost_usd"],
                "p50_latency_ms": _percentile(lats, 50),
                "p95_latency_ms": _percentile(lats, 95),
                "mean_drift_input": sum(drift_ins) / len(drift_ins) if drift_ins else None,
                "abs_mean_drift_input": sum(abs(d) for d in drift_ins) / len(drift_ins) if drift_ins else None,
                "mean_drift_output": sum(drift_outs) / len(drift_outs) if drift_outs else None,
                "abs_mean_drift_output": sum(abs(d) for d in drift_outs) / len(drift_outs) if drift_outs else None,
            }
        return result
