"""Structured per-request JSONL logger (issue #40).

Injectable via app.state.request_logger so tests can substitute a capture
buffer without file I/O or monkeypatching globals.
"""
import json
import logging
import os
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import IO

logger = logging.getLogger(__name__)

DEFAULT_LOG_PATH = Path.home() / ".local" / "state" / "claude-proxy" / "requests.jsonl"

_REQUIRED_FIELDS = frozenset({
    "request_id",
    "timestamp",
    "profile_name",
    "profile_kind",
    "requested_model",
    "upstream_model",
    "upstream_host",
    "method",
    "path",
    "status",
    "latency_ms",
    "streamed",
    "run_id",
    "role",
    "ticket",
    "token_drift_input",
    "token_drift_output",
})


class RequestLogger:
    """Emits one JSONL record per request to a file and stderr.

    Pass `capture` (a writable text IO) to intercept records in tests.
    Pass `log_path` to override the file path (also overrides CCPROXY_LOG_FILE).
    """

    def __init__(
        self,
        capture: IO[str] | None = None,
        log_path: Path | None = None,
    ) -> None:
        self._capture = capture
        self._log_path = log_path

    def _resolved_path(self) -> Path:
        if self._log_path is not None:
            return self._log_path
        env = os.getenv("CCPROXY_LOG_FILE")
        if env:
            return Path(env)
        return DEFAULT_LOG_PATH

    def make_record(
        self,
        *,
        profile_name: str,
        profile_kind: str,
        requested_model: str,
        upstream_model: str,
        upstream_host: str,
        method: str,
        path: str,
        status: int,
        latency_ms: float,
        streamed: bool,
        run_id: str | None = None,
        role: str | None = None,
        ticket: str | None = None,
        token_drift_input: int | None = None,
        token_drift_output: int | None = None,
        cache_read_input_tokens: int | None = None,
        cache_creation_input_tokens: int | None = None,
        cache_miss_estimate: int | None = None,
    ) -> dict:
        return {
            "request_id": str(uuid.uuid4()),
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "profile_name": profile_name,
            "profile_kind": profile_kind,
            "requested_model": requested_model,
            "upstream_model": upstream_model,
            "upstream_host": upstream_host,
            "method": method,
            "path": path,
            "status": status,
            "latency_ms": latency_ms,
            "streamed": streamed,
            "run_id": run_id,
            "role": role,
            "ticket": ticket,
            "token_drift_input": token_drift_input,
            "token_drift_output": token_drift_output,
            "cache_read_input_tokens": cache_read_input_tokens,
            "cache_creation_input_tokens": cache_creation_input_tokens,
            "cache_miss_estimate": cache_miss_estimate,
        }

    def emit(self, record: dict) -> None:
        """Write record atomically as a single JSONL line to file and stderr."""
        line = json.dumps(record) + "\n"

        sys.stderr.write(f"INFO {line}")
        sys.stderr.flush()

        if self._capture is not None:
            self._capture.write(line)
            self._capture.flush()
        else:
            path = self._resolved_path()
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, "a") as fh:
                fh.write(line)
