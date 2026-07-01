"""Per-request exchange capture service (issue #59).

Writes <request_id>.json to ~/.local/state/claude-proxy/captures/ when
capture is enabled globally (CCPROXY_CAPTURE=1) or per-profile (capture = true).
"""
import json
import os
import re
from pathlib import Path

DEFAULT_CAPTURE_DIR = Path.home() / ".local" / "state" / "claude-proxy" / "captures"

_CREDENTIAL_RE = re.compile(
    r"(authorization|api[._\-]?key|x[\-_]api[\-_]?key|secret|access[\-_]?token)",
    re.IGNORECASE,
)


def redact_credentials(data):
    """Recursively replace credential-like field values with '[REDACTED]'."""
    if isinstance(data, dict):
        return {
            k: "[REDACTED]" if _CREDENTIAL_RE.search(k) else redact_credentials(v)
            for k, v in data.items()
        }
    if isinstance(data, list):
        return [redact_credentials(item) for item in data]
    return data


def reassemble_anthropic_sse(buf: bytes) -> dict:
    """Parse buffered Anthropic SSE and assemble a single response object.

    Merges message_start metadata, accumulated text from content_block_delta,
    and final usage/stop_reason from message_delta into one dict.
    """
    message: dict = {}
    content_text = ""
    stop_reason = None
    final_usage: dict = {}

    for line in buf.decode("utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line.startswith("data:"):
            continue
        raw = line[len("data:"):].strip()
        try:
            payload = json.loads(raw)
        except (json.JSONDecodeError, ValueError):
            continue

        ptype = payload.get("type")
        if ptype == "message_start":
            message = dict(payload.get("message", {}))
        elif ptype == "content_block_delta":
            delta = payload.get("delta", {})
            if delta.get("type") == "text_delta":
                content_text += delta.get("text", "")
        elif ptype == "message_delta":
            delta = payload.get("delta", {})
            if delta.get("stop_reason"):
                stop_reason = delta["stop_reason"]
            final_usage.update(payload.get("usage", {}))

    assembled = dict(message)
    if content_text:
        assembled["content"] = [{"type": "text", "text": content_text}]
    if stop_reason:
        assembled["stop_reason"] = stop_reason
    if final_usage:
        existing_usage = dict(assembled.get("usage") or {})
        existing_usage.update(final_usage)
        assembled["usage"] = existing_usage

    return assembled


class CaptureService:
    """Writes capture files for individual requests when enabled.

    Pass capture_dir to override the default path (useful in tests).
    """

    def __init__(self, capture_dir: Path | None = None) -> None:
        self._dir = capture_dir or DEFAULT_CAPTURE_DIR

    def should_capture(self, profile_capture: bool = False) -> bool:
        """Return True when capture is enabled globally or for this profile."""
        env = os.getenv("CCPROXY_CAPTURE", "").strip()
        return env in ("1", "true", "yes") or profile_capture

    def write(
        self,
        *,
        request_id: str,
        inbound_body: dict,
        profile_name: str,
        profile_settings: dict,
        response_body: dict,
        start_ts: str,
        duration_ms: float,
    ) -> None:
        """Write a single capture JSON file for this request atomically."""
        self._dir.mkdir(parents=True, exist_ok=True)
        record = {
            "request_id": request_id,
            "profile": {
                "name": profile_name,
                "settings": redact_credentials(profile_settings),
            },
            "request": redact_credentials(inbound_body),
            "response": redact_credentials(response_body),
            "timing": {
                "start": start_ts,
                "duration_ms": duration_ms,
            },
        }
        usage = response_body.get("usage")
        if usage:
            record["usage"] = usage
        path = self._dir / f"{request_id}.json"
        path.write_text(json.dumps(record, indent=2))
