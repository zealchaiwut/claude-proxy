"""Pluggable tokenizer abstraction (issue #53).

HeuristicTokenizer: chars/4, zero dependencies, always available.
OpenAITokenizer: lazy-imports tiktoken; falls back to heuristic with a
  single WARN log message if tiktoken is absent or the model is unknown.
"""
from __future__ import annotations

import logging
from typing import Protocol, runtime_checkable

_log = logging.getLogger(__name__)

# Module-level sentinel: warn only once per process lifetime (AC5).
_tiktoken_warn_emitted: bool = False


@runtime_checkable
class Tokenizer(Protocol):
    """Protocol for token counters.  count_tokens must be non-negative."""

    def count_tokens(self, messages: list[dict], model: str) -> int: ...


class HeuristicTokenizer:
    """Estimates token count as total_chars // 4 (no external dependencies)."""

    def count_tokens(self, messages: list[dict], model: str) -> int:
        total_chars = 0
        for msg in messages:
            content = msg.get("content", "")
            if isinstance(content, str):
                total_chars += len(content)
            elif isinstance(content, list):
                for block in content:
                    if isinstance(block, dict) and block.get("type") == "text":
                        total_chars += len(block.get("text", ""))
        return max(1, total_chars // 4)


_HEURISTIC = HeuristicTokenizer()


class OpenAITokenizer:
    """Exact token counter using tiktoken.

    Lazy-imports tiktoken on first call.  If tiktoken is not installed, emits
    one WARN-level log message and falls back to the heuristic for the rest of
    the process lifetime.  Unknown model names fall back to cl100k_base.
    """

    def count_tokens(self, messages: list[dict], model: str) -> int:
        global _tiktoken_warn_emitted
        try:
            import tiktoken
        except (ImportError, Exception):
            if not _tiktoken_warn_emitted:
                _tiktoken_warn_emitted = True
                _log.warning(
                    "tiktoken is not installed; falling back to heuristic token counting"
                )
            return _HEURISTIC.count_tokens(messages, model)

        try:
            encoding = tiktoken.encoding_for_model(model)
        except KeyError:
            encoding = tiktoken.get_encoding("cl100k_base")

        total = 0
        for msg in messages:
            content = msg.get("content", "")
            if isinstance(content, str):
                total += len(encoding.encode(content))
            elif isinstance(content, list):
                for block in content:
                    if isinstance(block, dict) and block.get("type") == "text":
                        total += len(encoding.encode(block.get("text", "")))
        return max(1, total)


def get_tokenizer(name: str | None) -> Tokenizer:
    """Return a Tokenizer instance for the given name ('openai' | 'heuristic' | None)."""
    if name == "openai":
        return OpenAITokenizer()
    return HeuristicTokenizer()


def count_text_tokens(text: str) -> int:
    """Count tokens in text using cl100k_base BPE encoding (issue #54)."""
    if not text:
        return 0
    try:
        import tiktoken
        return len(tiktoken.get_encoding("cl100k_base").encode(text))
    except (ImportError, Exception):
        return max(1, len(text) // 4)


def count_messages_tokens(body_json: dict) -> int:
    """Count tokens across all messages in a request body using cl100k_base (issue #54)."""
    messages = body_json.get("messages", [])
    parts: list[str] = []
    for msg in messages:
        content = msg.get("content", "")
        if isinstance(content, str):
            parts.append(content)
        elif isinstance(content, list):
            for block in content:
                if isinstance(block, dict) and block.get("type") == "text":
                    parts.append(block.get("text", ""))
    text = "".join(parts)
    return count_text_tokens(text) if text else 1
