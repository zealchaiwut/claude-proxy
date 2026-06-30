"""Tests for issue #53: pluggable tokenizer abstraction with tiktoken support."""
from __future__ import annotations

import logging
import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest


def _tiktoken_available() -> bool:
    try:
        import tiktoken  # noqa: F401
        return True
    except ImportError:
        return False


_KNOWN_MESSAGES = [{"role": "user", "content": "Hello, world!"}]
_KNOWN_MODEL = "gpt-4o"


# ---------------------------------------------------------------------------
# AC1: Tokenizer protocol is importable and has count_tokens(messages, model)
# ---------------------------------------------------------------------------


def test_tokenizer_protocol_importable():
    """AC1: Tokenizer protocol/ABC is importable from services.tokenizer."""
    from services.tokenizer import Tokenizer  # noqa: F401


def test_heuristic_tokenizer_importable():
    """AC1/AC7: HeuristicTokenizer is importable."""
    from services.tokenizer import HeuristicTokenizer  # noqa: F401


def test_openai_tokenizer_importable():
    """AC3: OpenAITokenizer is importable at module load time without error."""
    # This must not raise even when tiktoken is absent
    from services.tokenizer import OpenAITokenizer  # noqa: F401


def test_get_tokenizer_factory_importable():
    """AC2: get_tokenizer factory function is importable."""
    from services.tokenizer import get_tokenizer  # noqa: F401


# ---------------------------------------------------------------------------
# AC7: HeuristicTokenizer — chars/4, no external deps
# ---------------------------------------------------------------------------


def test_heuristic_tokenizer_string_content():
    """AC7: HeuristicTokenizer implements chars/4 heuristic on string content."""
    from services.tokenizer import HeuristicTokenizer

    t = HeuristicTokenizer()
    # "Hello" = 5 chars → max(1, 5//4) = 1
    assert t.count_tokens([{"role": "user", "content": "Hello"}], "gpt-4o") == 1


def test_heuristic_tokenizer_longer_string():
    """AC7: HeuristicTokenizer chars/4 for longer content."""
    from services.tokenizer import HeuristicTokenizer

    t = HeuristicTokenizer()
    # 20 chars → 20//4 = 5
    text = "A" * 20
    assert t.count_tokens([{"role": "user", "content": text}], "gpt-4o") == 5


def test_heuristic_tokenizer_multiblock_content():
    """AC7: HeuristicTokenizer handles list-of-blocks content."""
    from services.tokenizer import HeuristicTokenizer

    t = HeuristicTokenizer()
    msgs = [{"role": "user", "content": [{"type": "text", "text": "Hello world!"}]}]
    # "Hello world!" = 12 chars → 12//4 = 3
    assert t.count_tokens(msgs, "gpt-4o") == 3


def test_heuristic_tokenizer_min_one():
    """AC7: HeuristicTokenizer returns at least 1 even for empty messages."""
    from services.tokenizer import HeuristicTokenizer

    t = HeuristicTokenizer()
    assert t.count_tokens([], "any-model") >= 1
    assert t.count_tokens([{"role": "user", "content": ""}], "any-model") >= 1


def test_heuristic_tokenizer_no_tiktoken_needed():
    """AC7: HeuristicTokenizer works when tiktoken is forcibly absent."""
    with patch.dict(sys.modules, {"tiktoken": None}):
        from services.tokenizer import HeuristicTokenizer

        t = HeuristicTokenizer()
        result = t.count_tokens(_KNOWN_MESSAGES, _KNOWN_MODEL)
    assert isinstance(result, int) and result >= 1


def test_get_tokenizer_heuristic_returns_heuristic_instance():
    """AC7: get_tokenizer('heuristic') returns a HeuristicTokenizer."""
    from services.tokenizer import HeuristicTokenizer, get_tokenizer

    t = get_tokenizer("heuristic")
    assert isinstance(t, HeuristicTokenizer)


def test_get_tokenizer_default_is_heuristic():
    """AC7/AC9: get_tokenizer(None) falls back to HeuristicTokenizer."""
    from services.tokenizer import HeuristicTokenizer, get_tokenizer

    t = get_tokenizer(None)
    assert isinstance(t, HeuristicTokenizer)


# ---------------------------------------------------------------------------
# AC3 & AC4: OpenAITokenizer with tiktoken (skipped when absent)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _tiktoken_available(), reason="tiktoken not installed")
def test_openai_tokenizer_matches_tiktoken_string_content():
    """AC4: count_tokens output matches tiktoken exactly for string content."""
    import tiktoken

    from services.tokenizer import OpenAITokenizer

    model = "gpt-4o"
    text = "Hello, this is a test message."
    msgs = [{"role": "user", "content": text}]

    try:
        enc = tiktoken.encoding_for_model(model)
    except KeyError:
        enc = tiktoken.get_encoding("cl100k_base")

    expected = len(enc.encode(text))
    t = OpenAITokenizer()
    result = t.count_tokens(msgs, model)
    assert result == expected


@pytest.mark.skipif(not _tiktoken_available(), reason="tiktoken not installed")
def test_openai_tokenizer_matches_tiktoken_multi_message():
    """AC4: count_tokens with multiple messages matches tiktoken sum."""
    import tiktoken

    from services.tokenizer import OpenAITokenizer

    model = "gpt-4o"
    msgs = [
        {"role": "user", "content": "What is the capital of France?"},
        {"role": "assistant", "content": "Paris."},
    ]

    try:
        enc = tiktoken.encoding_for_model(model)
    except KeyError:
        enc = tiktoken.get_encoding("cl100k_base")

    expected = sum(len(enc.encode(m["content"])) for m in msgs if isinstance(m["content"], str))
    t = OpenAITokenizer()
    result = t.count_tokens(msgs, model)
    assert result == expected


@pytest.mark.skipif(not _tiktoken_available(), reason="tiktoken not installed")
def test_openai_tokenizer_no_import_error_at_module_load():
    """AC3: importing OpenAITokenizer never raises ImportError."""
    # Already implicitly tested, but explicit is better
    import importlib

    mod = importlib.import_module("services.tokenizer")
    assert hasattr(mod, "OpenAITokenizer")


# ---------------------------------------------------------------------------
# AC5: fallback when tiktoken is absent — warn exactly once per process lifetime
# ---------------------------------------------------------------------------


def test_openai_tokenizer_fallback_returns_int_when_tiktoken_absent(caplog):
    """AC5: when tiktoken is absent, count_tokens still returns a positive int."""
    import services.tokenizer as tok_mod

    saved = tok_mod._tiktoken_warn_emitted
    tok_mod._tiktoken_warn_emitted = False
    try:
        with patch.dict(sys.modules, {"tiktoken": None}):
            from services.tokenizer import OpenAITokenizer

            t = OpenAITokenizer()
            with caplog.at_level(logging.WARNING, logger="services.tokenizer"):
                result = t.count_tokens(_KNOWN_MESSAGES, _KNOWN_MODEL)
        assert isinstance(result, int) and result >= 1
    finally:
        tok_mod._tiktoken_warn_emitted = saved


def test_openai_tokenizer_warns_exactly_once_when_tiktoken_absent(caplog):
    """AC5: exactly one WARN-level log message is emitted per process lifetime."""
    import services.tokenizer as tok_mod

    saved = tok_mod._tiktoken_warn_emitted
    tok_mod._tiktoken_warn_emitted = False
    try:
        with patch.dict(sys.modules, {"tiktoken": None}):
            from services.tokenizer import OpenAITokenizer

            t = OpenAITokenizer()
            with caplog.at_level(logging.WARNING, logger="services.tokenizer"):
                t.count_tokens(_KNOWN_MESSAGES, _KNOWN_MODEL)
                t.count_tokens(_KNOWN_MESSAGES, _KNOWN_MODEL)
                t.count_tokens(_KNOWN_MESSAGES, _KNOWN_MODEL)

        warn_records = [r for r in caplog.records if r.levelno == logging.WARNING]
        assert len(warn_records) == 1, (
            f"Expected exactly 1 warning, got {len(warn_records)}: {[r.message for r in warn_records]}"
        )
    finally:
        tok_mod._tiktoken_warn_emitted = saved


def test_openai_tokenizer_no_warn_after_flag_set(caplog):
    """AC5: no warning emitted when the warn flag is already set (simulates second process call)."""
    import services.tokenizer as tok_mod

    saved = tok_mod._tiktoken_warn_emitted
    tok_mod._tiktoken_warn_emitted = True  # pre-set as if already warned
    try:
        with patch.dict(sys.modules, {"tiktoken": None}):
            from services.tokenizer import OpenAITokenizer

            t = OpenAITokenizer()
            with caplog.at_level(logging.WARNING, logger="services.tokenizer"):
                t.count_tokens(_KNOWN_MESSAGES, _KNOWN_MODEL)

        warn_records = [r for r in caplog.records if r.levelno == logging.WARNING]
        assert len(warn_records) == 0
    finally:
        tok_mod._tiktoken_warn_emitted = saved


# ---------------------------------------------------------------------------
# AC6: unknown model falls back gracefully (no exception)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _tiktoken_available(), reason="tiktoken not installed")
def test_openai_tokenizer_unknown_model_no_exception():
    """AC6: unknown model name falls back to safe default, no exception raised."""
    from services.tokenizer import OpenAITokenizer

    t = OpenAITokenizer()
    msgs = [{"role": "user", "content": "test content"}]
    result = t.count_tokens(msgs, "my-custom-model-v99")
    assert isinstance(result, int) and result >= 1


def test_openai_tokenizer_unknown_model_fallback_when_tiktoken_absent():
    """AC6: unknown model with tiktoken absent — still returns count, no exception."""
    import services.tokenizer as tok_mod

    saved = tok_mod._tiktoken_warn_emitted
    tok_mod._tiktoken_warn_emitted = False
    try:
        with patch.dict(sys.modules, {"tiktoken": None}):
            from services.tokenizer import OpenAITokenizer

            t = OpenAITokenizer()
            result = t.count_tokens([{"role": "user", "content": "test"}], "my-custom-model-v99")
        assert isinstance(result, int) and result >= 1
    finally:
        tok_mod._tiktoken_warn_emitted = saved


# ---------------------------------------------------------------------------
# AC2: config.toml tokenizer field
# ---------------------------------------------------------------------------


def _write_toml(content: str) -> Path:
    fd, path = tempfile.mkstemp(suffix=".toml")
    os.close(fd)
    Path(path).write_text(content)
    return Path(path)


def test_profile_config_tokenizer_defaults_to_heuristic():
    """AC2/AC9: tokenizer defaults to 'heuristic' when absent in config.toml."""
    from profiles import load_config

    tmp = _write_toml("""
[profiles.default]
kind = "passthrough"
upstream = "http://localhost"
""")
    try:
        config = load_config(tmp)
        assert config.profiles["default"].tokenizer == "heuristic"
    finally:
        tmp.unlink()


def test_profile_config_tokenizer_openai_parsed():
    """AC2: tokenizer = 'openai' is parsed correctly from config.toml."""
    from profiles import load_config

    tmp = _write_toml("""
[profiles.myprofile]
kind = "openai"
upstream = "https://api.openai.com/v1"
tokenizer = "openai"
""")
    try:
        config = load_config(tmp)
        assert config.profiles["myprofile"].tokenizer == "openai"
    finally:
        tmp.unlink()


def test_profile_config_tokenizer_heuristic_explicit():
    """AC2: tokenizer = 'heuristic' is parsed and preserved."""
    from profiles import load_config

    tmp = _write_toml("""
[profiles.myprofile]
kind = "passthrough"
upstream = "http://localhost"
tokenizer = "heuristic"
""")
    try:
        config = load_config(tmp)
        assert config.profiles["myprofile"].tokenizer == "heuristic"
    finally:
        tmp.unlink()


def test_profile_registry_get_tokenizer_returns_heuristic_for_default():
    """AC2: ProfileRegistry.get_tokenizer returns HeuristicTokenizer for default profile."""
    from profiles import ProfileConfig, ProfileRegistry, ProxyConfig, ServerConfig
    from services.tokenizer import HeuristicTokenizer

    config = ProxyConfig(
        server=ServerConfig(),
        profiles={
            "default": ProfileConfig(kind="passthrough", upstream="http://localhost"),
        },
    )
    registry = ProfileRegistry(config)
    t = registry.get_tokenizer("default")
    assert isinstance(t, HeuristicTokenizer)


def test_profile_registry_get_tokenizer_returns_openai_when_configured():
    """AC2: ProfileRegistry.get_tokenizer returns OpenAITokenizer when tokenizer='openai'."""
    from profiles import ProfileConfig, ProfileRegistry, ProxyConfig, ServerConfig
    from services.tokenizer import OpenAITokenizer

    config = ProxyConfig(
        server=ServerConfig(),
        profiles={
            "myprofile": ProfileConfig(
                kind="openai",
                upstream="https://api.openai.com/v1",
                tokenizer="openai",
            ),
        },
    )
    registry = ProfileRegistry(config)
    t = registry.get_tokenizer("myprofile")
    assert isinstance(t, OpenAITokenizer)


# ---------------------------------------------------------------------------
# AC9: no existing behavior changes when tokenizer not set
# ---------------------------------------------------------------------------


def test_heuristic_tokenizer_matches_existing_count_input_tokens():
    """AC9: HeuristicTokenizer.count_tokens matches existing count_input_tokens for same input."""
    from services.cost_accounting import count_input_tokens
    from services.tokenizer import HeuristicTokenizer

    msgs = [{"role": "user", "content": "Hello world test message"}]
    body = {"messages": msgs}

    t = HeuristicTokenizer()
    tokenizer_result = t.count_tokens(msgs, "claude-3-opus")
    existing_result = count_input_tokens(body)
    assert tokenizer_result == existing_result


def test_no_tiktoken_import_when_heuristic_is_default(monkeypatch):
    """AC9: tiktoken is never imported when tokenizer defaults to heuristic."""
    import services.tokenizer as tok_mod
    from services.tokenizer import get_tokenizer

    imported = []
    real_import = __builtins__.__import__ if hasattr(__builtins__, "__import__") else None

    # Verify that get_tokenizer("heuristic") does not touch tiktoken
    with patch.dict(sys.modules, {"tiktoken": None}):
        t = get_tokenizer("heuristic")
        result = t.count_tokens(_KNOWN_MESSAGES, _KNOWN_MODEL)

    assert isinstance(result, int) and result >= 1
