"""Tests for issue #56: cache_control handling on Anthropic-to-OpenAI translation path."""
from __future__ import annotations

import textwrap

from schemas.anthropic import MessagesRequest
from services.translator import to_openai_request


def _req(**kwargs) -> MessagesRequest:
    defaults = {
        "model": "claude-3-haiku-20240307",
        "max_tokens": 256,
        "messages": [{"role": "user", "content": "hello"}],
    }
    defaults.update(kwargs)
    return MessagesRequest(**defaults)


# --- AC1 + AC6 strip case: system block cache_control removed under prompt_cache="none" ---

def test_strip_cache_control_system_block_none():
    """cache_control on system TextBlock is absent from translated system message when prompt_cache='none'."""
    req = MessagesRequest(
        model="claude-3-haiku-20240307",
        max_tokens=256,
        system=[{"type": "text", "text": "You are helpful.", "cache_control": {"type": "ephemeral"}}],
        messages=[{"role": "user", "content": "hello"}],
    )
    result = to_openai_request(req, model="gpt-4o", prompt_cache="none")
    system_msgs = [m for m in result.messages if m.get("role") == "system"]
    assert len(system_msgs) == 1
    assert "cache_control" not in system_msgs[0]
    assert system_msgs[0]["content"] == "You are helpful."


def test_strip_cache_control_multiple_system_blocks_none():
    """cache_control on multiple system TextBlocks: all markers removed, text concatenated."""
    req = MessagesRequest(
        model="claude-3-haiku-20240307",
        max_tokens=256,
        system=[
            {"type": "text", "text": "Part one.", "cache_control": {"type": "ephemeral"}},
            {"type": "text", "text": " Part two.", "cache_control": {"type": "ephemeral"}},
        ],
        messages=[{"role": "user", "content": "hi"}],
    )
    result = to_openai_request(req, model="gpt-4o", prompt_cache="none")
    system_msgs = [m for m in result.messages if m.get("role") == "system"]
    assert len(system_msgs) == 1
    assert "cache_control" not in system_msgs[0]
    assert system_msgs[0]["content"] == "Part one. Part two."


# --- AC2 + AC6 strip case: tool cache_control removed under prompt_cache="none" ---

def test_strip_cache_control_tool_definition_none():
    """cache_control on tool definition is absent from translated OpenAI tools when prompt_cache='none'."""
    tool_with_cache = {
        "name": "get_weather",
        "description": "Get weather",
        "input_schema": {"type": "object", "properties": {}},
        "cache_control": {"type": "ephemeral"},
    }
    req = _req(tools=[tool_with_cache])
    result = to_openai_request(req, model="gpt-4o", prompt_cache="none")
    assert result.tools is not None
    assert len(result.tools) == 1
    oai_tool = result.tools[0]
    assert "cache_control" not in oai_tool
    assert "cache_control" not in oai_tool.get("function", {})


def test_strip_cache_control_multiple_tools_none():
    """cache_control stripped from every tool definition; non-cache fields intact."""
    tools = [
        {
            "name": "alpha",
            "description": "First",
            "input_schema": {"type": "object"},
            "cache_control": {"type": "ephemeral"},
        },
        {
            "name": "beta",
            "description": "Second",
            "input_schema": {"type": "object"},
            "cache_control": {"type": "ephemeral"},
        },
    ]
    req = _req(tools=tools)
    result = to_openai_request(req, model="gpt-4o", prompt_cache="none")
    assert result.tools is not None
    for oai_tool in result.tools:
        assert "cache_control" not in oai_tool
        assert "cache_control" not in oai_tool.get("function", {})


# --- AC3 + AC6 no-mutation assertion ---

def test_no_mutation_system_content_order_preserved():
    """Stripping cache_control does not reorder system text blocks."""
    req = MessagesRequest(
        model="claude-3-haiku-20240307",
        max_tokens=256,
        system=[
            {"type": "text", "text": "Block one.", "cache_control": {"type": "ephemeral"}},
            {"type": "text", "text": "Block two."},
        ],
        messages=[{"role": "user", "content": "hello"}],
    )
    result = to_openai_request(req, model="gpt-4o", prompt_cache="none")
    system_msgs = [m for m in result.messages if m.get("role") == "system"]
    assert system_msgs[0]["content"] == "Block one.Block two."


def test_no_mutation_tool_order_preserved():
    """Stripping cache_control does not reorder tools or rename non-cache fields."""
    tools = [
        {"name": "alpha", "description": "First tool", "input_schema": {"type": "object"}, "cache_control": {"type": "ephemeral"}},
        {"name": "beta", "description": "Second tool", "input_schema": {"type": "object"}},
    ]
    req = _req(tools=tools)
    result = to_openai_request(req, model="gpt-4o", prompt_cache="none")
    assert result.tools[0]["function"]["name"] == "alpha"
    assert result.tools[0]["function"]["description"] == "First tool"
    assert result.tools[1]["function"]["name"] == "beta"
    assert result.tools[1]["function"]["description"] == "Second tool"


def test_no_mutation_messages_unaffected_by_strip():
    """Message turns are unchanged when stripping cache_control from system/tools."""
    req = MessagesRequest(
        model="claude-3-haiku-20240307",
        max_tokens=256,
        system=[{"type": "text", "text": "Sys.", "cache_control": {"type": "ephemeral"}}],
        messages=[
            {"role": "user", "content": "Question?"},
            {"role": "assistant", "content": "Answer."},
        ],
    )
    result = to_openai_request(req, model="gpt-4o", prompt_cache="none")
    user_msg = next(m for m in result.messages if m.get("role") == "user")
    asst_msg = next(m for m in result.messages if m.get("role") == "assistant")
    assert user_msg["content"] == "Question?"
    assert asst_msg["content"] == "Answer."


# --- AC4 + AC6 map case: prompt_cache="auto" + cache-capable hint carries upstream mechanism ---

def test_auto_cache_capable_hint_adds_cache_control_to_request():
    """prompt_cache='auto' + cache_provider_hint='openai' → ChatRequest carries cache_control field."""
    req = MessagesRequest(
        model="claude-3-haiku-20240307",
        max_tokens=256,
        system=[{"type": "text", "text": "System.", "cache_control": {"type": "ephemeral"}}],
        messages=[{"role": "user", "content": "hi"}],
    )
    result = to_openai_request(req, model="gpt-4o", prompt_cache="auto", cache_provider_hint="openai")
    result_dict = result.model_dump()
    assert result_dict.get("cache_control") == {"type": "ephemeral"}


def test_auto_cache_via_tool_cache_control():
    """prompt_cache='auto' detects cache_control on tools and adds upstream mechanism."""
    tool_with_cache = {
        "name": "search",
        "description": "Search",
        "input_schema": {"type": "object"},
        "cache_control": {"type": "ephemeral"},
    }
    req = _req(tools=[tool_with_cache])
    result = to_openai_request(req, model="gpt-4o", prompt_cache="auto", cache_provider_hint="openai")
    result_dict = result.model_dump()
    assert result_dict.get("cache_control") == {"type": "ephemeral"}


def test_auto_without_capable_hint_no_cache_field():
    """prompt_cache='auto' with no recognized hint does NOT add cache_control to request."""
    req = MessagesRequest(
        model="claude-3-haiku-20240307",
        max_tokens=256,
        system=[{"type": "text", "text": "System.", "cache_control": {"type": "ephemeral"}}],
        messages=[{"role": "user", "content": "hi"}],
    )
    result = to_openai_request(req, model="gpt-4o", prompt_cache="auto", cache_provider_hint=None)
    result_dict = result.model_dump()
    assert "cache_control" not in result_dict


def test_auto_no_cache_control_in_input_no_cache_field():
    """prompt_cache='auto' with no cache_control markers in input: no caching field added."""
    req = _req(system="Plain system", tools=[{"name": "t", "description": "T", "input_schema": {}}])
    result = to_openai_request(req, model="gpt-4o", prompt_cache="auto", cache_provider_hint="openai")
    result_dict = result.model_dump()
    assert "cache_control" not in result_dict


# --- AC5: config.toml profile keys parsed ---

def test_profile_config_parses_prompt_cache_auto(tmp_path):
    """load_config parses prompt_cache='auto' from profile TOML."""
    toml = tmp_path / "config.toml"
    toml.write_text(textwrap.dedent("""\
        [profiles.openai]
        kind = "openai"
        upstream = "https://api.openai.com/v1"
        api_key_env = "OPENAI_API_KEY"
        model = "gpt-4o"
        prompt_cache = "auto"
        cache_provider_hint = "openai"
    """))
    from profiles import load_config
    cfg = load_config(toml)
    p = cfg.profiles["openai"]
    assert p.prompt_cache == "auto"
    assert p.cache_provider_hint == "openai"


def test_profile_config_prompt_cache_defaults_to_none(tmp_path):
    """prompt_cache defaults to 'none' when not specified in TOML profile."""
    toml = tmp_path / "config.toml"
    toml.write_text(textwrap.dedent("""\
        [profiles.openai]
        kind = "openai"
        upstream = "https://api.openai.com/v1"
        model = "gpt-4o"
    """))
    from profiles import load_config
    cfg = load_config(toml)
    p = cfg.profiles["openai"]
    assert p.prompt_cache == "none"
    assert p.cache_provider_hint is None


def test_profile_registry_get_cache_config():
    """ProfileRegistry.get_cache_config returns (prompt_cache, cache_provider_hint)."""
    from profiles import ProfileConfig, ProxyConfig, ProfileRegistry
    cfg = ProxyConfig(
        profiles={
            "cached": ProfileConfig(
                kind="openai",
                upstream="https://api.openai.com/v1",
                prompt_cache="auto",
                cache_provider_hint="openai",
            )
        }
    )
    registry = ProfileRegistry(cfg)
    pc, hint = registry.get_cache_config("cached")
    assert pc == "auto"
    assert hint == "openai"


def test_profile_registry_get_cache_config_defaults_for_missing_profile():
    """get_cache_config returns ('none', None) for an unknown profile name."""
    from profiles import ProxyConfig, ProfileRegistry
    registry = ProfileRegistry(ProxyConfig())
    pc, hint = registry.get_cache_config("nonexistent")
    assert pc == "none"
    assert hint is None


# --- AC7: no regression — existing behavior unchanged when no cache_control present ---

def test_no_regression_plain_request_default_params():
    """to_openai_request with default params produces same result as pre-feature call."""
    req = _req(
        system="You are a helpful assistant.",
        tools=[{"name": "search", "description": "Search", "input_schema": {"type": "object"}}],
    )
    result_default = to_openai_request(req, model="gpt-4o")
    result_none = to_openai_request(req, model="gpt-4o", prompt_cache="none")
    assert result_default.model_dump() == result_none.model_dump()


def test_no_regression_string_system_unaffected():
    """A plain string system prompt passes through unchanged under all cache modes."""
    req = _req(system="Plain text system.")
    result = to_openai_request(req, model="gpt-4o", prompt_cache="none")
    system_msgs = [m for m in result.messages if m.get("role") == "system"]
    assert system_msgs[0]["content"] == "Plain text system."
    assert "cache_control" not in system_msgs[0]


def test_no_regression_no_tools_no_cache_field():
    """Request with no tools and no cache markers: translated request has no cache_control field."""
    req = _req()
    result = to_openai_request(req, model="gpt-4o", prompt_cache="auto", cache_provider_hint="openai")
    assert "cache_control" not in result.model_dump()
