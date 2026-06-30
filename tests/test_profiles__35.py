"""Tests for issue #35: profiles config system via config.toml."""
import textwrap


from profiles import (
    ProfileConfig,
    ProfileRegistry,
    ProxyConfig,
    get_or_load_config,
    load_config,
)


# --- Config loading ---

def test_load_config_server_defaults(tmp_path):
    """[server] with no values uses host=127.0.0.1 and port=8788 defaults."""
    toml = tmp_path / "config.toml"
    toml.write_text("[server]\n")
    cfg = load_config(toml)
    assert cfg.server.host == "127.0.0.1"
    assert cfg.server.port == 8788


def test_load_config_server_values(tmp_path):
    """[server] table correctly sets host and port."""
    toml = tmp_path / "config.toml"
    toml.write_text('[server]\nhost = "0.0.0.0"\nport = 9000\n')
    cfg = load_config(toml)
    assert cfg.server.host == "0.0.0.0"
    assert cfg.server.port == 9000


def test_load_config_passthrough_profile(tmp_path):
    """A passthrough profile loads correctly."""
    toml = tmp_path / "config.toml"
    toml.write_text(textwrap.dedent("""\
        [profiles.anthropic]
        kind = "passthrough"
        upstream = "https://api.anthropic.com"
    """))
    cfg = load_config(toml)
    assert "anthropic" in cfg.profiles
    p = cfg.profiles["anthropic"]
    assert p.kind == "passthrough"
    assert p.upstream == "https://api.anthropic.com"
    assert p.api_key_env is None
    assert p.model is None
    assert p.model_map == {}


def test_load_config_openai_profile_with_model_map(tmp_path):
    """An openai profile with api_key_env, model, and model_map loads correctly."""
    toml = tmp_path / "config.toml"
    toml.write_text(textwrap.dedent("""\
        [profiles.openai]
        kind = "openai"
        upstream = "https://api.openai.com/v1"
        api_key_env = "OPENAI_API_KEY"
        model = "gpt-4o"

        [profiles.openai.model_map]
        "claude-3-5-sonnet-20241022" = "gpt-4o"
    """))
    cfg = load_config(toml)
    p = cfg.profiles["openai"]
    assert p.kind == "openai"
    assert p.upstream == "https://api.openai.com/v1"
    assert p.api_key_env == "OPENAI_API_KEY"
    assert p.model == "gpt-4o"
    assert p.model_map == {"claude-3-5-sonnet-20241022": "gpt-4o"}


def test_load_config_both_profiles_no_error(tmp_path):
    """A config with one passthrough and one openai profile loads without error."""
    toml = tmp_path / "config.toml"
    toml.write_text(textwrap.dedent("""\
        [server]
        port = 8788

        [profiles.anthropic]
        kind = "passthrough"
        upstream = "https://api.anthropic.com"

        [profiles.openai]
        kind = "openai"
        upstream = "https://api.openai.com/v1"
        api_key_env = "OPENAI_API_KEY"
        model = "gpt-4o"

        [profiles.openai.model_map]
        "claude-3-5-sonnet-20241022" = "gpt-4o"
    """))
    cfg = load_config(toml)
    assert "anthropic" in cfg.profiles
    assert "openai" in cfg.profiles
    assert cfg.server.port == 8788
    assert cfg.profiles["anthropic"].kind == "passthrough"
    assert cfg.profiles["openai"].kind == "openai"


# --- Profile resolution ---

def test_resolve_passthrough_profile():
    """resolve() returns correct 5-tuple for a passthrough profile."""
    cfg = ProxyConfig(
        profiles={
            "anthropic": ProfileConfig(
                kind="passthrough",
                upstream="https://api.anthropic.com",
            )
        }
    )
    registry = ProfileRegistry(cfg)
    kind, upstream, api_key, model, model_map = registry.resolve("anthropic")
    assert kind == "passthrough"
    assert upstream == "https://api.anthropic.com"
    assert api_key is None
    assert model is None
    assert model_map == {}


def test_resolve_openai_profile_reads_api_key(monkeypatch):
    """resolve() returns api_key read from the named env var."""
    monkeypatch.setenv("OPENAI_API_KEY", "test-secret")
    cfg = ProxyConfig(
        profiles={
            "openai": ProfileConfig(
                kind="openai",
                upstream="https://api.openai.com/v1",
                api_key_env="OPENAI_API_KEY",
                model="gpt-4o",
                model_map={"claude-3-5-sonnet-20241022": "gpt-4o"},
            )
        }
    )
    registry = ProfileRegistry(cfg)
    kind, upstream, api_key, model, model_map = registry.resolve("openai")
    assert kind == "openai"
    assert upstream == "https://api.openai.com/v1"
    assert api_key == "test-secret"
    assert model == "gpt-4o"
    assert model_map == {"claude-3-5-sonnet-20241022": "gpt-4o"}


def test_env_secret_read_at_call_time_not_load_time(monkeypatch):
    """api_key is NOT cached at registry creation; it is read from env at each resolve() call."""
    cfg = ProxyConfig(
        profiles={
            "openai": ProfileConfig(
                kind="openai",
                upstream="https://api.openai.com/v1",
                api_key_env="OPENAI_API_KEY",
                model="gpt-4o",
            )
        }
    )
    # Create registry while the env var is absent
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    registry = ProfileRegistry(cfg)

    # Set the key after registry creation
    monkeypatch.setenv("OPENAI_API_KEY", "late-secret")
    _, _, api_key, _, _ = registry.resolve("openai")
    assert api_key == "late-secret"


def test_env_secret_missing_returns_none(monkeypatch):
    """resolve() returns None for api_key when the named env var is not set."""
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    cfg = ProxyConfig(
        profiles={
            "openai": ProfileConfig(
                kind="openai",
                upstream="https://api.openai.com/v1",
                api_key_env="OPENAI_API_KEY",
                model="gpt-4o",
            )
        }
    )
    registry = ProfileRegistry(cfg)
    _, _, api_key, _, _ = registry.resolve("openai")
    assert api_key is None


# --- model_map translation ---

def test_model_map_translates_known_model(monkeypatch):
    """model_map returned by resolve() maps a client model string to the upstream model."""
    monkeypatch.setenv("OPENAI_API_KEY", "k")
    cfg = ProxyConfig(
        profiles={
            "openai": ProfileConfig(
                kind="openai",
                upstream="https://api.openai.com/v1",
                api_key_env="OPENAI_API_KEY",
                model="gpt-4o",
                model_map={"claude-3-5-sonnet-20241022": "gpt-4o"},
            )
        }
    )
    registry = ProfileRegistry(cfg)
    _, _, _, _, model_map = registry.resolve("openai")
    client_model = "claude-3-5-sonnet-20241022"
    upstream_model = model_map.get(client_model, client_model)
    assert upstream_model == "gpt-4o"


def test_model_map_passes_through_unknown_model(monkeypatch):
    """model_map returns the original model string when no mapping is configured."""
    monkeypatch.setenv("OPENAI_API_KEY", "k")
    cfg = ProxyConfig(
        profiles={
            "openai": ProfileConfig(
                kind="openai",
                upstream="https://api.openai.com/v1",
                api_key_env="OPENAI_API_KEY",
                model="gpt-4o",
                model_map={"claude-3-5-sonnet-20241022": "gpt-4o"},
            )
        }
    )
    registry = ProfileRegistry(cfg)
    _, _, _, _, model_map = registry.resolve("openai")
    unknown = "claude-3-opus-20240229"
    assert model_map.get(unknown, unknown) == unknown


# --- Back-compat fallback ---

def test_fallback_when_no_config_toml(tmp_path, monkeypatch):
    """get_or_load_config() falls back to legacy env-var behavior when config.toml is absent."""
    monkeypatch.setenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
    monkeypatch.setenv("OPENAI_MODEL", "gpt-4o-mini")

    absent = tmp_path / "config.toml"  # does not exist
    cfg, from_file = get_or_load_config(absent)

    assert from_file is False
    assert "anthropic" in cfg.profiles
    assert cfg.profiles["anthropic"].kind == "passthrough"
    assert "openai" in cfg.profiles
    assert cfg.profiles["openai"].api_key_env == "OPENAI_API_KEY"
    assert cfg.profiles["openai"].model == "gpt-4o-mini"


def test_fallback_no_error_on_missing_file(tmp_path):
    """get_or_load_config() does not raise when config.toml is absent."""
    absent = tmp_path / "config.toml"
    cfg, from_file = get_or_load_config(absent)
    assert from_file is False
    assert isinstance(cfg, ProxyConfig)


def test_fallback_registry_resolves_env_vars(tmp_path, monkeypatch):
    """In fallback mode, ProfileRegistry.resolve reads OPENAI_API_KEY from env correctly."""
    monkeypatch.setenv("OPENAI_API_KEY", "legacy-secret")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://api.openai.com/v1")

    absent = tmp_path / "config.toml"
    cfg, _ = get_or_load_config(absent)
    registry = ProfileRegistry(cfg)
    _, _, api_key, _, _ = registry.resolve("openai")
    assert api_key == "legacy-secret"


def test_fallback_uses_ccproxy_host_port(tmp_path, monkeypatch):
    """In fallback mode, server.host and server.port come from CCPROXY_HOST/CCPROXY_PORT."""
    monkeypatch.setenv("CCPROXY_HOST", "0.0.0.0")
    monkeypatch.setenv("CCPROXY_PORT", "9999")

    absent = tmp_path / "config.toml"
    cfg, _ = get_or_load_config(absent)
    assert cfg.server.host == "0.0.0.0"
    assert cfg.server.port == 9999
