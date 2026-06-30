import json
import os
import tomllib
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class ServerConfig:
    host: str = "127.0.0.1"
    port: int = 8788


@dataclass
class ProfileConfig:
    kind: str  # "passthrough" | "openai"
    upstream: str
    api_key_env: str | None = None
    model: str | None = None
    model_map: dict[str, str] = field(default_factory=dict)


@dataclass
class ProxyConfig:
    server: ServerConfig = field(default_factory=ServerConfig)
    profiles: dict[str, ProfileConfig] = field(default_factory=dict)


class ProfileRegistry:
    """Resolves named profiles into runtime connection values.

    api_key is read from os.environ at call time — never stored on the object.
    """

    def __init__(self, config: ProxyConfig) -> None:
        self._config = config

    def resolve(
        self, name: str
    ) -> tuple[str, str, str | None, str | None, dict[str, str]]:
        """Return (kind, upstream, api_key, model, model_map).

        api_key is looked up from os.environ[api_key_env] on every call.
        """
        profile = self._config.profiles[name]
        api_key = os.environ.get(profile.api_key_env) if profile.api_key_env else None
        return (profile.kind, profile.upstream, api_key, profile.model, profile.model_map)


def load_config(path: Path) -> ProxyConfig:
    """Parse a config.toml file into a ProxyConfig."""
    with open(path, "rb") as f:
        raw = tomllib.load(f)

    server_raw = raw.get("server", {})
    server = ServerConfig(
        host=server_raw.get("host", "127.0.0.1"),
        port=int(server_raw.get("port", 8788)),
    )

    profiles: dict[str, ProfileConfig] = {}
    for name, p in raw.get("profiles", {}).items():
        profiles[name] = ProfileConfig(
            kind=p["kind"],
            upstream=p["upstream"],
            api_key_env=p.get("api_key_env"),
            model=p.get("model"),
            model_map=dict(p.get("model_map", {})),
        )

    return ProxyConfig(server=server, profiles=profiles)


def _make_legacy_config() -> ProxyConfig:
    """Build a ProxyConfig from M1-era env vars (CCPROXY_PROFILE, OPENAI_*)."""
    server = ServerConfig(
        host=os.getenv("CCPROXY_HOST", "127.0.0.1"),
        port=int(os.getenv("CCPROXY_PORT", "8788")),
    )
    profiles: dict[str, ProfileConfig] = {
        "anthropic": ProfileConfig(
            kind="passthrough",
            upstream=os.getenv("UPSTREAM_BASE_URL", "https://api.anthropic.com"),
        ),
        "openai": ProfileConfig(
            kind="openai",
            upstream=os.getenv("OPENAI_BASE_URL", ""),
            api_key_env="OPENAI_API_KEY",
            model=os.getenv("OPENAI_MODEL", "gpt-4o"),
        ),
    }
    return ProxyConfig(server=server, profiles=profiles)


_state_json_path: Path = Path.home() / ".config" / "ccswitch" / "state.json"


def _read_state_default(path: Path) -> str | None:
    """Read active profile from ccswitch state.json; return None on any error.

    Reads the 'active' key written by ccswitch; falls back to 'active_profile'
    for backwards compatibility with prior state files.
    """
    try:
        with open(path) as f:
            data = json.load(f)
        return data.get("active") or data.get("active_profile") or None
    except (FileNotFoundError, json.JSONDecodeError, TypeError, OSError):
        return None


def resolve_profile_name(
    header: str | None,
    query_param: str | None,
    state_json_path: Path | None = None,
) -> str:
    """Resolve profile name using 4-level precedence.

    1. X-CCProxy-Profile header
    2. ?profile= query param
    3. CCPROXY_PROFILE environment variable
    4. active profile from ~/.config/ccswitch/state.json (written by ccswitch)
    5. Built-in 'anthropic' default
    """
    if header:
        return header
    if query_param:
        return query_param
    env_profile = os.getenv("CCPROXY_PROFILE")
    if env_profile:
        return env_profile
    path = state_json_path if state_json_path is not None else _state_json_path
    state_default = _read_state_default(path)
    if state_default:
        return state_default
    return "anthropic"


def get_or_load_config(
    config_path: Path = Path("config.toml"),
) -> tuple[ProxyConfig, bool]:
    """Load config.toml if present; otherwise fall back to M1 env-var behavior.

    Returns (ProxyConfig, from_file) where from_file is True when config.toml
    was found and parsed.
    """
    if config_path.exists():
        return load_config(config_path), True
    return _make_legacy_config(), False
