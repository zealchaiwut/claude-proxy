from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    upstream_base_url: str = "https://api.anthropic.com"
    upstream_read_timeout: float = 300.0
    # Global default for extended-thinking handling on the OpenAI path.
    # "disabled" | "forward" | "strip". Profiles may override per-profile.
    openai_thinking_mode: str = "disabled"
    ready_cache_ttl: float = 5.0


def get_settings() -> Settings:
    return Settings()
