from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    upstream_base_url: str = "https://api.anthropic.com"
    upstream_read_timeout: float = 300.0
    ready_cache_ttl: float = 5.0


def get_settings() -> Settings:
    return Settings()
