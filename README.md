# claude-proxy

A lightweight HTTP proxy for the Anthropic API, designed for use with Claude Code and other Anthropic API clients.

## Quick Start

```bash
pip install -e .
uvicorn main:app --host 127.0.0.1 --port 8788
```

## Claude Code Integration

To route Claude Code through this proxy, set the following environment variable before launching Claude Code:

```bash
export ANTHROPIC_BASE_URL=http://localhost:8788
```

Your existing `ANTHROPIC_API_KEY` or Anthropic subscription login works **unchanged** — the proxy forwards your credentials to the upstream API without modification. You do not need to generate a new key or alter your authentication setup.

## Configuration

| Variable | Default | Description |
|---|---|---|
| `UPSTREAM_BASE_URL` | `https://api.anthropic.com` | Upstream Anthropic API base URL |
| `UPSTREAM_READ_TIMEOUT` | `300.0` | Seconds to wait for upstream response |
| `CCPROXY_HOST` | `127.0.0.1` | Host to bind the proxy server |
| `CCPROXY_PORT` | `8788` | Port to bind the proxy server |

## Proxied Endpoints

| Endpoint | Behaviour |
|---|---|
| `GET /health` | Proxy health check — not forwarded upstream |
| `POST /v1/messages` | Transparent passthrough |
| `POST /v1/messages/count_tokens` | Transparent passthrough |
| `GET /v1/models` | Transparent passthrough |
| `* /v1/{any}` | All other `/v1/` paths forwarded upstream |

## Error Responses

| Condition | Status | Body |
|---|---|---|
| Upstream unreachable (TCP failure) | `502` | `{"error": "bad_gateway", "message": "upstream unreachable"}` |
| Upstream timeout | `504` | `{"error": "gateway_timeout", "message": "upstream timed out"}` |

## Running Tests

```bash
pip install -e ".[dev]"
pytest
```
