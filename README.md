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

## Proxy Modes

The proxy operates in one of two modes, selected by `CCPROXY_PROFILE` (default: `anthropic`).

### Anthropic mode (default)

Passes requests byte-for-byte to the configured Anthropic upstream. No translation is performed.

### OpenAI mode

```bash
CCPROXY_PROFILE=openai
OPENAI_BASE_URL=https://api.openai.com/v1
OPENAI_API_KEY=sk-...
OPENAI_MODEL=gpt-4o   # optional, default: gpt-4o
```

Accepts standard Anthropic `POST /v1/messages` requests, translates them to OpenAI `/chat/completions` format, and translates the response back. Lets Claude Code clients target any OpenAI-compatible backend without client-side changes.

Streaming (`stream: true`) is fully supported — OpenAI SSE chunks are translated to Anthropic SSE events on-the-fly and streamed back to the client as they arrive.

## Configuration

| Variable | Default | Description |
|---|---|---|
| `UPSTREAM_BASE_URL` | `https://api.anthropic.com` | Upstream Anthropic API base URL (Anthropic mode) |
| `UPSTREAM_READ_TIMEOUT` | `300.0` | Seconds to wait for upstream response |
| `CCPROXY_HOST` | `127.0.0.1` | Host to bind the proxy server |
| `CCPROXY_PORT` | `8788` | Port to bind the proxy server |
| `CCPROXY_PROFILE` | `anthropic` | Proxy mode: `anthropic` or `openai` |
| `OPENAI_BASE_URL` | — | OpenAI-compatible API base URL (OpenAI mode) |
| `OPENAI_API_KEY` | — | API key for OpenAI-compatible API (OpenAI mode) |
| `OPENAI_MODEL` | `gpt-4o` | Model name sent upstream (OpenAI mode) |

## Proxied Endpoints

| Endpoint | Behaviour |
|---|---|
| `GET /health` | Proxy health check — not forwarded upstream |
| `POST /v1/messages` | Passthrough (Anthropic mode) or translated (OpenAI mode) |
| `POST /v1/messages/count_tokens` | Passthrough (Anthropic mode) or heuristic estimate (OpenAI mode) |
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
