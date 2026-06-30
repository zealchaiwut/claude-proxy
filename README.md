# claude-proxy

A lightweight HTTP proxy for the Anthropic API, designed for use with Claude Code and other Anthropic API clients.

## Install

Install globally as a command-line tool using `uv` or `pipx`:

```bash
# with uv (recommended)
uv tool install .

# with pipx
pipx install .
```

After install, both `claude-proxy` and `ccswitch` are on your `$PATH`.

## Quick Start

```bash
# 1. Copy the example config
cp config.example.toml config.toml

# 2. Set your API key
export ANTHROPIC_API_KEY=<your-key>

# 3. Start the proxy
claude-proxy
```

The server binds to the `host` and `port` defined in `config.toml` (`127.0.0.1:8788` by default).

### First-run verification

```bash
curl http://127.0.0.1:8788/health
# {"status":"ok","upstream":"https://api.anthropic.com"}
```

### Environment variables

All variables are optional — `config.toml` takes precedence for `[server]` settings.
Copy `.env.example` to `.env` for a template:

```bash
cp .env.example .env
```

See the **Configuration** table below for the full list of recognised variables.

### Development install

```bash
pip install -e ".[dev]"
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

**Tool call support:** Anthropic `tools` and `tool_choice` are translated to OpenAI function-calling format. Responses containing OpenAI `tool_calls` are translated back to Anthropic `tool_use` blocks. Multi-turn conversations with `tool_result` turns are fully supported.

**XML tool mode (`CCPROXY_TOOL_MODE=xml`):** For OpenAI-compatible upstreams that do not support native function calling, set `CCPROXY_TOOL_MODE=xml`. The proxy will inject tool definitions into the system prompt as an XML specification and parse `<tool_call>` blocks from the upstream response back into Anthropic `tool_use` blocks.

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
| `CCPROXY_TOOL_MODE` | `native` | Tool call mode in OpenAI mode: `native` (function calling) or `xml` (XML prompt injection) |
| `CCPROXY_LOG_FILE` | `~/.local/state/claude-proxy/requests.jsonl` | Path for the per-request JSONL log file; parent directories are created automatically |
| `METRICS_WINDOW_SECONDS` | _(all-time)_ | If set, `GET /metrics` only includes samples from the last N seconds |

## Proxied Endpoints

| Endpoint | Behaviour |
|---|---|
| `GET /health` | Proxy health check — not forwarded upstream |
| `GET /metrics` | Per-profile rolling metrics summary (request count, token totals, latency percentiles) |
| `POST /v1/messages` | Passthrough (Anthropic mode) or translated (OpenAI mode) |
| `POST /v1/messages/count_tokens` | Passthrough (Anthropic mode) or heuristic estimate (OpenAI mode) |
| `GET /v1/models` | Transparent passthrough |
| `* /v1/{any}` | All other `/v1/` paths forwarded upstream |

## Error Responses

| Condition | Status | Body |
|---|---|---|
| Upstream unreachable (TCP failure) | `502` | `{"error": "bad_gateway", "message": "upstream unreachable"}` |
| Upstream timeout | `504` | `{"error": "gateway_timeout", "message": "upstream timed out"}` |

## Using claude-proxy with Commander

Commander dispatches multiple concurrent agent subprocesses (coder, tester, estimator, docs-only). Each subprocess can target a different backend through the same proxy instance — a cheap/local backend for low-cost agents and the full Anthropic subscription backend for the coder — without any global state race.

### Setup

1. Start the proxy with a `config.toml` that defines the profiles you need:

```toml
[profiles.anthropic]
kind = "passthrough"
upstream = "https://api.anthropic.com"

[profiles.cheap]
kind = "openai"
upstream = "http://localhost:11434/v1"   # e.g. Ollama
api_key_env = "OLLAMA_API_KEY"
model = "llama3.2"

[profiles.cheap.model_map]
"claude-haiku-4-5-20251001" = "llama3.2"
"claude-sonnet-4-6" = "llama3.2"
```

2. In your Commander configuration, point every agent at the proxy and assign a profile per agent role via the `X-CCProxy-Profile` request header or the `CCPROXY_PROFILE` environment variable:

```bash
# Proxy URL for all agents
export ANTHROPIC_BASE_URL=http://localhost:8788

# Per-subprocess profile selection (precedence: header > env > state.json > default)
#
# Coder subprocess — full Anthropic subscription backend:
CCPROXY_PROFILE=anthropic claude ...
#
# Tester / Estimator / Docs subprocess — cheap/local backend:
CCPROXY_PROFILE=cheap claude ...
```

When two subprocesses run concurrently with different `CCPROXY_PROFILE` values, requests from each subprocess are routed to their respective backends independently — there is no shared mutable state in the proxy hot path, so profiles never bleed across subprocesses.

### Profile selection precedence

| Priority | Mechanism | Scope |
|----------|-----------|-------|
| 1 (highest) | `X-CCProxy-Profile` request header | single request |
| 2 | `CCPROXY_PROFILE` env var (subprocess environment) | all requests from that process |
| 3 | `active_profile` in `state.json` | global default (dashboard-managed) |
| 4 (lowest) | Built-in `anthropic` | fallback |

### model_map rewriting

When the resolved profile includes a `model_map`, the client's requested model name is rewritten to the upstream model string before the request is forwarded. This lets Claude Code send its native model names (e.g. `claude-haiku-4-5-20251001`) while the proxy transparently maps them to whatever the target backend expects.

## Request Logging

Every proxied request through `POST /v1/messages` produces one JSONL record written to `~/.local/state/claude-proxy/requests.jsonl` (override with `CCPROXY_LOG_FILE`). The parent directory is created automatically. Each line is a complete JSON object:

| Field | Type | Description |
|-------|------|-------------|
| `request_id` | string (UUID) | Unique identifier for this request |
| `timestamp` | ISO-8601 string | UTC time the record was emitted |
| `profile_name` | string | Profile used (`anthropic`, `openai`, etc.) |
| `profile_kind` | string | `passthrough` or `openai` |
| `requested_model` | string | Model name the client sent |
| `upstream_model` | string | Model name forwarded upstream (after `model_map` rewrite) |
| `upstream_host` | string | Hostname of the upstream API |
| `method` | string | HTTP method (`POST`) |
| `path` | string | Request path (`/v1/messages`) |
| `status` | integer | HTTP status code from upstream |
| `latency_ms` | float | Total round-trip latency in milliseconds |
| `streamed` | boolean | Whether the response was streamed |
| `run_id` | string\|null | Value of `X-CCProxy-Run` header, or null |
| `role` | string\|null | Value of `X-CCProxy-Role` header, or null |
| `ticket` | string\|null | Value of `X-CCProxy-Ticket` header, or null |

No request body, response body, or API keys are written to the log.

## Metrics

`GET /metrics` returns a snapshot of in-memory per-profile statistics accumulated since the proxy started (or since `METRICS_WINDOW_SECONDS` ago, if that env var is set):

```json
{
  "profiles": {
    "anthropic": {
      "request_count": 42,
      "error_count": 1,
      "total_input_tokens": 18340,
      "total_output_tokens": 6120,
      "total_est_cost_usd": 0.0469,
      "p50_latency_ms": 312.4,
      "p95_latency_ms": 891.2
    }
  }
}
```

Token totals and cost are only accumulated for successful (2xx) responses. The metrics collector is in-memory only — it never reads the JSONL log file.

## Correlation Headers

Three optional request headers let callers tag each request with Commander dispatch metadata. The proxy reads them, stores their values on the request record, and ignores them for routing or profile selection.

| Header | Record field | Purpose |
|--------|-------------|---------|
| `X-CCProxy-Run` | `run_id` | Identifies the Commander run (e.g. sprint ID or UUID) that spawned this subprocess |
| `X-CCProxy-Role` | `role` | The agent role (e.g. `coder`, `tester`, `estimator`) assigned to this subprocess |
| `X-CCProxy-Ticket` | `ticket` | The issue or ticket reference being worked (e.g. `PROJ-42`) |

When a header is present its value is stored as a string. When a header is absent the field is `null`. No combination of header values affects profile selection or upstream routing.

### Injecting headers in Commander

Pass the headers via `subprocess.Popen` using the `env` parameter or an explicit header dict in the launcher. Environment-variable forwarding is the simplest approach because `claude` (the CLI) sets `ANTHROPIC_BASE_URL` but does not automatically forward custom headers — the launcher must inject them explicitly:

```python
# In Commander's agent launcher (e.g. services/dispatcher.py)
import subprocess, os

env = {**os.environ, "ANTHROPIC_BASE_URL": "http://localhost:8788"}

proc = subprocess.Popen(
    ["claude", "--profile", "coder", ...],
    env=env,
    # Commander wraps the claude CLI with a thin HTTP shim that injects headers:
    # X-CCProxy-Run: <run_id>
    # X-CCProxy-Role: coder
    # X-CCProxy-Ticket: PROJ-42
)
```

Alternatively, if your launcher speaks HTTP directly (e.g. via `httpx`):

```python
headers = {
    "X-CCProxy-Run": run_id,
    "X-CCProxy-Role": "coder",
    "X-CCProxy-Ticket": "PROJ-42",
}
response = client.post("http://localhost:8788/v1/messages", headers=headers, json=body)
```

### Sample log record

A request tagged with all three headers produces a JSONL record like:

```json
{
  "request_id": "a1b2c3d4-...",
  "timestamp": "2026-06-30T10:00:00.000000+00:00",
  "profile_name": "anthropic",
  "profile_kind": "passthrough",
  "requested_model": "claude-sonnet-4-6",
  "upstream_model": "claude-sonnet-4-6",
  "upstream_host": "api.anthropic.com",
  "method": "POST",
  "path": "/v1/messages",
  "status": 200,
  "latency_ms": 312.4,
  "streamed": false,
  "run_id": "run-42",
  "role": "coder",
  "ticket": "PROJ-42"
}
```

## Running Tests

```bash
pip install -e ".[dev]"
pytest
```
