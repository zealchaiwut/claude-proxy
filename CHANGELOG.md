# Changelog

Per-sprint changelog for claude-proxy. Entries are written by the documentor when a
sprint finishes. Dated per-sprint files live under [docs/changelog/](docs/changelog/).

## Sprint 4

- #22: Map Anthropic tools and tool_choice to OpenAI format
- #23: Translate OpenAI tool_calls to Anthropic tool_use in from_openai_response
- #24: Translate Anthropic tool_result turns into OpenAI tool messages
- #26: Add optional XML tool-call fallback for non-function-calling upstreams

## Sprint 3.1

- #18: Bridge OpenAI stream to Anthropic emitter in translator
- #19: Wire live SSE streaming for OpenAI proxy mode

## Sprint 3

- #16: Add Anthropic SSE streaming event emitter service
- #17: Add OpenAI SSE streaming consumer for chat completions

## Sprint 2

- #9: Add translator service: Anthropic → OpenAI request mapping
- #11: Add OpenAI proxy mode behind CCPROXY_PROFILE env switch

## Sprint 1

- #1: Create claude-proxy FastAPI skeleton with health endpoint
- #2: Add transparent non-streaming passthrough for POST /v1/messages
- #3: Add streaming passthrough for POST /v1/messages
- #4: Extend proxy passthrough and harden upstream error handling
