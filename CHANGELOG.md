# Changelog

Per-sprint changelog for claude-proxy. Entries are written by the documentor when a
sprint finishes. Dated per-sprint files live under [docs/changelog/](docs/changelog/).

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
