# M6 — Unattended robustness

**Date:** 2026-06-30
**Sprint label:** NEW
**Default labels:** enhancement, backend
**Status:** drafted

Commander runs sprints unattended, so a flaky upstream must degrade cleanly, not
crash a dispatch with a 500 traceback. M6 adds per-profile timeouts, bounded
retry on transient failures, and Anthropic-shaped error responses — and
deliberately refuses silent cross-provider failover, because an agent quietly
switching models mid-edit is a debugging nightmare. Fail fast, fail loud, name
the profile. Depends on M0–M4; pairs with M5 (failures get logged).

## Prompts

Paste one code block into the Bulk Create textarea. Prompts are `---`-separated.

```
Add per-profile timeout configuration. Support a [profiles.<name>.timeouts] table in config.toml (connect, read, write, pool — seconds) and apply it to the httpx client used for that profile, falling back to a sane global default (connect 10s, read = UPSTREAM_READ_TIMEOUT, write 30s, pool 10s) when unset. Streaming reads should use the read timeout per-chunk, not for the whole stream. Acceptance: (1) a profile with custom timeouts uses them; (2) an unset profile uses the documented defaults; (3) a slow upstream that exceeds the connect/read timeout raises a timeout that the error layer (next ticket) can shape, rather than hanging; (4) pytest covers config application and a simulated read timeout against a stub upstream.
---
Add bounded retry with backoff for transient upstream failures on NON-streaming requests. Retry on connection errors and HTTP 429/500/502/503/504 up to a small cap (default 2 retries, configurable per profile via [profiles.<name>.retry] max_attempts) with exponential backoff + jitter, honoring a Retry-After header when present. Do NOT retry 4xx other than 429, and do NOT retry a streaming response once any byte has reached the client (only a pre-first-byte streaming failure may retry). Never retry across a different profile/provider. Acceptance: (1) a stub upstream that fails twice then succeeds yields a success after retries; (2) a 400 is returned immediately without retry; (3) Retry-After is respected; (4) a streaming response that has already emitted bytes is not retried; (5) pytest covers the retry-then-succeed, no-retry-4xx, and retry-cap-exhausted paths.
---
Normalize all failures into the Anthropic error shape. Map upstream errors and proxy-side failures (timeout, connection refused, retry-exhausted, translation error) to an Anthropic-style JSON body {"type":"error","error":{"type":<class>,"message":<safe text>}} with an appropriate status (504 timeout, 502 connection, 4xx/5xx passthrough of upstream status where meaningful). Messages must be safe — no stack traces, no secrets, no raw upstream bodies that might echo a key. For streaming: if failure occurs before the first event, return a normal error response; if after content has started, emit an Anthropic `error` SSE event then stop cleanly. Acceptance: (1) a timeout returns 504 with an Anthropic error body; (2) a connection failure returns 502 likewise; (3) no response body contains a traceback or secret; (4) a mid-stream failure emits an error event and closes without a 500; (5) pytest covers timeout, connection, and mid-stream error shaping.
---
Add an explicit no-silent-failover guard and clear failure surfacing. Assert in code (and in a test) that retry/error handling never swaps the resolved profile or upstream for a different one — a failed dispatch fails on its chosen backend, it does not silently move to another. On exhausted retries, return the shaped error naming the profile (e.g. error message includes "profile=openai upstream=<host>") and log at WARN with the M5 correlation fields so the dashboard shows which dispatch failed where. Acceptance: (1) when a profile's upstream is hard-down, the response error names that profile and does not reach any other configured upstream; (2) the failure is logged at WARN with correlation id/profile; (3) a test proves no cross-profile fallback occurs even with multiple profiles configured; (4) pytest covers the exhausted-retry path end to end.
```

## Notes

- **No silent failover is a feature, not a gap.** The whole point is that a coder
  dispatch that fails on the OpenAI backend fails *visibly* on the OpenAI
  backend — Commander/the human decides what to do, the proxy doesn't guess.
- **Order:** 1 (timeouts) → 2 (retry) → 3 (error shape) → 4 (guard + surfacing).
  3 is what every other ticket's failures flow through.
- **Streaming retry is restricted on purpose** — once bytes are out, a retry would
  duplicate/garble the stream. Only pre-first-byte retries are safe.
- **Pairs with M5:** every shaped failure carries the correlation fields so it
  shows up in logs/metrics.
- **Headless project.** Design Refs from DESIGN.md "Intent"; UAT via forcing a
  down/slow stub upstream and observing clean errors, MANUAL.

## Posted issues

| # | Title | Size |
|---|-------|------|
| _pending_ | | |
