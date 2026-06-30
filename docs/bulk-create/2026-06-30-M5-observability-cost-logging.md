# M5 — Observability & cost logging

**Date:** 2026-06-30
**Sprint label:** NEW
**Default labels:** enhancement, backend
**Status:** drafted

The moment real work hits a non-subscription backend you are flying blind: you
can't tell what the OpenAI path costs or which backend served a given ticket.
M5 makes every request observable — structured per-request logs (profile, model,
upstream, latency, tokens, estimated cost) with a correlation id that ties a line
back to a Commander dispatch, plus a small summary endpoint. The proxy stays a
dumb translator; this only *records* what it already does. Depends on M0–M4.

## Prompts

Paste one code block into the Bulk Create textarea. Prompts are `---`-separated.

```
Add a structured per-request logging layer. Wrap request handling so every proxied call emits one JSON line (JSONL) with: a generated request id, timestamp, resolved profile name, profile kind, requested model, upstream model, upstream host, HTTP method, path, response status, latency_ms, and whether it streamed. Write to a configurable path (env CCPROXY_LOG_FILE, default ~/.local/state/claude-proxy/requests.jsonl) and also to stderr at INFO. NEVER log header values, api keys, request bodies, or message content — only the metadata fields listed. Make the logger injectable so tests can capture records. Acceptance: (1) a proxied request produces exactly one JSONL record with the listed fields populated; (2) no secret or body/message content appears in any record; (3) the log path is env-overridable; (4) pytest captures emitted records for both a passthrough and an openai request and asserts the schema and absence of secrets.
---
Add token + cost accounting to the log records. Capture input_tokens and output_tokens from the upstream usage when present, else from a counted fallback, and compute an estimated cost from a per-profile price table defined in config.toml ([profiles.<name>.pricing] with input_per_mtok and output_per_mtok). Attach input_tokens, output_tokens, and est_cost_usd to each request record from the previous ticket; when no pricing is configured for a profile, set est_cost_usd to null (do not guess). Acceptance: (1) records carry input/output tokens and est_cost_usd computed from the profile's pricing; (2) a profile with no pricing logs null cost without error; (3) streaming responses still report output tokens (upstream usage or counted fallback); (4) pytest covers cost math for a known token count + price and the no-pricing case.
---
Correlate proxy records with Commander dispatches. Read optional inbound correlation headers the caller may set — X-CCProxy-Run (sprint/run id), X-CCProxy-Role (agent role), X-CCProxy-Ticket — and include them on each request record (null when absent). These are metadata only and must never alter routing. Document in README how Commander injects them per dispatched subprocess (e.g. via the agent's request headers or an env the launcher forwards) so a log line can be traced to a specific agent/ticket. Acceptance: (1) when the correlation headers are present they appear on the record; (2) when absent the fields are null and nothing breaks; (3) the headers never influence profile selection or translation; (4) README documents the injection; (5) pytest asserts headers flow into records and do not affect routing.
---
Surface a rolling summary endpoint. Add `GET /metrics` returning per-profile aggregates since process start (or a configurable window): request count, total input/output tokens, total est_cost_usd, error count, and p50/p95 latency_ms. Keep the aggregation in-memory and cheap; do not read back the full JSONL on each call. Exclude /health and /metrics themselves from the counts. Acceptance: (1) after several requests across two profiles, /metrics shows correct per-profile counts, token totals, and summed cost; (2) errors are counted separately from successes; (3) /metrics and /health are not self-counted; (4) pytest drives a handful of stub requests and asserts the aggregates.
```

## Notes

- **Do this one first of M5+.** Everything else (caching gains, replay, routing
  decisions in Commander) is guesswork until you can see per-profile cost.
- **Secrets/content never logged** — metadata only. Treat that as a hard test,
  not a guideline.
- **Order:** 1 (records) → 2 (cost) and 3 (correlation) extend the record → 4
  (summary) aggregates it.
- **Correlation headers are metadata only** — they must not influence routing, or
  you've smuggled routing logic into the proxy.
- **Headless project.** Design Refs from DESIGN.md "Intent" / "Tokens" (the CLI
  output conventions); UAT via `curl /metrics` + inspecting the JSONL, MANUAL.

## Posted issues

| # | Title | Size |
|---|-------|------|
| _pending_ | | |
