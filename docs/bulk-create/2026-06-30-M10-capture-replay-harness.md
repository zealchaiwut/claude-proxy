# M10 — Capture & replay harness

**Date:** 2026-06-30
**Sprint label:** NEW
**Default labels:** enhancement, backend
**Status:** drafted

When a cheap backend botches a ticket, you want to know *why* — and to vet a new
model before trusting it with coder work — without re-running a whole sprint.
M10 optionally records redacted request/response pairs per dispatch, lets you
replay a captured request against a different profile, and compares cost/tokens/
latency/output across profiles. This is how you answer "would profile X have done
this cheaper/better" offline. Depends on M5 (records/correlation), M1–M3
(translation); pairs with M8/M9 for accurate cost in the comparison.

## Prompts

Paste one code block into the Bulk Create textarea. Prompts are `---`-separated.

```
Add optional request/response capture. Gated by env CCPROXY_CAPTURE=1 (off by default) and/or per-profile [profiles.<name>].capture = true, record each proxied exchange to a capture dir (default ~/.local/state/claude-proxy/captures/<request_id>.json): the inbound Anthropic request, the resolved profile, the final Anthropic response (reassembled from the stream if streamed), and timing/usage. REDACT secrets — never store Authorization/api keys; the request body is stored but the harness is for your own machine, so document that captures contain message content and should be treated as sensitive. Reuse the M5 request_id so a capture links to its log record. Acceptance: (1) with capture enabled a request writes one capture file keyed by request_id; (2) no auth header / api key is present in the file; (3) a streamed response is reassembled into the captured Anthropic response; (4) capture is fully off (no files, no overhead) by default; (5) pytest covers a captured non-streaming and streaming exchange and the redaction.
---
Add a replay command. `ccproxy replay <capture-file> --profile <name>` loads the captured Anthropic request and re-issues it against the chosen profile through the same translation/forwarding code path (not a separate client), printing the response and writing a replay artifact (response + tokens + est_cost + latency) next to the capture. Support `--stream/--no-stream`. Replays must go through the normal profile resolution + translation so the result reflects real proxy behavior. Acceptance: (1) replaying a capture against its original profile reproduces a comparable response; (2) replaying against a different profile routes to that profile's upstream and records its tokens/cost/latency; (3) the replay uses the real translation path, not a bypass; (4) pytest replays a fixture capture against two stub profiles and asserts per-profile artifacts.
---
Add a compare command. `ccproxy compare <capture-file> --profiles a,b[,c]` replays the captured request across the listed profiles and prints a summary table: per profile the est_cost_usd, input/output tokens, latency_ms, finish/stop reason, and a short response preview, plus a cheapest/fastest marker. Save a JSON manifest of the comparison. This is decision support only — it does not change any config or routing. Acceptance: (1) compare across two+ profiles produces a table with cost/tokens/latency per profile and a cheapest marker; (2) a JSON manifest is written; (3) a profile that errors is shown as failed in the table without aborting the others; (4) pytest covers a two-profile comparison and the one-fails-one-succeeds case.
```

## Notes

- **This is decision support, not routing.** Compare tells *you* which backend to
  put behind a role in Commander's dispatch config — the proxy never auto-acts on
  it. Keeps the routing decision in one place (Commander).
- **Captures contain message content** — they're a local debugging aid; document
  that and keep them out of any shared/committed location. Secrets are redacted;
  content is not.
- **Replay must use the real path** (ticket 2) or the comparison is a lie — no
  shortcut client.
- **Order:** 1 (capture) → 2 (replay) → 3 (compare). 3 builds directly on 2.
- **Accurate cost in compare depends on M8** (tokenizer) and benefits from M9
  (cache gap) — run those first if cost fidelity matters for the decision.
- **Headless project.** Design Refs from DESIGN.md "Intent"; UAT via capture a
  real `claude -p` exchange → `ccproxy compare` across profiles, MANUAL.

## Posted issues

| # | Title | Size |
|---|-------|------|
| _pending_ | | |
