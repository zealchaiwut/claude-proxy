# M8 — Token-count accuracy

**Date:** 2026-06-30
**Sprint label:** NEW
**Default labels:** enhancement, backend
**Status:** drafted

The M1 count_tokens estimate (~chars/4) is fine to keep Claude Code's pre-flight
from erroring, but it makes M5's cost numbers and context-window math
untrustworthy. M8 swaps it for a real per-upstream-family tokenizer, wires that
into count_tokens and the streaming token fallback, and measures drift against
upstream-reported usage. Small milestone, high trust payoff. Depends on M1
(count_tokens), M5 (cost records consume these counts).

## Prompts

Paste one code block into the Bulk Create textarea. Prompts are `---`-separated.

```
Add a pluggable tokenizer abstraction. Define a small interface (count_tokens(messages, model) -> int) with implementations selectable per profile via config.toml ([profiles.<name>].tokenizer = "openai" | "heuristic"). Implement an OpenAI-family tokenizer backed by tiktoken as an OPTIONAL dependency: lazy-import it, pick the encoding by model name, and fall back to the existing chars/4 heuristic (with a one-time WARN) if tiktoken is unavailable or the model is unknown. Keep the heuristic as the default when no tokenizer is configured. Acceptance: (1) with tokenizer="openai" and tiktoken installed, counts match tiktoken for known fixtures; (2) with tiktoken absent, it falls back to the heuristic and warns once, without crashing; (3) an unknown model degrades to a default encoding/heuristic; (4) pytest covers the openai path (skip/xfail if tiktoken missing in CI) and the fallback path.
---
Wire the real tokenizer into count_tokens and the streaming output fallback. In openai mode, `/v1/messages/count_tokens` returns the configured tokenizer's count over the rendered request instead of chars/4; and where a streamed response lacks upstream usage, compute output_tokens with the same tokenizer rather than the heuristic. Anthropic-mode count_tokens still passes through to Anthropic unchanged. Acceptance: (1) count_tokens in openai mode reflects the configured tokenizer; (2) anthropic-mode count_tokens is still a passthrough; (3) streamed responses without upstream usage report tokenizer-based output_tokens; (4) pytest covers both modes and the streaming fallback.
---
Measure and surface tokenizer drift. When the upstream returns its own usage, compare proxy-estimated vs upstream-reported input/output tokens, record the delta on the M5 request record (token_drift_input, token_drift_output), and expose drift stats in /metrics (mean/abs-mean per profile). This turns "are our counts trustworthy" into a number you can watch. Acceptance: (1) records carry drift fields when upstream usage is present (null otherwise); (2) /metrics shows per-profile drift aggregates; (3) a fixture with known estimated vs reported counts yields the expected drift; (4) pytest covers drift computation and aggregation.
```

## Notes

- **tiktoken is optional on purpose** — keep the proxy installable without it
  (heuristic fallback), so a minimal deployment still runs.
- **Only worth doing once a paid/metered backend carries real load** — until then
  the M1 heuristic is acceptable.
- **Order:** 1 (tokenizer) → 2 (wire-in) → 3 (drift). 3 depends on M5 records.
- **Headless project.** Design Refs from DESIGN.md "Intent"; UAT via count_tokens
  responses + /metrics drift, MANUAL.

## Posted issues

| # | Title | Size |
|---|-------|------|
| _pending_ | | |
