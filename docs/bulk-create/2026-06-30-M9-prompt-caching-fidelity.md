# M9 — Prompt-caching fidelity

**Date:** 2026-06-30
**Sprint label:** NEW
**Default labels:** enhancement, backend
**Status:** drafted

Claude Code leans hard on Anthropic prompt caching (large stable system prompts +
tool definitions marked with cache_control). On the passthrough path that just
works; on the OpenAI path it's lost or handled differently, which on long agent
loops is real money and latency. M9 preserves caching where it exists, translates
or cleanly strips cache_control on the OpenAI path, and measures the cache gap so
you can see what the OpenAI backend is costing in re-sent context. Depends on
M1–M3 (translation), M5 (logging), ideally M8 (accurate counts).

## Prompts

Paste one code block into the Bulk Create textarea. Prompts are `---`-separated.

```
Handle Anthropic cache_control on the OpenAI translation path. When translating an Anthropic request that marks content blocks with cache_control (e.g. system blocks, tool definitions) to OpenAI form, do the right thing per profile: if the profile declares cache support (config.toml [profiles.<name>].prompt_cache = "auto"|"none" and an optional provider hint), map to the upstream's caching mechanism where one exists; otherwise STRIP cache_control cleanly so the request is still valid and never errors. Stripping must not reorder or alter the underlying content — only remove the cache marker. Acceptance: (1) a request with cache_control on system + tools translates to a valid OpenAI request with markers removed when prompt_cache="none"; (2) content and ordering are otherwise unchanged; (3) where a profile declares cache support, the mapping is applied (assert the shape; behavior may be provider-specific); (4) pytest covers strip and map cases and proves no content mutation beyond marker removal.
---
Preserve caching on the passthrough / anthropic path. Audit the passthrough and any anthropic-compatible handling to guarantee the proxy does NOT strip cache_control, reorder system blocks, or otherwise break Anthropic prompt caching — byte-for-byte forwarding must keep cache markers intact so subscription/anthropic requests keep their cache hits. Add a regression test asserting a cache_control-bearing request passes through unmodified. Acceptance: (1) a request with cache_control passes through the anthropic profile unmodified (markers and order intact); (2) cache_read/cache_creation usage from the upstream response is preserved in what the client receives; (3) pytest proves the passthrough body is byte-identical for a cache-bearing request.
---
Surface cache effectiveness in logging/metrics. Record cache-related usage on the M5 request record: on the anthropic path capture cache_creation_input_tokens and cache_read_input_tokens from Anthropic usage; on the OpenAI path record a cache_miss_estimate (the stable-prefix tokens that would have been cached but were re-sent, using the M8 tokenizer). Aggregate per profile in /metrics (cache hit ratio on anthropic, estimated re-sent tokens + est cost of the gap on openai). Acceptance: (1) anthropic records carry cache_read/creation tokens when present; (2) openai records carry a cache_miss_estimate; (3) /metrics shows per-profile cache effectiveness / estimated waste; (4) pytest covers extraction (anthropic) and estimate (openai) with fixtures.
```

## Notes

- **Value scales with sprint size on the OpenAI path** — for short runs the gap is
  noise; for long agent loops with big system prompts it's the main cost line.
- **Passthrough caching is preserve-don't-break** (ticket 2): the risk is an
  accidental body mutation in some handler silently killing cache hits — hence
  the byte-identical regression test.
- **OpenAI caching is provider-specific** — map where supported, strip safely
  elsewhere; never let a cache marker cause a 400.
- **Order:** 1 (openai translate) and 2 (passthrough preserve) are independent →
  3 (measure) depends on M5 + M8.
- **Headless project.** Design Refs from DESIGN.md "Intent"; UAT via /metrics
  cache stats over a repeated-prompt run, MANUAL.

## Posted issues

| # | Title | Size |
|---|-------|------|
| _pending_ | | |
