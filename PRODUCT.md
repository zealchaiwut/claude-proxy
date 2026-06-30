# Product Context

## What claude-proxy Is

_One or two sentences: what claude-proxy does and the problem it solves._

claude-proxy is a local proxy that sits in front of the `claude` CLI so each invocation can be served either by your Anthropic (Claude) subscription or by any OpenAI-compatible model, chosen per run. Claude Code points at one stable endpoint (`ANTHROPIC_BASE_URL`) while the backend is selected by a one-word toggle for direct use, or by Commander per agent dispatch — so you switch models/providers without editing Claude Code or its keys.

## Target Users

_Who uses it, and what they are trying to accomplish._

- **The operator (solo dev, local machine).** Running `claude -p` directly and wanting to flip between the Claude subscription and a cheaper, local, or different OpenAI-compatible model on demand — without re-wiring env vars or losing the subscription login.
- **Commander (the orchestrator).** Its sprint manager dispatches BA / Coder / Tester / Estimator agents as `claude` subprocesses and already routes per ticket (size → model, docs-only → Haiku, `coder_backend` = claude-code/cline). claude-proxy adds the missing axis: choosing *which provider serves that model* per dispatch — e.g. a cheap or local backend for mechanical roles (tester, estimator, docs-only) while coding stays on subscription Claude.

## Core User Flows

1. **Manual toggle (primary).** `ccswitch use openai` (or `anthropic`) → `claude -p "…"`. The next invocation lands on the chosen backend with no restart.
2. **Commander dispatch (primary for sprints).** The sprint manager sets the subprocess environment per agent (`ANTHROPIC_BASE_URL` + a profile selector) so each dispatch runs on its chosen backend. Because every dispatch is its own process env, this is concurrency-safe under multi-coder and pipeline mode — where a single global toggle would race.
3. **Native passthrough (supporting).** The `anthropic` profile relays requests untouched to api.anthropic.com, forwarding the `CLAUDE_CODE_OAUTH_TOKEN` Commander already injects, so a Claude Pro/Max subscription keeps working with no API-key swap and no OAuth handling in the proxy.

## Design Principles

- **Per-dispatch selection over global state.** The backend is chosen per process/request (env or header), so concurrent Commander agents never clobber each other; the global `ccswitch` toggle is only the human's convenience default.
- **One stable endpoint, instant switch.** Claude Code always points at the proxy; switching the active backend never needs a restart.
- **Transparent by default.** The Anthropic side is a byte-for-byte passthrough that preserves whatever login the caller holds (subscription OAuth or API key) — don't mangle what already works.
- **Provider-agnostic and extensible.** Adding a model or vendor is adding a profile, not changing code; this extends Commander's existing model routing with a provider axis rather than duplicating it.
- **Claude Code-grade fidelity on the OpenAI path.** Streaming and tool calls are table stakes (the agent loop is read/edit/run), not optional extras.
- **Local and secret-safe.** Runs on localhost; credentials come from env by name and are never written to disk.

> Starter context stamped by scaffold_project so sprints can run (the design-docs
> guard requires this file). Refine with `/impeccable init` or edit directly.