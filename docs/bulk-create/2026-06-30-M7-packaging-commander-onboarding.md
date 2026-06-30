# M7 — Packaging & Commander onboarding

**Date:** 2026-06-30
**Sprint label:** NEW
**Default labels:** enhancement, backend
**Status:** drafted

For Commander to depend on the proxy, it has to be always-up infra, not a script
you remember to start. M7 makes claude-proxy installable, runnable as a managed
service that restarts on failure, health-checkable for Commander's pre-dispatch
gate, and documents the onboarding path. This milestone matters even if you only
ever use passthrough — a reliable, always-on endpoint is a prerequisite for
Commander leaning on it at all. Depends on M0–M4 (M5/M6 optional but ideal).

## Prompts

Paste one code block into the Bulk Create textarea. Prompts are `---`-separated.

```
Make claude-proxy installable with console entrypoints. In pyproject.toml define two scripts: `claude-proxy` (starts the server, reads config.toml + env, binds [server].host/port) and `ccswitch` (the M4 CLI). Add build metadata so `uv tool install .` / `pipx install .` works, pin runtime deps, and ship a `config.example.toml` (anthropic passthrough + one openai profile with pricing/timeouts/retry placeholders and an api_key_env reference) plus a `.env.example` (the env var NAMES only, no values). Document install + first-run in README. Acceptance: (1) `uv tool install .` (or pipx) exposes working `claude-proxy` and `ccswitch` commands; (2) `claude-proxy` starts from config.example.toml after copying it to the expected path; (3) config.example.toml and .env.example contain no real secrets; (4) pytest/CI smoke: the entrypoints import and `ccswitch list` runs against the example config.
---
Provide service units that keep the proxy running. Add templates: a macOS launchd plist and a Linux systemd user unit that run `claude-proxy` bound to 127.0.0.1:8788, restart on failure, and load environment from a documented env file (KeepAlive/Restart=on-failure). Add `scripts/install_service.py` (or shell) that installs/loads the unit for the current platform and prints status. Do not bake secrets into the unit — reference the env file. Acceptance: (1) installing the unit starts the proxy and it survives a kill (auto-restart); (2) the unit loads env from the documented file, not inline secrets; (3) the installer detects platform and gives a clear message on unsupported ones; (4) README documents install/uninstall/logs for both platforms.
---
Harden health/readiness for Commander's lifecycle gate. Extend `GET /health` to return {status, version, active_default_profile, upstream} quickly (no upstream call). Add `GET /ready` that does a shallow reachability check of the active default profile's upstream (cheap, short-timeout, cached for a few seconds) returning ok/degraded with the profile name — so Commander can gate dispatch on readiness without hammering the upstream. Neither endpoint exposes secrets. Acceptance: (1) /health returns version + active default profile + upstream and never calls the upstream; (2) /ready reports degraded when the active upstream is unreachable and ok when reachable, within a short timeout; (3) /ready results are cached briefly to avoid per-poll upstream load; (4) pytest covers /health shape and /ready ok/degraded against stub upstreams.
---
Write the Commander onboarding runbook at `docs/workflow.md`. Step-by-step: (a) create the claude-proxy repo with PRODUCT.md + DESIGN.md on develop (the design-docs guard requires both); (b) run scaffold_project.py and init_project.py to stamp standard docs and create the managed clones; (c) install + start the proxy service (M7 tickets 1–2); (d) point Claude Code / Commander dispatch at it via ANTHROPIC_BASE_URL=http://localhost:8788; (e) set CCPROXY_PROFILE per dispatched subprocess (cheap/local for tester/estimator/docs-only, subscription anthropic for coder); (f) verify with /health, /ready, and a one-ticket smoke sprint. Acceptance: (1) following the runbook from a clean repo yields a proxy Commander can health-gate and dispatch through; (2) the doc names the exact env vars and the per-dispatch profile mechanism; (3) the design-docs guard + scaffold prerequisites are called out before the first sprint; (4) a reviewer can complete onboarding using only docs/workflow.md.
```

## Notes

- **This may be more urgent than the OpenAI-side milestones** — it applies even in
  pure passthrough. If "Commander can rely on the proxy" is the near-term goal,
  run M7 before M8/M9.
- **Order:** 1 (install) → 2 (service) → 3 (health/ready) → 4 (runbook ties it
  together). The runbook should reference the real commands the prior tickets
  create.
- **Secrets live in the env file, never in units, config.example, or .env.example
  (names only).**
- **Headless project.** Design Refs from DESIGN.md "Intent"; UAT via install →
  kill → auto-restart → `/health`+`/ready`, MANUAL.

## Posted issues

| # | Title | Size |
|---|-------|------|
| _pending_ | | |
