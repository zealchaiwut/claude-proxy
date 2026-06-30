# MILESTONES.md — Claude Code Switch Proxy

Goal of the MVP: **`claude -p` works against an OpenAI-compatible model**, and a
one-word toggle flips between that and your Anthropic subscription with no
restart.

The order below front-loads the thing that proves the architecture
(passthrough), then builds the OpenAI path up to the point where Claude Code is
actually usable (streaming + tools), then adds the toggle. Tool translation is
inside the MVP on purpose — without it Claude Code can read/think but can't
edit/run, so it's not really "usable."

---

## M0 — Skeleton + passthrough (Mode A)  ·  ~0.5–1 day

**Deliver:** FastAPI app on `127.0.0.1:8788` that transparently relays
`/v1/messages`, `/v1/messages/count_tokens`, `/v1/models` to
`https://api.anthropic.com`, forwarding Claude Code's auth/version headers
untouched. Streaming relayed as a raw byte/SSE passthrough.

**Done when:** `ANTHROPIC_BASE_URL=http://localhost:8788 claude -p "hi"` behaves
identically to talking to Anthropic directly, including on a subscription login.

**Proves:** the proxy is transparent and subscription auth survives the hop —
the load-bearing assumption of the whole design.

---

## M1 — OpenAI translation, non-streaming (Mode B core)  ·  ~1–2 days

**Deliver:** for a hard-coded `openai` upstream, translate non-streaming
`POST /v1/messages`:

- system + text messages Anthropic → OpenAI Chat Completions and back
- stop-reason and `usage` mapping
- a working `/v1/messages/count_tokens` (estimate is fine)

Reuse `schemas/` and `translator.py` from claude-proxy.

**Done when:** `claude -p "say hello in one word"` returns a correct
non-streaming answer from the OpenAI-compatible model.

---

## M2 — Streaming (SSE)  ·  ~1–2 days

**Deliver:** map OpenAI streaming deltas → Anthropic SSE events
(`message_start` → `content_block_*` → `message_delta` → `message_stop`).

**Done when:** `claude -p` streams tokens live from the OpenAI upstream, matching
the UX you get from Anthropic.

---

## M3 — Tool calls  ·  ~2–3 days  *(the make-or-break milestone)*

**Deliver:** full tool translation in both directions, streaming included:

- Anthropic `tools` → OpenAI function schema
- `tool_use` ⇄ `tool_calls`, `tool_result` ⇄ `role:"tool"`
- `tool_choice` mapping
- streamed tool-argument fragments
- XML tool-call fallback for upstreams without native function calling
  (optional, per claude-proxy)

**Done when:** Claude Code completes a real task on the OpenAI upstream — reads a
file, proposes an edit, applies it, runs a command — end to end.

---

## M4 — Toggle CLI + profiles + per-dispatch selection  ·  ~1–2 days

**Deliver:**

- `config.toml` (profiles) + `state.json` (active profile), read per request
- `ccswitch use|status|list` (the human's global default)
- **per-dispatch override** via env (`CCPROXY_PROFILE`) and/or `?profile=` —
  this is what Commander uses, and it's concurrency-safe because each agent
  subprocess has its own env (a global state file would race under multi-coder
  / pipeline dispatch)
- resolution order: request/header profile → env profile → global state default
- proxy selects passthrough vs. openai translation from the resolved profile
- secrets read from env by name, never persisted

**Done when:**
```bash
ccswitch use anthropic && claude -p "…"              # human default
CCPROXY_PROFILE=openai claude -p "…"                  # per-process override (Commander's path)
```
both work, and two concurrent runs with different `CCPROXY_PROFILE` values hit
different backends without interfering.

---

## Commander prerequisites (do before the first sprint)

Independent of M0–M4, for Commander to run sprints on this repo at all:

- **Design-docs guard:** `PRODUCT.md` + `DESIGN.md` must exist on the develop
  branch (already drafted). The guard blocks every ticket otherwise
  (`design_docs_missing`).
- **Standard layout:** run `scripts/scaffold_project.py --project <path>` from the
  Commander repo to stamp `README.md`, `CHANGELOG.md`, and the `docs/` tree.
- **Onboard the project:** `scripts/init_project.py` (`--nested` recommended) so
  the main/coder/tester/uat clones and `.commander/` exist.

---

## ✅ MVP = M0 + M1 + M2 + M3 + M4

A single stable endpoint, instant toggle, both backends doing real tool-using
Claude Code work. Rough estimate: **~1.5–2 focused weeks.**

---

## Post-MVP (pick as needed)

- **M5 — Multiple OpenAI profiles + model_map + role routing:** several providers
  side by side; map Claude Code's requested model (big vs. haiku) to different
  upstreams; and let Commander pick a profile by agent role/size — mechanical
  roles (tester, estimator, docs-only) → cheap/local backend, coder → subscription
  Claude — layering a provider axis onto Commander's existing model routing.
- **M6 — Debug logging & cost/usage:** daily-rotating request/response logs and
  per-profile token/cost accounting (claude-proxy's `debug_logger` is a start).
- **M7 — Anthropic-compatible profile kind:** Bedrock / Vertex passthrough.
- **M8 — Per-request override:** `?profile=` / header for scripted A/B runs, plus
  a `cc <profile> -p …` shell wrapper.
- **M9 — Packaging:** `pipx`/`uv tool install`, container image, `--reload` dev
  script, `/health` endpoint, basic translation tests.

---

## Risks / watch-items

- **Tool-call fidelity (M3)** is where most of the real effort and most bugs
  live; budget accordingly and test with an actual edit+bash task, not just chat.
- **Token-count parity** won't be exact; accept an estimate so Claude Code's
  pre-flight checks don't break.
- **Subscription via remote host** is a ToS gray area — keep Mode A local-only.
- **`anthropic-beta` features** Claude Code sends may not map cleanly to OpenAI
  upstreams; degrade gracefully rather than 500.