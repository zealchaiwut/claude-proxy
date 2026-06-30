# ARCHITECTURE.md — Claude Code Switch Proxy (claude-proxy)

> Deep design + translation contract. The lean, sprint-guard-facing summary
> lives in `PRODUCT.md`; the roadmap in `MILESTONES.md`. This file is the detail
> commander/sprints can consult but don't gate on.

## 1. One-line summary

A local proxy that sits in front of Claude Code so you can **toggle, before
running `claude -p`, which backend serves the request** — your Anthropic
(Claude) subscription, or any OpenAI-compatible endpoint — with room to add
more providers later.

## 2. Problem

Claude Code talks to a single endpoint determined by `ANTHROPIC_BASE_URL`.
Switching backends today means editing env vars, restarting your shell, and
remembering which key goes where. You want one stable endpoint and a **one-word
switch**:

```bash
ccswitch use anthropic     # next `claude -p` runs on your Claude subscription
ccswitch use openai        # next `claude -p` runs on an OpenAI-compatible model
claude -p "refactor this function"
```

The switch must take effect for the *next* `claude` invocation without editing
config files or restarting anything.

## 3. Who it's for

A single developer (you) on one machine, running Claude Code locally. Not a
multi-user gateway, not a hosted service. (If that changes later, omnigent is
the off-the-shelf answer — see §11.)

## 4. How Claude Code routing actually works (the contract we implement)

Claude Code is just an Anthropic API client. When `ANTHROPIC_BASE_URL` points at
our proxy, Claude Code sends us:

- `POST /v1/messages` — the main inference call (streaming and non-streaming).
- `POST /v1/messages/count_tokens` — token counting before/around requests.
- `GET /v1/models` (sometimes) — model listing.
- An **auth header** it already holds: either an OAuth bearer (subscription
  login) or `x-api-key` (API-key login), plus `anthropic-version` and
  `anthropic-beta` headers.

Two consequences shape the whole design:

1. **Whatever auth Claude Code is logged into, it attaches itself.** A proxy
   that forwards bytes untouched to `api.anthropic.com` therefore preserves the
   subscription login — we don't need to handle OAuth ourselves.
2. Claude Code's loop is **tool-call heavy** (read/edit files, run bash). A
   backend that can't do tool calls is effectively unusable for it. Tool
   translation is therefore part of the MVP, not a nice-to-have.

## 5. The two modes

### Mode A — `anthropic` (transparent passthrough)

The proxy relays the request to `https://api.anthropic.com` **without
translation**, forwarding Claude Code's own auth and version headers verbatim.

- Works with a **Claude Pro/Max subscription** (the OAuth bearer is passed
  through) *or* an Anthropic API key — whichever Claude Code is logged into.
- Near-zero behavior change vs. talking to Anthropic directly; this mode exists
  so the toggle has a "native" side and so we can log/inspect traffic.

> **Honesty note on subscriptions:** the subscription path uses OAuth tokens
> that Claude Code obtained by browser login. Passing them through a
> **localhost** proxy you control is fine. Routing them through a *remote* host,
> or scripting around the OAuth flow, is a ToS gray area — keep Mode A
> local-only.

### Mode B — `openai` (translate to OpenAI-compatible)

The proxy translates Anthropic Messages ⇄ OpenAI Chat Completions and forwards
to a configured OpenAI-compatible endpoint (OpenRouter, LiteLLM, vLLM, Ollama,
Azure OpenAI, IBM ICA, etc.). This is the same job claude-proxy does; we reuse
its translation approach.

### Future modes

`anthropic-compatible` (e.g. Bedrock/Vertex Anthropic), additional named
OpenAI-compatible profiles, and model-level routing (§10).

## 6. The toggle

State lives in a small file the running server reads **per request**, so the
switch is instant and needs no restart:

```
~/.config/ccswitch/state.json     # { "active": "openai" }
~/.config/ccswitch/config.toml    # profile definitions (below)
```

- `ccswitch use <profile>` writes `state.json` and prints a confirmation.
- `ccswitch status` prints the active profile and its resolved upstream.
- `ccswitch list` shows all profiles.
- Claude Code stays pointed at `ANTHROPIC_BASE_URL=http://localhost:8788`
  permanently; only the active profile changes underneath it.

Optional ergonomics (post-MVP): a `claude` shell wrapper / `cc <profile> -p ...`
shorthand, and a `--profile` query param the proxy honors for one-off overrides.

### Commander / sprint integration

Commander's sprint manager dispatches each agent (BA / Coder / Tester /
Estimator) as a `claude` CLI **subprocess**, already deciding a `(backend, model)`
pair per dispatch (`model_routing.py`: size → model, docs-only → Haiku;
`coder_backend` ∈ {claude-code, cline}). claude-proxy slots in *under* the
`claude-code` backend as the component that decides **which provider actually
serves that model**, without Commander needing to change.

**Selection mechanism — per dispatch, not global.** Commander runs agents
concurrently (warm multi-coder worktree pool via `max_coder_slots`, plus pipeline
mode where a coder and a tester run at once). A single global mutable
`state.json` "active profile" would therefore **race** between simultaneous
dispatches. The correct switch is per-subprocess env (and/or a per-request
`?profile=`/header), since each dispatch already gets its own environment:

```
ANTHROPIC_BASE_URL=http://localhost:8788      # always the proxy
CCPROXY_PROFILE=openai                         # this dispatch's backend
CLAUDE_CODE_OAUTH_TOKEN=<subscription token>   # injected by Commander; passed through in 'anthropic' mode
```

The human's global `ccswitch use <profile>` toggle (state file read per request)
remains valid for direct, single-stream `claude -p`; it is the convenience
default, while Commander overrides per dispatch.

**This maps cleanly onto Commander's existing routing.** Just as Commander routes
mechanical roles to Haiku and coding to Sonnet, it can route *mechanical roles to
a cheap/local OpenAI-compatible profile* and *coding to the subscription
`anthropic` profile* — a provider axis layered on top of the model axis. The
subscription path works because the `anthropic` profile is a transparent
passthrough that forwards the `CLAUDE_CODE_OAUTH_TOKEN` Commander injects.

**Lifecycle.** Commander ensures the proxy is up (`/health`) and that
`ANTHROPIC_BASE_URL` points at it before dispatching. The proxy stays a dumb,
stable endpoint: Commander owns *when*, *which model*, and *which profile*; the
proxy owns *how the call is forwarded/translated*.

**Project compliance.** claude-proxy is itself a Commander-managed project, so it
must satisfy the sprint **design-docs guard** (`PRODUCT.md` + `DESIGN.md` present
on the develop branch) and the standard docs layout. Run
`python3 scripts/scaffold_project.py --project <path>` from the Commander repo to
stamp any missing standard files before the first sprint.

## 7. Configuration model (profiles)

A profile names a backend. The MVP ships two profile *kinds*; adding a provider
later is adding a profile, not changing code.

```toml
[server]
host = "127.0.0.1"
port = 8788

[profiles.anthropic]
kind = "passthrough"
upstream = "https://api.anthropic.com"

[profiles.openai]
kind = "openai"
upstream = "https://openrouter.ai/api/v1"
api_key_env = "OPENAI_PROXY_KEY"     # read from env, never written to disk
model = "anthropic/claude-sonnet-4.6"   # model sent upstream
# optional model map: Claude Code's requested model -> upstream model
[profiles.openai.model_map]
"claude-3-5-haiku-20241022" = "openai/gpt-4o-mini"
```

Design rules:

- **Secrets come from env**, referenced by name in config. Never store keys in
  `config.toml` or `state.json`.
- Each profile fully determines upstream URL, auth, model, and translation
  behavior, so the toggle is a single source of truth.

## 8. Translation requirements (Mode B, the hard part)

To make Claude Code genuinely usable on an OpenAI-compatible model, translate:

- **System prompt:** Anthropic `system` (string or blocks) → OpenAI `system`
  message.
- **Messages & content blocks:** `text`, `image` ⇄ OpenAI message content;
  preserve ordering and roles.
- **Tools:** Anthropic `tools` → OpenAI `tools` (function schema); Anthropic
  `tool_use` blocks ⇄ OpenAI `tool_calls`; Anthropic `tool_result` ⇄ OpenAI
  `role:"tool"` messages. This is what lets Claude Code read/edit/run.
- **Tool choice:** `tool_choice` mapping (`auto`/`any`/named).
- **Streaming (SSE):** map OpenAI deltas → Anthropic event stream
  (`message_start`, `content_block_start`, `content_block_delta`,
  `content_block_stop`, `message_delta`, `message_stop`), including streamed
  `tool_use` argument fragments.
- **Stop reasons:** OpenAI `finish_reason` → Anthropic `stop_reason`
  (`end_turn` / `tool_use` / `max_tokens`).
- **Usage:** map token counts back into Anthropic's `usage` shape.
- **`/v1/messages/count_tokens`:** answer it (proxy-side tokenizer estimate is
  acceptable for MVP) so Claude Code doesn't error.
- **Extended thinking / XML tool-call fallback:** optional, mirror
  claude-proxy's approach for upstreams lacking native function calling.

## 9. Non-goals (MVP)

- Multi-user, auth, or remote hosting.
- Orchestrating multiple coding agents (that's omnigent's job).
- Sandboxing / policy enforcement / spend caps.
- A GUI. The toggle is a CLI.
- Perfect token-count parity with Anthropic's tokenizer.

## 10. Future / "switch to other models"

- **More OpenAI-compatible profiles** side by side (one per provider/model).
- **Per-model routing:** map Claude Code's requested model (it asks for a
  "big" model and a "small"/haiku model) to different upstreams — e.g. haiku →
  a cheap local model, sonnet → a hosted one.
- **Anthropic-compatible upstreams** (Bedrock, Vertex) as a third profile kind.
- **Per-request override** via header/query param for scripted A/B runs.
- **Usage/cost logging** per profile.

## 11. Build vs. reuse — where the reference repos fit

- **teer823/claude-proxy** — the direct blueprint for Mode B (Anthropic⇄OpenAI
  translation, streaming, web_search/thinking handling, port-bound FastAPI
  app). Reuse its `schemas/` and `services/translator.py` shape; add passthrough
  (Mode A) + the profile/toggle layer on top.
- **omnigent-ai/omnigent** — *not* a base for this MVP (it's a full multi-agent
  meta-harness with web/mobile UI, policies, sandboxes, team accounts). But it
  validates the concept and is the right tool if you later want multi-agent
  orchestration. Borrow its **credential model** as a mental map: it splits
  credentials into *API key* / *subscription* / *gateway*, which is exactly our
  passthrough-vs-OpenAI-compatible split, and it switches models mid-session
  with `/model`. If you ever outgrow this proxy, adopt omnigent instead of
  scaling this up.

## 12. Tech stack

Python 3.11+ / FastAPI / httpx / pydantic — same as claude-proxy, so its
translation code drops in with minimal change.

## 13. Definition of done (MVP)

`ANTHROPIC_BASE_URL=http://localhost:8788` set once; then:

```bash
ccswitch use anthropic && claude -p "summarize README"   # runs on subscription
ccswitch use openai    && claude -p "summarize README"   # runs on OpenAI-compatible model
```

Both complete a real multi-turn, tool-using Claude Code task (read a file, make
an edit) with streaming, no restarts between switches.