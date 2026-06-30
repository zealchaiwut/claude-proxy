# Commander Onboarding Runbook

This runbook walks a new contributor or sprint reviewer from a clean environment
to a fully operational Commander dispatch loop. Follow steps (a)–(f) in order.
No other file outside `docs/` is required to complete onboarding.

> This document describes the setup for the `claude-proxy` project and the
> Commander agents that drive it. For a description of how work flows through
> the pipeline once the system is running, see the **Pipeline overview** section
> at the bottom of this file.

---

## Prerequisites

Before any sprint can be dispatched, two design-docs guard files **must** exist
on the `develop` branch of the `claude-proxy` repo:

| File | Purpose |
|------|---------|
| `PRODUCT.md` | Product context — what the service is and who uses it |
| `DESIGN.md` | Design system — visual tokens, typography, intent |

**Why this matters:** Commander's scaffold step checks for both files at startup.
If either file is absent on `develop`, the design-docs guard raises an error and
blocks the sprint from running. Placeholder content is enough — the guard only
checks for file presence, not content completeness.

You also need:

- **Git** ≥ 2.38 and **GitHub CLI** (`gh`) authenticated to your GitHub account
- **Python** ≥ 3.12 and `uv` (or `pipx`) installed
- An **Anthropic API key** (subscription or API key) for the coder subprocess
- A **cheap / local inference backend** for tester, estimator, and docs-only
  subprocesses (e.g. Ollama, OpenRouter, or any OpenAI-compatible endpoint)

---

## Step (a) — Create the `claude-proxy` repo with design docs on `develop`

```bash
# 1. Create the repo on GitHub (public or private)
gh repo create claude-proxy --clone --public
cd claude-proxy

# 2. Create the develop branch and commit the design-docs guard files
git checkout -b develop

# Minimal PRODUCT.md — edit later with /impeccable init
cat > PRODUCT.md << 'EOF'
# Product Context

## What claude-proxy Is

A lightweight HTTP proxy for the Anthropic API.

## Target Users

Developers using Claude Code and Commander.

## Core User Flows

1. Route Claude Code through a local proxy to switch backends.
2. Run Commander dispatch with per-subprocess profile routing.

## Design Principles

- Speed: minimal latency overhead.
- Clarity: transparent request/response forwarding.
EOF

# Minimal DESIGN.md — edit later with /impeccable init
cat > DESIGN.md << 'EOF'
# Design System

## Intent

Functional CLI / API tooling. Minimal visual surface.

## Tokens

| Role | Light | Dark |
|------|-------|------|
| `--bg` | #ffffff | #0d1117 |
| `--text` | #24292f | #e6edf3 |
| `--accent` | #0969da | #58a6ff |

## Typography

Monospace for all code surfaces; system-ui for prose.
EOF

git add PRODUCT.md DESIGN.md
git commit -m "chore: add design-docs guard files to develop"
git push -u origin develop
```

**Why:** Without `PRODUCT.md` and `DESIGN.md` on `develop`, the design-docs guard
blocks `scaffold_project.py` with an error like:

```
Error: design-docs guard failed — PRODUCT.md and/or DESIGN.md not found on develop.
Commit both files to develop before running scaffold.
```

---

## Step (b) — Scaffold the project and initialise Commander

Run `scaffold_project.py` first to stamp standard docs, then `init_project.py`
to create GitHub labels and the Commander sprint structure.

### 1. scaffold_project.py

```bash
python scaffold_project.py --repo claude-proxy
```

Expected output:

```
[scaffold] Checking design-docs guard... ok
[scaffold] Stamping docs/workflow.md ... created
[scaffold] Stamping docs/quickstart.md ... created
[scaffold] Stamping docs/tutorial.md ... created
[scaffold] Stamping docs/architecture.md ... created
[scaffold] Stamping docs/milestones.md ... created
[scaffold] Stamping CHANGELOG.md ... created
[scaffold] Done. 6 files written to develop.
```

`PRODUCT.md` and `DESIGN.md` are left unchanged — `scaffold_project.py` reads
them but does not overwrite them.

### 2. init_project.py

```bash
python init_project.py --repo claude-proxy
```

Expected output:

```
[init] Creating GitHub labels...
  + in-progress
  + SIT
  + UAT
  + needs-rework
  + enhancement
[init] Verifying develop branch... ok
[init] Commander project initialised for claude-proxy.
```

After this step, the repo is ready to accept sprint tickets.

---

## Step (c) — Install and start the proxy service

This step covers the two installer-related M7 tickets:

- **#48 — Make claude-proxy installable with console entrypoints**: adds
  `claude-proxy` and `ccswitch` as `$PATH` commands via `uv tool install` /
  `pipx install`.
- **#49 — Add systemd and launchd service units**: provides a platform-native
  service that starts automatically on login and restarts after failures.

### Install the CLI commands (issue #48)

```bash
# Clone and install (from the repo root)
git clone https://github.com/<your-org>/claude-proxy
cd claude-proxy

# with uv (recommended)
uv tool install .

# OR with pipx
pipx install .
```

After install, verify both commands are on your `$PATH`:

```bash
which claude-proxy   # → /home/<user>/.local/bin/claude-proxy (Linux)
                     # → /Users/<user>/.local/bin/claude-proxy (macOS)
which ccswitch
```

Copy the example config and add your API key:

```bash
cp config.example.toml config.toml
# Open config.toml and confirm [server] host/port (default: 127.0.0.1:8788)
export ANTHROPIC_API_KEY=sk-ant-...
```

### Install the background service (issue #49)

Run the installer script — it detects your platform automatically:

```bash
python scripts/install_service.py
```

The installer:
- **macOS** — installs a launchd user agent
  (`~/Library/LaunchAgents/com.zealchaiwut.claude-proxy.plist`) that starts at
  login and restarts on crash.
- **Linux** — installs a systemd user unit
  (`~/.config/systemd/user/claude-proxy.service`) with `Restart=on-failure`.

No secrets are written into the unit file. Credentials are loaded from
`~/.config/claude-proxy/env` at startup — the installer creates a template if
the file does not exist.

**Add your API key to the env file:**

```bash
# ~/.config/claude-proxy/env
ANTHROPIC_API_KEY=sk-ant-...
CCPROXY_PROFILE=anthropic
```

**Check that the service is running:**

```bash
# macOS
launchctl list com.zealchaiwut.claude-proxy

# Linux
systemctl --user status claude-proxy
```

**View logs:**

```bash
# macOS
tail -f ~/Library/Logs/claude-proxy/claude-proxy.log

# Linux
journalctl --user -u claude-proxy -f
```

---

## Step (d) — Point Claude Code and Commander at the proxy

Set `ANTHROPIC_BASE_URL` to the proxy's local address before launching either
Claude Code or Commander dispatch:

```bash
export ANTHROPIC_BASE_URL=http://localhost:8788
```

This single variable redirects all Anthropic API calls through the proxy. Your
existing `ANTHROPIC_API_KEY` or Anthropic subscription login works unchanged —
the proxy forwards credentials to the upstream API without modification.

**Where to set it:**

- In your shell profile (`~/.zshrc`, `~/.bashrc`) so every new terminal session
  picks it up automatically.
- In the Commander launcher env block so every dispatched subprocess inherits it.

Both Claude Code **and** Commander dispatch must be pointed at this value.
If Commander runs with the correct `ANTHROPIC_BASE_URL` but you open Claude Code
in a separate terminal without it, that terminal will bypass the proxy.

---

## Step (e) — Configure per-subprocess profiles with `CCPROXY_PROFILE`

`CCPROXY_PROFILE` selects which proxy profile (and therefore which upstream
backend) a subprocess uses. Set it **per subprocess** in the Commander launcher:

| Subprocess | Profile value | Upstream |
|-----------|--------------|---------|
| Coder | `anthropic` | Anthropic API (subscription or API key) — full capability |
| Tester | `cheap` or `local` | Local or low-cost backend (e.g. Ollama, OpenRouter) |
| Estimator | `cheap` or `local` | Same as tester |
| Docs-only | `cheap` or `local` | Same as tester |

Example Commander launcher env blocks:

```bash
# Coder subprocess — full Anthropic subscription backend
CCPROXY_PROFILE=anthropic claude --profile coder ...

# Tester / Estimator / Docs-only subprocesses — cheap or local backend
CCPROXY_PROFILE=cheap claude --profile tester ...
CCPROXY_PROFILE=cheap claude --profile estimator ...
CCPROXY_PROFILE=cheap claude --profile documentor ...
```

The proxy routes each subprocess independently — there is no shared mutable
state, so profiles never bleed between concurrent subprocesses.

Define the `cheap` or `local` profile in `config.toml`:

```toml
[profiles.cheap]
kind = "openai"
upstream = "http://localhost:11434/v1"   # e.g. Ollama
api_key_env = "OLLAMA_API_KEY"
model = "llama3.2"

[profiles.cheap.model_map]
"claude-haiku-4-5-20251001" = "llama3.2"
"claude-sonnet-4-6" = "llama3.2"
```

---

## Step (f) — Health-gate verification and smoke sprint

### Verify `/health`

```bash
curl -s http://127.0.0.1:8788/health
```

Expected success response:

```json
{
  "status": "ok",
  "version": "0.1.0",
  "active_default_profile": "anthropic",
  "upstream": "https://api.anthropic.com"
}
```

`/health` always returns HTTP 200 while the process is running. It never probes
the upstream — it reflects process health only.

### Verify `/ready`

```bash
curl -s http://127.0.0.1:8788/ready
```

Expected success response (upstream reachable):

```json
{
  "status": "ok",
  "profile": "anthropic"
}
```

If the upstream is unreachable, `/ready` returns HTTP 200 with:

```json
{
  "status": "degraded",
  "profile": "anthropic"
}
```

`/ready` performs a shallow TCP probe (≤ 2 s timeout) and caches the result for
5 s to avoid hammering the upstream on repeated polls. Commander checks `/ready`
before dispatching any sprint work — dispatch is withheld while status is
`degraded` and resumes automatically once the upstream is reachable again.

### One-ticket smoke sprint

With the proxy running and `/ready` returning `ok`, dispatch a single-ticket
smoke sprint to confirm end-to-end Commander dispatch:

```bash
# Create a minimal test issue
gh issue create \
  --repo <your-org>/claude-proxy \
  --title "Smoke test: hello-world endpoint" \
  --label "sprint-smoke" \
  --body "$(cat <<'EOF'
## What & Why
Verify Commander dispatch works end-to-end through the proxy.

## Acceptance Criteria
- [ ] GET /hello returns {"hello": "world"} with HTTP 200
EOF
)"

# Dispatch Commander on that label
commander run --label sprint-smoke
```

Expected Commander output (abbreviated):

```
[commander] /ready → ok (profile: anthropic)
[commander] Dispatching sprint-smoke (1 ticket)
[commander] ticket #<N>: coder → in-progress
[commander] ticket #<N>: coder → SIT
[commander] ticket #<N>: tester → UAT
[commander] Sprint sprint-smoke complete. 1/1 tickets passed.
```

Proxy logs confirm dispatch routing:

```bash
tail -n 5 ~/.local/state/claude-proxy/requests.jsonl | python -m json.tool
```

You should see one record per agent subprocess, each with the expected
`profile_name` (`anthropic` for the coder, `cheap` or `local` for tester/estimator).

---

## Troubleshooting

| Symptom | Likely cause | Fix |
|---------|-------------|-----|
| `scaffold_project.py` raises design-docs guard error | `PRODUCT.md` or `DESIGN.md` missing on `develop` | Follow step (a) |
| `claude-proxy: command not found` | Package not installed | Run `uv tool install .` (step c) |
| `/health` returns connection refused | Proxy not running | Run `python scripts/install_service.py` or `claude-proxy` directly |
| `/ready` returns `degraded` | Upstream unreachable | Check `ANTHROPIC_API_KEY` and network; inspect `config.toml` upstream |
| Commander dispatch hangs | `ANTHROPIC_BASE_URL` not set | Export `ANTHROPIC_BASE_URL=http://localhost:8788` (step d) |

---

## Pipeline overview

Once the system is running, work flows through three stages driven by Commander:

### Stage 1 — Bulk Create

- Paste prompts (separated by `---`) into the Bulk Create tab.
- **BA agent** drafts each ticket (title, body, AC, UAT steps), one per prompt.
- **Estimator** sizes each draft (S/M/L/XL).
- Review and edit the drafts, then post the selected ones as GitHub issues.

Records of past batches live in [bulk-create/](bulk-create/).

### Stage 2 — Run Sprint

For each ticket in a `sprint-N` label:

- **Coder** branches off develop, implements, and pushes (`in-progress` → `SIT`).
- **Tester** writes and runs tests per acceptance criterion, posts a report.
- **Fix loop** re-dispatches the coder on failure, up to 3 attempts, then tags
  `needs-rework`.
- **Quality gates** (typecheck, lint, design, pytest, merge-preview) must pass.
- **Documentor** updates the changelog and docs.
- On pass, the feature branch merges into the sprint branch and the issue → `UAT`.
- **Reviewer** runs once after the sprint PR, posts findings, and opens
  follow-up tickets.

### Stage 3 — Finish / Rerun Sprint

- **Finish:** the human reviews UAT tickets, closes the good ones; a sprint
  summary is posted as a GitHub issue, which marks the sprint finished.
- **Rerun:** tickets tagged `needs-rework` run as an independent sub-sprint
  (`sprint-N.1`, `sprint-N.2`, …) with their own label, branch, PR, and summary.

### Agents at a glance

| Stage | Agent | Role |
|-------|-------|------|
| Bulk Create | BA | Draft ticket title, body, AC, UAT steps |
| Bulk Create | Estimator | Size each draft |
| Run Sprint | Coder | Implement on a feature branch |
| Run Sprint | Tester | Write/run tests, post report |
| Run Sprint | Documentor | Update changelog and docs |
| Run Sprint | Reviewer | Review diff, open follow-up tickets |
