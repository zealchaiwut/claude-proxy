# Design System

**Register:** product — the design serves the product; claude-proxy is invisible infrastructure, not a brand surface.

**Scene:** a solo developer in a terminal, or Commander running unattended on a headless host, needs to know at a glance which backend a run used and why — under fluorescent office light or at 2am over an iPad SSH session.

## Intent

claude-proxy has **no GUI**. Its only surfaces are a CLI (`ccswitch`), config files, structured logs, and the proxied HTTP traffic. The design goal is *legibility under glance*: a person should be able to tell, in one line, which profile is active and which upstream a request hit — and never have to guess. It is deliberately NOT a dashboard, NOT colorful chrome, NOT another thing to babysit. When everything is healthy, you should forget it exists.

Because there is no frontend (`.html/.css/.jsx/.tsx`), Commander's `impeccable detect` design gate skips this project; this file exists to satisfy the design-docs guard and to record the CLI/output conventions below in place of a visual token system.

## Tokens

No visual surface, so the palette table is intentionally not applicable. The equivalent "tokens" are terminal-output conventions, kept minimal and ANSI-optional (degrade to plain text when not a TTY):

| Role | Convention |
|------|------------|
| active profile | bold profile name, e.g. **openai** |
| upstream | dim, parenthesized, e.g. `(https://openrouter.ai/api/v1)` |
| ok / healthy | green `ok`, or plain `ok` when no color |
| warn | yellow `warn` |
| error | red `error`, non-zero exit code |
| secrets | never printed — referenced by env var name only |

## Typography

Monospace only — whatever the user's terminal uses. Hierarchy comes from structure, not type: one status line per fact, `key: value` alignment, and JSON-lines for machine-readable logs so Commander's log viewer and `ccswitch status` read the same records.

> Starter system stamped by scaffold_project so sprints can run (the design-docs
> guard requires this file). Refine with `/impeccable init`, then
> `/impeccable critique` on the first real screen.