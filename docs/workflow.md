# Workflow

How work flows from raw idea to signed-off code in claude-proxy, driven by
Commander. Three stages: **Bulk Create**, **Run Sprint**, and **Finish / Rerun
Sprint**. Each stage names the agents it uses.

> This document describes current behavior. When a sprint changes the pipeline,
> the documentor updates this file.

## Stage 1 — Bulk Create

- Paste prompts (separated by `---`) into the Bulk Create tab.
- **BA agent** drafts each ticket (title, body, AC, UAT steps), one per prompt.
- **Estimator** sizes each draft (S/M/L/XL).
- Review and edit the drafts, then post the selected ones as GitHub issues.

Records of past batches live in [bulk-create/](bulk-create/).

## Stage 2 — Run Sprint

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

## Stage 3 — Finish / Rerun Sprint

- **Finish:** the human reviews UAT tickets, closes the good ones; a sprint
  summary is posted as a GitHub issue, which marks the sprint finished.
- **Rerun:** tickets tagged `needs-rework` run as an independent sub-sprint
  (`sprint-N.1`, `sprint-N.2`, …) with their own label, branch, PR, and summary.

## Agents at a glance

| Stage | Agent | Role |
|-------|-------|------|
| Bulk Create | BA | Draft ticket title, body, AC, UAT steps |
| Bulk Create | Estimator | Size each draft |
| Run Sprint | Coder | Implement on a feature branch |
| Run Sprint | Tester | Write/run tests, post report |
| Run Sprint | Documentor | Update changelog and docs |
| Run Sprint | Reviewer | Review diff, open follow-up tickets |
