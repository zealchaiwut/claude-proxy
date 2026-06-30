# Bulk Create — Prompt & Output Record

Durable record of every bulk-create batch run against claude-proxy. Each file holds
the prompts pasted into the Bulk Create tab plus the issues they produced, so a
batch can be reviewed, re-run, or copied later.

## Naming

```
docs/bulk-create/YYYY-MM-DD-<topic>.md
```

One file per batch.

## File format

```markdown
# <Batch title>

**Date:** YYYY-MM-DD
**Sprint label:** sprint-N (or NEW)
**Default labels:** ...
**Status:** drafted | posted | run

## Prompts

(one code block; prompts separated by `---`)

## Posted issues

| # | Title | Size |
|---|-------|------|
```
