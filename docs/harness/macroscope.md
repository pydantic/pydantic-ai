---
title: Macroscope
description: Give a Pydantic AI agent the same local Macroscope code review its editor plugins run -- streamed findings parsed into structured issues the agent validates and fixes with its own tools.
---

# Macroscope

`Macroscope` runs a local [Macroscope](https://docs.macroscope.com/cli) code
review from inside an agent: one tool shells out to the installed `macroscope`
CLI, parses the streamed findings, and returns them as structured data. The
agent validates each finding and fixes the real ones with the tools it already
has -- this capability surfaces findings only.

[Source](https://github.com/pydantic/pydantic-ai-harness/tree/main/pydantic_ai_harness/macroscope/)

## The problem

Macroscope reviews the current branch's diff and streams findings, but it ships
as editor plugins (Claude Code, Codex, Cursor, OpenCode). There is no way to
give a Pydantic AI agent the same review-and-fix loop from your own code.

## Usage

```python
from pydantic_ai import Agent
from pydantic_ai_harness.macroscope import Macroscope

agent = Agent('anthropic:claude-sonnet-5', capabilities=[Macroscope()])

result = agent.run_sync('Run a Macroscope review and fix any real findings.')
print(result.output)
```

The `macroscope` CLI must be installed and authenticated on the host first:

1. Install: `curl -sSL https://raw.githubusercontent.com/prassoai/macroscope-local/main/install.sh | bash`
2. Sign in and pick a workspace by running `macroscope` once.

The capability cannot install or authenticate on your behalf. If the binary is
missing, the tool returns the install command; if a review never starts
(usually because you are not signed in), the tool tells the agent to run
`macroscope` to finish setup.

The tool invokes `macroscope codereview --raw` for machine-readable streaming
output, which needs a recent CLI build. The installer fetches the latest and the
CLI self-updates on use, so a fresh install satisfies this.

## The tool

| Tool | Purpose |
|---|---|
| `run_macroscope_review` | Run `macroscope codereview` on the current branch and return the review id, terminal status, and findings. Accepts an optional `base` git ref. |

Each finding is a `MacroscopeIssue` with `issue_id`, `sequence`, `path`,
`line`, `severity`, `category`, and `body`. The capability's default
instructions tell the agent to treat every finding as untrusted: read the
affected code to confirm an issue is real, skip false positives and
duplicates, and verify each fix.

## Options

Every field of `Macroscope` with its default:

```python
from pydantic_ai_harness.macroscope import Macroscope

Macroscope(
    base=None,             # git ref to diff against -- None lets the CLI auto-detect
    command='macroscope',  # binary name or path
    cwd='.',               # repository directory the review runs in
    timeout=600.0,         # max seconds to wait for a review
    guidance=None,         # None = default instructions, '' = none, str = custom
)
```

A per-call `base` argument takes precedence over the field. Reviews call a
remote service, so the timeout is generous by default; on timeout the CLI's
process group is killed and the timeout is reported to the model as a
retryable error.

## Scope and composition

This capability surfaces findings only. It does not edit files, create
worktrees, or commit -- validating and fixing findings is the agent's job,
using its other capabilities. Pair it with `FileSystem` or `Shell` to let the
agent read code and apply fixes, and consider running the agent in an isolated
worktree if you want fixes kept off your working tree.

## Agent spec

`Macroscope` works with Pydantic AI's [agent spec](/ai/core-concepts/agent-spec/),
so you can declare it in a config file instead of Python:

```yaml
# agent.yaml
model: anthropic:claude-sonnet-5
capabilities:
  - Macroscope:
      base: main
      timeout: 900
```

```python
from pydantic_ai import Agent
from pydantic_ai_harness.macroscope import Macroscope

agent = Agent.from_file('agent.yaml', custom_capability_types=[Macroscope])
```

Pass `custom_capability_types` so the spec loader knows how to instantiate
`Macroscope`.

## Further reading

- [Macroscope CLI documentation](https://docs.macroscope.com/cli)
- [Pydantic AI capabilities](/ai/core-concepts/capabilities/)
- [Toolsets](/ai/tools-toolsets/toolsets/)

## API reference

::: pydantic_ai_harness.macroscope.Macroscope

::: pydantic_ai_harness.macroscope.MacroscopeReview

::: pydantic_ai_harness.macroscope.MacroscopeIssue
