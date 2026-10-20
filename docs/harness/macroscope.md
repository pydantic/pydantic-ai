---
title: Macroscope
description: "Run a local Macroscope code review from a Pydantic AI agent and get structured findings it can verify and fix with its own file and shell tools."
---

# Macroscope

`Macroscope` runs a local [Macroscope](https://docs.macroscope.com/cli) code
review from inside an agent: one tool runs the installed `macroscope` CLI in
the run's workspace, parses the streamed findings, and returns them as
structured data. The agent validates each finding and fixes the real ones with
the tools it already has -- this capability surfaces findings only.

[Source](https://github.com/pydantic/pydantic-ai/tree/main/src/pydantic_ai_harness/pydantic_ai_harness/macroscope/)

> While Pydantic AI Harness is on 0.x releases, the API may change between minor releases; when it does, deprecation warnings and release-note migration guidance tell you (or your agent) exactly how to upgrade. See the [version policy](index.md#version-policy).

## The problem

Macroscope reviews the current branch's diff and streams findings, but it ships
as editor plugins (Claude Code, Codex, Cursor, OpenCode). This capability gives
a Pydantic AI agent the same review-and-fix loop.

## Usage

```python
from pydantic_ai import Agent
from pydantic_ai.capabilities import LocalWorkspace
from pydantic_ai_harness import Macroscope

agent = Agent(
    '{{ anthropic:medium }}',
    capabilities=[
        LocalWorkspace('.'),
        Macroscope(),
    ],
)

result = agent.run_sync('Run a Macroscope review and fix any real findings.')
print(result.output)
```

The review runs in the agent's [workspace](https://pydantic.dev/docs/ai/core-concepts/workspace/),
in its working directory, which should be the repository: your machine with
`LocalWorkspace`, or the sandbox when the run uses one. A run without a workspace fails at its
start. The `macroscope` CLI must be installed and authenticated in the
workspace first:

1. Install: `curl -sSL https://raw.githubusercontent.com/prassoai/macroscope-local/main/install.sh | bash`
2. Sign in and pick a Macroscope workspace by running `macroscope` once.

The capability cannot install or authenticate on your behalf, and the model
cannot fix setup problems either, so they end the run with `UserError`:

| Cause | Message |
|---|---|
| CLI not installed | the install command, or the configured path when `command` is one |
| CLI cannot launch | the launcher's error |
| Review never starts | run `macroscope` to sign in, or fix the configured `base` |

If the review fails to start with a `base` the model chose, the model gets a
retry instead, so it can change or drop the ref.

The tool runs `macroscope codereview --raw`. Builds without `--raw` are too old;
the CLI updates itself on use.

## The tool

| Tool | Purpose |
|---|---|
| `run_macroscope_review` | Run `macroscope codereview` on the current branch and return the review id, terminal status, and findings. Accepts an optional `base` git ref. |

Each finding is a `MacroscopeIssue` with `issue_id`, `sequence`, `path`,
`line`, `severity`, `category`, and `body`. The capability's default
instructions tell the agent to treat every finding as untrusted: read the
affected code to confirm an issue is real, skip false positives and
duplicates, and verify each fix.

`status` is the status the CLI reports, such as `completed` or `failed`. When
the CLI stops before reporting one, the tool still returns the findings that
arrived and sets `status` to say why: `timed_out` when the review hit
`timeout`, or `interrupted` when the CLI exited with an error. If a review times
out before it starts, the model gets a retry.

## Options

The fields of `Macroscope` and their defaults (`id` is inherited from
`AbstractCapability`):

```python
from pydantic_ai_harness import Macroscope

Macroscope(
    base=None,             # git ref to diff against -- None lets the CLI auto-detect
    command='macroscope',  # binary name or path
    timeout=600.0,         # max seconds to wait for a review
    guidance=None,         # None = default instructions, '' = none, str = custom
    id=None,               # toolset id for durable execution; None uses 'macroscope'
)
```

- The model's per-call `base` overrides the field. A `base` that starts with
  `-` is rejected, so it cannot reach the CLI as an option.
- `command` cannot start with `-` or contain `=`; put the CLI on the
  workspace's `PATH` instead.
- On timeout, the workspace stops the CLI (see [The tool](#the-tool) for what
  the model gets back).
- The tool is not offered when the workspace cannot run commands, such as a
  read-only one.

## Scope and composition

Pair it with `FileSystem` or `Shell` so the agent can read code and apply
fixes. To keep fixes off your working tree, give the agent a separate worktree
as its workspace. A large review returns every finding in one tool result; add
[Tool Output Limits](tool-output-limits.md) to cap its size.

## Data and trust

The review uploads the branch diff to Macroscope's service. Finding bodies are
written by Macroscope's model from that diff, so on a branch you do not trust,
treat them as untrusted input: run the agent in a sandbox workspace, or require
[approval](../deferred-tools.md) for its shell and write tools.

## Telemetry

`Macroscope` emits no spans of its own. Core's
[instrumentation](../capabilities/instrumentation.md) records each
`run_macroscope_review` call as a tool span, and the capability makes no
decision worth a span of its own.

## Agent spec

`Macroscope` works with Pydantic AI's [agent spec](../agent-spec.md),
so you can declare it in a config file instead of Python:

```yaml
# agent.yaml
model: {{ anthropic:medium }}
capabilities:
  - Macroscope:
      base: main
      timeout: 900
```

```python
from pydantic_ai import Agent
from pydantic_ai_harness import Macroscope

agent = Agent.from_file('agent.yaml', custom_capability_types=[Macroscope])
```

Pass `custom_capability_types` so the spec loader knows how to instantiate
`Macroscope`.

## Further reading

- [Macroscope CLI documentation](https://docs.macroscope.com/cli)
- [Pydantic AI capabilities](../capabilities/overview.md)
- [Toolsets](../toolsets.md)

## API reference

::: pydantic_ai_harness.macroscope.Macroscope

::: pydantic_ai_harness.macroscope.MacroscopeReview

::: pydantic_ai_harness.macroscope.MacroscopeIssue
