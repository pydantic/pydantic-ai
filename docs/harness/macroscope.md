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

![The model calls run_macroscope_review, which runs macroscope codereview --raw in the run's workspace. The CLI uploads the branch diff to Macroscope's service and streams findings back, and the tool returns them as a MacroscopeReview with a review id, a status and a list of MacroscopeIssue findings. The agent then checks each finding and fixes the real ones with its own file and shell tools. Setup problems, such as a missing or signed-out CLI, raise UserError and stop the run.](../img/harness/macroscope.svg)

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

The capability cannot install or authenticate on your behalf. The model cannot
fix these setup problems either, so the tool raises `UserError` and the run
stops rather than spending tool retries on them. A missing binary reports the
install command (or names the path, when `command` is a path), a binary that
cannot be launched reports the OS error, and a review that never starts (usually
because you are not signed in) tells you to run `macroscope` to finish setup,
or to fix the configured `base` if the CLI reports an unknown ref. The
exception is a review that fails to start with a `base` the model passed: that
is reported to the model as a retry so it can drop or change the ref.

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

Every field of `Macroscope` with its default:

```python
from pydantic_ai_harness import Macroscope

Macroscope(
    base=None,             # git ref to diff against -- None lets the CLI auto-detect
    command='macroscope',  # binary name or path
    timeout=600.0,         # max seconds to wait for a review
    guidance=None,         # None = default instructions, '' = none, str = custom
    id='macroscope',       # capability and toolset id, see Two of them
)
```

A per-call `base` argument takes precedence over the field. Reviews call a
remote service, so the timeout is generous by default; on timeout the
workspace stops the CLI and the model gets the findings that arrived (see
[The tool](#the-tool)). The tool is not offered when the workspace cannot run
commands (for example a read-only one). A `base` that starts with `-` is
rejected. `command` cannot start with `-` or contain `=`; put the CLI on the
workspace's `PATH` instead.

## Scope and composition

This capability surfaces findings only: it does not edit files, create
worktrees, or commit. Pair it with `FileSystem` or `Shell` so the agent can
read code and apply fixes. To keep fixes off your working tree, give the agent
a separate worktree as its workspace. The tool returns a review's findings in
one result, which can be large; add [Tool Output Limits](tool-output-limits.md)
to cap its size.

## Durable execution

Under [durable execution](../durable_execution/overview.md) on Temporal, each
review runs in its own activity, which gets `timeout` plus 30 seconds. The
workspace therefore stops a slow review first, and the model still gets the
findings that arrived. To set a different activity limit, use
[`SetToolMetadata`][pydantic_ai.capabilities.SetToolMetadata] as in
[Temporal command timeouts](../durable_execution/temporal.md#harness-command-timeouts).

The activities are named after the capability's `id`, which defaults to
`macroscope`, so durable execution needs no configuration. Changing `id`
renames them, which in-flight runs then cannot find.

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

## Two of them

`Macroscope` declares the default `id` `'macroscope'`. The `id` keys the capability's instructions, so they
can be addressed, and lets `defer_loading=True` work without one. Two `Macroscope`s on one agent that
share it are one review configuration stated twice: they become one when they agree, and the agent raises a
`UserError` when it's built if they differ. To keep two, give each its own `id` and wrap them
in [PrefixTools](../capabilities/prefix-tools.md), since both register `run_macroscope_review`.

Earlier releases declared no default `id`, so two always failed on the tool name when the run
started. Two identical ones now become one, and two that differ fail when the agent is built.

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

- [The agent writes faster than you can review](https://pydantic.dev/articles/harness-macroscope) (Pydantic): from a ten-line self-critic to Macroscope's review
- [Macroscope CLI documentation](https://docs.macroscope.com/cli)
- [Pydantic AI capabilities](../capabilities/overview.md)
- [Toolsets](../toolsets.md)

## API reference

::: pydantic_ai_harness.macroscope.Macroscope

::: pydantic_ai_harness.macroscope.MacroscopeReview

::: pydantic_ai_harness.macroscope.MacroscopeIssue
