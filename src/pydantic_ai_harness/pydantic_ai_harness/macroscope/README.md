# Macroscope

Run a [Macroscope](https://docs.macroscope.com/cli) code review from a Pydantic AI
agent and hand the findings back for validation and fixing.

[Source](https://github.com/pydantic/pydantic-ai/tree/main/src/pydantic_ai_harness/pydantic_ai_harness/macroscope/)

## The problem

Macroscope reviews the current branch's diff and streams findings, but it ships as
editor plugins (Claude Code, Codex, Cursor, OpenCode). This capability gives a
Pydantic AI agent the same review-and-fix loop.

## The solution

`Macroscope` adds a `run_macroscope_review` tool that runs the installed
`macroscope codereview` CLI in the run's workspace, parses the streamed
findings, and returns them as a structured `MacroscopeReview`. The agent then
validates each finding and fixes the real ones with whatever tools it already has
(for example `FileSystem` or `Shell`).

```python
from pydantic_ai import Agent
from pydantic_ai.capabilities import LocalWorkspace
from pydantic_ai_harness import Macroscope

agent = Agent(
    'anthropic:claude-sonnet-5',
    capabilities=[
        LocalWorkspace('.'),
        Macroscope(),
    ],
)

result = agent.run_sync('Run a Macroscope review and fix any real findings.')
print(result.output)
```

## Prerequisites: install and sign in

The capability drives the `macroscope` binary installed in the agent's
[workspace](https://pydantic.dev/docs/ai/core-concepts/workspace/): your machine
with `LocalWorkspace`, or the sandbox when the run uses one. A run without a
workspace fails at its start. The review runs in the workspace's working
directory, which should be the repository. The capability cannot install or
authenticate on your behalf, so do this once wherever the review runs:

1. Install the CLI:

   ```bash
   curl -sSL https://raw.githubusercontent.com/prassoai/macroscope-local/main/install.sh | bash
   ```

   The installer puts `macroscope` on your `PATH` (typically `~/.local/bin`).

2. Sign in and pick a Macroscope workspace by running the CLI once and completing the wizard:

   ```bash
   macroscope
   ```

   This writes `~/.macroscope/config.yaml`.

The tool runs `macroscope codereview --raw`. Builds without `--raw` are too old;
the CLI updates itself on use.

The model cannot fix setup problems either, so they end the run with `UserError`:

| Cause | Message |
|---|---|
| CLI not installed | the install command, or the configured path when `command` is one |
| CLI cannot launch | the launcher's error |
| Review never starts | run `macroscope` to sign in, or fix the configured `base` |

If the review fails to start with a `base` the model chose, the model gets a
retry instead, so it can change or drop the ref.

## The tool

| Tool | Purpose |
|---|---|
| `run_macroscope_review` | Run `macroscope codereview` on the current branch and return the review id, terminal status, and findings. Accepts an optional `base` git ref. |

Each finding is a `MacroscopeIssue` with `issue_id`, `sequence`, `path`, `line`,
`severity`, `category`, and `body`.

`status` is the status the CLI reports, such as `completed` or `failed`. When the
CLI stops before reporting one, the tool still returns the findings that arrived
and sets `status` to say why: `timed_out` when the review hit `timeout`, or
`interrupted` when the CLI exited with an error. If a review times out before it
starts, the model gets a retry.

## Options

| Field | Default | Meaning |
|---|---|---|
| `base` | `None` | Git ref to diff against. `None` omits `--base` so the CLI auto-detects the base branch itself (and creates its own review worktree). The model's per-call `base` overrides this field. A `base` that starts with `-` is rejected. |
| `command` | `'macroscope'` | Binary name or path. Override for a non-default install location. Cannot start with `-` or contain `=`. |
| `timeout` | `600.0` | Maximum seconds to wait for a review. |
| `guidance` | `None` | Custom review guidance for the system prompt. `None` contributes the default validate-then-fix guidance; `''` contributes none. |
| `id` | `None` | Toolset id for durable execution. `None` uses `'macroscope'`. |

## Scope and composition

Pair it with `FileSystem` or `Shell` so the agent can read code and apply fixes. To
keep fixes off your working tree, give the agent a separate worktree as its
workspace. A large review returns every finding in one tool result; add
`ToolOutputLimits` to cap its size.

## Durable execution

Under [durable execution](https://pydantic.dev/docs/ai/capabilities/durable_execution/overview/) on Temporal, each review runs in its own activity, which
gets `timeout` plus 30 seconds. The workspace therefore stops a slow review first,
and the model still gets the findings that arrived. To set a different activity
limit, use `SetToolMetadata` as in
[Temporal command timeouts](https://pydantic.dev/docs/ai/capabilities/durable_execution/temporal/#harness-command-timeouts).

The activities are named after the capability's `id`, which defaults to
`macroscope`, so durable execution needs no configuration. Changing `id` renames
them, which in-flight runs then cannot find.

## Data and trust

The review uploads the branch diff to Macroscope's service. Finding bodies are
written by Macroscope's model from that diff, so on a branch you do not trust,
treat them as untrusted input: run the agent in a sandbox workspace, or require
approval for its shell and write tools.

## Telemetry

`Macroscope` emits no spans of its own. Core's instrumentation records each
`run_macroscope_review` call as a tool span, and the capability makes no decision
worth a span of its own.
