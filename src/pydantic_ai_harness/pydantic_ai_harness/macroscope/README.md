# Macroscope

Run a [Macroscope](https://docs.macroscope.com/cli) code review from a Pydantic AI
agent and hand the findings back for validation and fixing.

[Source](https://github.com/pydantic/pydantic-ai-harness/tree/main/pydantic_ai_harness/macroscope/)

## The problem

Macroscope reviews the current branch's diff and streams findings, but it ships as
editor plugins (Claude Code, Codex, Cursor, OpenCode). There is no way to give a
Pydantic AI agent the same review-and-fix loop from your own code.

## The solution

`Macroscope` adds a `run_macroscope_review` tool that shells out to the installed
`macroscope codereview` CLI, parses the streamed findings, and returns them as a
structured `MacroscopeReview`. The agent then validates each finding and fixes the
real ones with whatever tools it already has (for example `FileSystem` or `Shell`).

```python
from pydantic_ai import Agent
from pydantic_ai_harness.macroscope import Macroscope

agent = Agent('anthropic:claude-sonnet-5', capabilities=[Macroscope()])

result = agent.run_sync('Run a Macroscope review and fix any real findings.')
print(result.output)
```

## Prerequisites: install and sign in

The capability drives the user-installed `macroscope` binary. It cannot install or
authenticate on your behalf, so do this once on the host:

1. Install the CLI:

   ```bash
   curl -sSL https://raw.githubusercontent.com/prassoai/macroscope-local/main/install.sh | bash
   ```

   The installer puts `macroscope` on your `PATH` (typically `~/.local/bin`).

2. Sign in and pick a workspace by running the CLI once and completing the wizard:

   ```bash
   macroscope
   ```

   This writes `~/.macroscope/config.yaml`.

The tool invokes `macroscope codereview --raw` for machine-readable streaming
output, which needs a recent CLI build. The installer fetches the latest and the
CLI self-updates on use, so a fresh install satisfies this.

If the binary is missing, the tool returns the install command. If a review never
starts (usually because you are not signed in), the tool tells the agent to run
`macroscope` to finish setup.

## The tool

| Tool | Purpose |
|---|---|
| `run_macroscope_review` | Run `macroscope codereview` on the current branch and return the review id, terminal status, and findings. Accepts an optional `base` git ref. |

Each finding is a `MacroscopeIssue` with `issue_id`, `sequence`, `path`, `line`,
`severity`, `category`, and `body`.

## Options

| Field | Default | Meaning |
|---|---|---|
| `base` | `None` | Git ref to diff against. `None` omits `--base` so the CLI auto-detects the base branch itself (and creates its own review worktree). A per-call `base` argument, then this field, take precedence when set. |
| `command` | `'macroscope'` | Binary name or path. Override for a non-default install location. |
| `cwd` | `'.'` | Repository directory the review runs in. |
| `timeout` | `600.0` | Maximum seconds to wait for a review. |
| `guidance` | `None` | Custom review guidance for the system prompt. `None` contributes the default validate-then-fix guidance; `''` contributes none. |

## Scope and composition

This capability surfaces findings only. It does not edit files, create worktrees, or
commit -- validating and fixing findings is the agent's job, using its other
capabilities. Pair it with `FileSystem` or `Shell` to let the agent read code and
apply fixes, and consider running the agent in an isolated worktree if you want fixes
kept off your working tree.
