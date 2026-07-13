---
title: Shell
description: Give a Pydantic AI agent shell command execution with allow/deny controls, environment scrubbing, and managed background processes.
---

# Shell

`Shell` gives an agent the ability to run shell commands, with allow/deny
controls, environment scrubbing, and managed background processes. It exposes
command-execution tools rooted at a working directory and cleans up any
background processes automatically when the agent run ends.

[Source](https://github.com/pydantic/pydantic-ai-harness/tree/main/pydantic_ai_harness/shell/)

## The problem

Agents frequently need to run a build, a test suite, a linter, or a quick
`grep`. Wiring up subprocess handling -- streaming output, timeouts, truncation,
killing runaway processes, and cleaning up background jobs at the end of a run --
is fiddly boilerplate that every agent reinvents.

`Shell` bundles that plumbing into a single [capability](/ai/core-concepts/capabilities/):
configurable allow/deny lists, output truncation tuned to keep the useful tail,
optional sticky working directory, environment control that can keep host
secrets out of spawned commands, and automatic cleanup of background processes
when the run finishes.

## Usage

Construct `Shell` with a working directory and pass it to an `Agent` via the
`capabilities` parameter:

```python
from pydantic_ai import Agent
from pydantic_ai_harness import Shell

agent = Agent(
    'anthropic:claude-sonnet-4-6',
    capabilities=[Shell(cwd='./workspace', allowed_commands=['ls', 'cat', 'rg'])],
)

result = agent.run_sync('List the Python files and summarize the largest one.')
print(result.output)
```

By default `Shell` runs in the current directory with the built-in destructive-command
denylist active -- `Shell()` alone is a working (if permissive) configuration.

## Tools

`Shell` contributes four tools to the agent:

| Tool | Purpose |
|---|---|
| `run_command` | Run a command synchronously and return labelled stdout/stderr plus exit code. Honors a per-call or default timeout. |
| `start_command` | Launch a long-running command (server, watcher) in the background; returns an ID. |
| `check_command` | Report the status and accumulated output of a background command. |
| `stop_command` | Terminate a background command and return its final output. |

`run_command` accepts an optional `timeout_seconds` argument that overrides
`default_timeout` for a single call. `check_command` and `stop_command` take the
`command_id` string returned by `start_command`.

Output is labelled with `[stdout]` / `[stderr]` markers and an `[exit code: N]`
line on non-zero exit. When it exceeds `max_output_chars` the **tail** is kept
(the head is dropped), so errors, stack traces, and the `[stderr]` section --
which all land at the end -- survive truncation.

## Command controls

Two mutually exclusive lists decide which executables may run, plus filters for
shell operators and interactive commands:

| Field | Effect |
|---|---|
| `allowed_commands` | If non-empty, only these executables may run (allowlist). |
| `denied_commands` | These executables are always rejected (denylist). |
| `denied_operators` | Shell operators (e.g. `>`, `>>`, `\|`) that are rejected when present. |
| `allow_interactive` | If `False` (default), commands that expect a TTY (`vi`, `sudo`, `ssh`, ...) are blocked. |

`allowed_commands` and `denied_commands` are mutually exclusive -- set one, not
both. Setting both raises a `ValueError` at construction. `denied_commands`
defaults to a list of destructive commands (`rm`, `rmdir`, `mkfs`, `dd`,
`format`, `shutdown`, `reboot`, `halt`, `poweroff`, `init`); pass an empty list
to disable it. The executable name is extracted with `shlex`, so arguments don't
bypass the check.

A denied command surfaces to the model as a
[`ModelRetry`](/ai/tools-toolsets/tools-advanced/#tool-retries), not a hard error:
the run continues and the model can pick an allowed command instead.

!!! warning "Best-effort, not a security boundary"
    These command checks are best-effort. A sufficiently motivated agent can
    defeat them (e.g. `bash -c '...'`, env-var indirection). For hard
    guarantees, run the agent inside OS-level isolation -- a container or
    sandbox.

## Environment control

By default a spawned command inherits the agent process's full environment. In a
sandbox that holds LLM API keys, tokens, or other secrets, a command the model
writes can read them. Two fields control what the subprocess sees:

| Field | Effect |
|---|---|
| `env` | Explicit environment that replaces inheritance entirely. The subprocess sees exactly these variables and nothing else. |
| `denied_env_patterns` | Glob patterns (`fnmatch`) for variable names stripped from the base environment. Mirrors `denied_commands`. |

`env` is a hard boundary for inherited environment variables: set it and inherited secrets cannot reach the
subprocess at all (you supply `PATH` and anything else the command needs).
`denied_env_patterns` is a denylist over the inherited environment -- lighter to
configure when you only need to drop a few known-sensitive names. The two
compose: when both are set, patterns also filter the explicit `env`. Leaving
both unset preserves the inherit-everything default.

```python
import os

from pydantic_ai_harness import Shell
from pydantic_ai_harness.shell import LLM_API_KEY_ENV_PATTERNS

# Strip provider credentials from the inherited environment.
Shell(cwd='./repo', denied_env_patterns=LLM_API_KEY_ENV_PATTERNS)

# Or hand the subprocess a fixed environment, inheriting nothing.
Shell(cwd='./repo', env={'PATH': os.environ['PATH'], 'HOME': os.environ['HOME']})
```

`LLM_API_KEY_ENV_PATTERNS` covers common provider prefixes (`ANTHROPIC_*`,
`GATEWAY_*`, `GEMINI_*`, `GOOGLE_*`, `OPENAI_*`, `OPENROUTER_*`) plus
`PYDANTIC_AI_GATEWAY_API_KEY`. It targets LLM credentials only -- it does not
cover other host secrets (a `LOGFIRE_TOKEN`, a GitHub token, cloud
credentials), and its prefixes are coarse, so `GOOGLE_*` also strips
non-credential vars like `GOOGLE_APPLICATION_CREDENTIALS`. Treat it as a
starting point and add your own patterns. It is not the default: stripping
environment variables silently would break agents that rely on inherited
credentials, so it is opt-in.

`env` is enforced at spawn, not applied as a post-hoc filter on a running
process: the subprocess starts with exactly the resolved environment (your
`env`, minus anything `denied_env_patterns` removes from it). That makes it a
real boundary for inherited environment variables, unlike the best-effort command denylist. It is not a full
security boundary: a command running under the same OS identity can still read
host files -- use OS-level isolation for that. The flip side is that a
pattern broad enough to strip `PATH` or `HOME`, or an `env` that omits them, can
break command resolution. External commands may still run via the shell's
built-in default `PATH` on some systems, but don't rely on it -- set `PATH`
explicitly when you replace the environment.

## Background processes

`start_command` writes stdout/stderr to temp files and returns a short ID. Use
`check_command(command_id)` to poll and `stop_command(command_id)` to terminate
and collect final output. Processes are launched in their own session
(`start_new_session`) so the whole process group can be signalled -- `SIGTERM`,
escalating to `SIGKILL` after a grace period.

On run end, the toolset's cleanup terminates every still-running background
process and deletes its temp files. The agent runtime enters toolsets via an
`AsyncExitStack`, so this cleanup runs whether the run succeeds or raises -- an
agent that forgets to call `stop_command` won't leak processes.

```python
from pydantic_ai import Agent
from pydantic_ai_harness import Shell

agent = Agent(
    'anthropic:claude-sonnet-4-6',
    capabilities=[Shell(cwd='./app', allowed_commands=['npm', 'curl'])],
)

result = agent.run_sync(
    'Start the dev server with `npm run dev`, wait for it to boot, '
    'then curl http://localhost:3000/health and report the status.'
)
print(result.output)
```

## Working directory

By default each command runs in `cwd` and `cd` has no lasting effect. Set
`persist_cwd=True` to make `cd` sticky across calls: each command is wrapped so
that after it runs, its final working directory is recorded to a private temp
file, and that directory is carried into subsequent calls. The path is only
updated when the command exits `0`, and the record is written out-of-band (not
to stdout) so command output can never spoof the tracked directory.

```python
from pydantic_ai import Agent
from pydantic_ai_harness import Shell

agent = Agent(
    'anthropic:claude-sonnet-4-6',
    capabilities=[Shell(cwd='.', persist_cwd=True, allowed_commands=['cd', 'ls', 'pwd'])],
)
```

Each run gets a fresh toolset instance, so the tracked directory and any
background processes are isolated between concurrent runs and always start back
at the configured `cwd`.

## Configuration

Every field of `Shell` with its default:

```python
from pydantic_ai_harness import Shell

Shell(
    cwd='.',                       # str | Path -- working directory
    allowed_commands=[],           # allowlist (mutually exclusive with denied)
    denied_commands=[...],         # denylist (defaults to destructive commands)
    denied_operators=[],           # blocked shell operators
    default_timeout=30.0,          # seconds, per run_command
    max_output_chars=50_000,       # output cap returned to the model
    persist_cwd=False,             # make cd sticky across calls
    allow_interactive=False,       # allow TTY-style commands
    env=None,                      # explicit env, replacing inheritance (None = inherit)
    denied_env_patterns=[],        # glob patterns stripped from the env
)
```

## Agent spec (YAML/JSON)

`Shell` works with Pydantic AI's
[agent spec](/ai/core-concepts/agent-spec/), so you can declare it in a
config file instead of Python:

```yaml
# agent.yaml
model: anthropic:claude-sonnet-4-6
capabilities:
  - Shell:
      cwd: ./workspace
      allowed_commands: ['ls', 'cat', 'rg', 'pytest']
```

```python
from pydantic_ai import Agent
from pydantic_ai_harness import Shell

agent = Agent.from_file('agent.yaml', custom_capability_types=[Shell])
```

Pass `custom_capability_types` so the spec loader knows how to instantiate
`Shell`.

## Further reading

- [Pydantic AI capabilities](/ai/core-concepts/capabilities/)
- [Toolsets](/ai/tools-toolsets/toolsets/)

## API reference

::: pydantic_ai_harness.Shell
