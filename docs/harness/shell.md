---
title: Shell
description: Give a Pydantic AI agent shell command execution with allow/deny controls, environment scrubbing, and managed background processes.
---

# Shell

`Shell` gives an agent the ability to run shell commands, with allow/deny
controls, environment scrubbing, and managed background processes. It exposes
command-execution tools that run in the agent's workspace, with background
processes that keep running across the runs of a conversation.

[Source](https://github.com/pydantic/pydantic-ai/tree/main/src/pydantic_ai_harness/pydantic_ai_harness/shell/)

> While Pydantic AI Harness is on 0.x releases, the API may change between minor releases; when it does, deprecation warnings and release-note migration guidance tell you (or your agent) exactly how to upgrade. See the [version policy](index.md#version-policy).

## The problem

Agents frequently need to run a build, a test suite, a linter, or a quick
`grep`. Wiring up subprocess handling -- streaming output, timeouts, truncation,
killing runaway processes, and tracking background jobs across runs --
is fiddly boilerplate that every agent reinvents.

`Shell` bundles that plumbing into a single [capability](../capabilities/overview.md):
configurable allow/deny lists, output truncation tuned to keep the useful tail,
optional sticky working directory, environment control that can keep host
secrets out of spawned commands, and background processes the model can check
and stop by ID.

## Usage

Pass `Shell` to an `Agent` via the `capabilities` parameter, together with a
workspace for the commands to run in:

```python
from pathlib import Path

from pydantic_ai import Agent
from pydantic_ai.capabilities import LocalWorkspace
from pydantic_ai_harness import Shell

Path('./workspace').mkdir(exist_ok=True)
agent = Agent(
    'anthropic:claude-opus-5-5',
    capabilities=[
        LocalWorkspace('./workspace'),
        Shell(allowed_commands=['ls', 'cat', 'rg']),
    ],
)

result = agent.run_sync('List the Python files and summarize the largest one.')
print(result.output)
```

`Shell()` alone works, with the built-in destructive-command denylist on.

## Where commands run

Commands run in the run's [workspace](https://pydantic.dev/docs/ai/core-concepts/workspace/):
on your machine with `LocalWorkspace`, or in a sandbox. A run without a
workspace fails at its start. Commands start in the workspace's working
directory; to run them in a subdirectory, set it on the workspace:
`LocalWorkspace('./repo')`. A read-only workspace can't run
commands, so `Shell` offers no tools for that run.

## Tools

`Shell` contributes four tools by default, plus the opt-in persistent `shell` tool:

| Tool | Purpose |
|---|---|
| `run_command` | Run a command synchronously and return labelled stdout/stderr plus exit code. Honors a per-call or default timeout. |
| `start_command` | Launch a long-running command (server, watcher) in the background; returns an ID. |
| `check_command` | Report the status and accumulated output of a background command. |
| `stop_command` | Terminate a background command and return its final output. |
| `shell` | Opt-in: run a command in `foreground` (wait up to `timeout`, then hand back the still-running process) or `background` mode. Returns the PID and the paths of its output log and JSON status file. |

`run_command` accepts an optional `timeout_seconds` argument that overrides
`default_timeout` for a single call. `check_command` and `stop_command` take the
`command_id` string returned by `start_command`.

Output is labelled with `[stdout]` / `[stderr]` markers and an `[exit code: N]`
line on non-zero exit. When it exceeds `max_output_chars` the **tail** is kept
(the head is dropped), so errors, stack traces, and the `[stderr]` section --
which all land at the end -- survive truncation. Background command status and
exit metadata follow the captured output so they remain in the retained tail.

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
both. Setting non-empty values for both raises a `ValueError` when the toolset
is constructed. `denied_commands` defaults to a list of destructive commands
(`rm`, `rmdir`, `mkfs`, `dd`, `format`, `shutdown`, `reboot`, `halt`,
`poweroff`, `init`); pass an empty list to disable it. The executable name is
extracted with `shlex`, so arguments don't bypass the check.

An empty `allowed_commands` collection does not select allowlist mode. The
configured `denied_commands` remain active; when omitted, this is the built-in
denylist. Pass `denied_commands=[]` to disable command-name filtering.

A denied command surfaces to the model as a
[`ModelRetry`](../tools-advanced.md#tool-retries), not a hard error:
the run continues and the model can pick an allowed command instead. So does
every other failure the model can act on: a working directory an earlier command
deleted or replaced with a file, and a command the operating system refuses to
spawn because it holds a NUL byte or contains a character the operating system
cannot encode. Failures
the model can do nothing about still abort the run: a host that cannot allocate
a process, an argument or environment that exceeds the platform's combined
size limit, and an invalid character in an application-supplied `env`. A command that runs past its timeout returns its output so far, ending in
`[Command timed out after Ns]`. A workspace that is gone ends the run.

!!! warning "Best-effort, not a security boundary"
    `allowed_commands` is a guardrail against accidents, not a security boundary.
    Validation checks only the first token, and allowlisted commands such as
    `python`, `git`, `uv`, and `make` can spawn arbitrary processes. A model that
    wants to work around the allowlist can. For untrusted work, give the run an
    isolated workspace, such as a [Modal sandbox](modal-sandbox.md) or a container,
    so commands run there instead of on the agent's machine.

## Limit files written by commands

Set `Shell(max_file_bytes=10_000_000)` to bound the size of each regular file
written by `run_command` and `start_command`, including redirected output and
background stdout/stderr logs. `None` (the default) adds no limit. This is
separate from `max_output_chars`, which bounds the returned tool result.

The limit covers the command and every process it starts, rounded up to whole
512-byte or 1 KiB blocks. If the workspace's own limit is lower, the command
fails with `Unable to apply max_file_bytes.` It can't be combined with
`persist_cwd=True` or `tools=['shell']`.

A process killed by the file-size signal gets a diagnosed tool result; other
nonzero exits include the configured limit as context because programs can
catch the write error and choose their own exit code. The agent can reduce its
output and retry. Background failures appear in `check_command` or
`stop_command`. Commands that handle the error and exit successfully cannot be
diagnosed from their exit status. Results still obey `max_output_chars`.

This is a per-file bound, not a disk quota or a sandbox: it does not remove
partial files, shrink existing files, or prevent creating many smaller files.
Use filesystem quotas or OS isolation for aggregate disk protection and
untrusted commands. No additional telemetry spans are emitted; the existing
tool-call result carries the failure and limit context.

## Environment control

A command gets the workspace's environment plus `Shell(env=...)`.
`LocalWorkspace` passes on your `PATH`, `HOME`, `LANG`, `LC_ALL` and `LC_CTYPE`
and nothing else; a sandbox has whatever its provider configures. Two fields shape what `Shell` adds:

| Field | Effect |
|---|---|
| `env` | Variables added to every command's environment, on top of the workspace's own. |
| `denied_env_patterns` | Glob patterns (`fnmatch`) for variable names dropped from `env`. Mirrors `denied_commands`. |

`denied_env_patterns` filters `Shell(env=)` only, not `LocalWorkspace(env=)`, so
you can build `env` from a larger mapping and drop sensitive names on the way in.
Leaving both unset adds nothing.

```python
from pydantic_ai_harness import LLM_API_KEY_ENV_PATTERNS, Shell

# Add only the variables commands need.
Shell(env={'PYTHONUNBUFFERED': '1'})

# Or start from a larger mapping and drop LLM credentials from it (other secrets still pass).
app_env = {'PYTHONUNBUFFERED': '1', 'ANTHROPIC_API_KEY': 'sk-...'}
Shell(env=app_env, denied_env_patterns=LLM_API_KEY_ENV_PATTERNS)
```

`LLM_API_KEY_ENV_PATTERNS` covers common provider prefixes (`ANTHROPIC_*`,
`GATEWAY_*`, `GEMINI_*`, `GOOGLE_*`, `OPENAI_*`, `OPENROUTER_*`) plus
`PYDANTIC_AI_GATEWAY_API_KEY`. It targets LLM credentials only -- it does not
cover other host secrets (a `LOGFIRE_TOKEN`, a GitHub token, cloud
credentials), and its prefixes are coarse, so `GOOGLE_*` also strips
non-credential vars like `GOOGLE_APPLICATION_CREDENTIALS`. Treat it as a
starting point and add your own patterns.

Neither is a security boundary: with `LocalWorkspace`, a command can still read
your process's environment and your files. Use a sandbox for untrusted commands.

## Background processes

`start_command` starts the command in the background inside the workspace and
returns an ID. Its output is logged under `.pydantic-ai-harness/shell/` in
the working directory, which is git-ignored. Use `check_command(command_id)` to
poll and `stop_command(command_id)` to terminate and collect final output: the
whole process group gets `SIGTERM`, then `SIGKILL` after a grace period. A
command stopped by `SIGTERM` reports `[exit code: 143]`.

A background command is not tied to the run: a conversation spans several
runs, so a later run on the same workspace can check or stop it by its ID. It
keeps running until it exits, `stop_command` stops it, or the workspace ends.

```python
from pydantic_ai import Agent
from pydantic_ai.capabilities import LocalWorkspace
from pydantic_ai_harness import Shell

agent = Agent(
    'anthropic:claude-opus-5-5',
    capabilities=[
        LocalWorkspace('./app'),
        Shell(allowed_commands=['npm', 'curl']),
    ],
)

result = agent.run_sync(
    'Start the dev server with `npm run dev`, wait for it to boot, '
    'then curl http://localhost:3000/health and report the status.'
)
print(result.output)
```

## Persistent commands

Name `shell` in `tools` to register the persistent tool instead of the four
tools above:

```python
from pydantic_ai_harness import Shell

Shell(tools=['shell'])
```

`shell(command, mode='foreground', timeout=None)` starts the command under a
small supervisor that writes its combined output to a log and its exit status to
a JSON file (`{"pid": ..., "exit_code": ...}`, `exit_code` null while it runs).
The tool returns the PID and both paths, so the model reads progress with its
file tools and stops the command with the `kill` command the result names. Foreground waits up to
`timeout` seconds (default `default_timeout`, at most `MAX_FOREGROUND_WAIT`,
270) for the exit status and returns the last 16,000 bytes of the log followed
by the handles, even if the command is still running. When the log is longer,
the result starts at a line and says how many earlier bytes it leaves out; background returns the
handles at once. The handles come last so that `max_output_chars`, which keeps
the tail of an over-long result, cannot drop them. The 270-second cap keeps a tool call shorter than typical provider
request timeouts, so a long build or test run does not stall the conversation:
the model gets the handles back, does other work, and polls the status file.

The command outlives the agent run and the event loop, so a server the model
starts keeps serving for as long as the workspace does. Nothing wakes the agent
when the command finishes; the model polls. Cancelling a foreground call kills the command. Logs that were handed back are
never deleted: clean them up yourself, and keep verbose commands' output
bounded.

`allowed_commands`, `denied_commands`, `denied_operators`, `allow_interactive`,
`env`, and `denied_env_patterns` apply to `shell` exactly as to `run_command`.
`persist_cwd` does not: every `shell` command starts in the workspace's working
directory, whatever `run_command` has tracked. Commands and logs live in the workspace; the
workspace calls that start and poll them are not replay-safe.

Each `shell` call emits progress events in the `shell` namespace, so a UI can
show output as it arrives without parsing the tool result:

| Event | Dispatch | Payload |
|---|---|---|
| `CommandStartedEvent` | stream | `command`, `pid` |
| `CommandOutputEvent` | stream | `text`: a chunk of the combined log, decoded incrementally |
| `CommandFinishedEvent` | stream | `pid`, `output_path`, `status_path`, `exit_code`, `truncated`, `total_lines` |

Output events are emitted while a foreground call waits: at most the first
16,000 bytes of the log per call, in chunks of up to 4,096 bytes, polled every
50 ms to 1 s. A background call emits only the started and finished events. `finished`
means the tool stopped waiting, not that the command exited: `exit_code` is
`None` while no status has been published, and `truncated` says the log held
more than the events showed. `total_lines` counts logical lines in logs up to
1 MiB and is `None` for larger logs, which are not scanned. A cancelled call may
emit no finished event. Events carry command text and command output, so treat
them as untrusted when rendering. They add no telemetry spans; core already
traces the tool call.

## Working directory

By default each command runs in the workspace's working directory and `cd` has
no lasting effect. Set
`persist_cwd=True` to make `cd` sticky across calls: each command is wrapped so
that after it runs, its final working directory is recorded to a private file
inside the workspace, and that directory is carried into subsequent calls. The path is only
updated when the command exits `0`. If the recorded directory disappears, the next command reports a retry and resets to the workspace working directory. The record is written out-of-band (not
to stdout) so command output can never spoof the tracked directory.

```python
from pydantic_ai import Agent
from pydantic_ai.capabilities import LocalWorkspace
from pydantic_ai_harness import Shell

agent = Agent(
    'anthropic:claude-opus-5-5',
    capabilities=[LocalWorkspace('.'), Shell(persist_cwd=True, allowed_commands=['cd', 'ls', 'pwd'])],
)
```

Each run gets a fresh toolset instance, so the tracked directory is isolated
between concurrent runs and always starts back at the workspace's working
directory.

`stop_command` signals the entire process group, including children left after the wrapper exits, then removes the job's output files. In a workspace without `setsid` where the job cannot get a process group of its own, only the wrapper process is signalled, which does not stop the command or its children.

Background commands and the `shell` tool keep their status and output in files, which needs `mv` and `base64` on the workspace's `PATH`. Without them the call fails at once and names the missing tool.

The model sees a capped preview of command output. For large output, redirect it to a file in the workspace, then use `grep` or `tail` to inspect bounded portions rather than printing the whole file.

## Configuration

Every field of `Shell` with its default:

```python {names="defined"}
from pydantic_ai_harness.shell import RUN_SCOPED_TOOL_NAMES, Shell

Shell(
    allowed_commands=[],           # allowlist (mutually exclusive with denied)
    denied_commands=[...],         # denylist (defaults to destructive commands)
    denied_operators=[],           # blocked shell operators
    default_timeout=30.0,          # seconds, per run_command
    max_output_chars=50_000,       # output cap returned to the model
    max_file_bytes=None,           # per-file size limit for commands (None = no limit)
    persist_cwd=False,             # make cd sticky across calls
    allow_interactive=False,       # allow TTY-style commands
    env=None,                      # variables added to the workspace's environment
    denied_env_patterns=[],        # glob patterns dropped from env
    tools=RUN_SCOPED_TOOL_NAMES,   # which tools to register ('shell' for persistent commands)
)
```

With `tools=['shell']`, `default_timeout` is also the foreground wait and must be
greater than zero and at most 270 seconds; that is checked at construction.

## Agent spec (YAML/JSON)

`Shell` works with Pydantic AI's
[agent spec](../agent-spec.md), so you can declare it in a
config file instead of Python:

```yaml
# agent.yaml
model: anthropic:claude-opus-5-5
capabilities:
  - Shell:
      allowed_commands: ['ls', 'cat', 'rg', 'pytest']
```

```python
from pydantic_ai import Agent
from pydantic_ai_harness import Shell

agent = Agent.from_file('agent.yaml', custom_capability_types=[Shell])
```

Pass `custom_capability_types` so the spec loader knows how to instantiate
`Shell`, and attach a workspace to the run (`workspace=` on the run method, or a
workspace capability in Python).

## Durable execution

`Shell` works under Temporal, DBOS, and Prefect, including background jobs and `persist_cwd`. [Durable execution](durable-execution.md) shows an example for each engine, which command events Temporal delivers live, and how long a command can run in a Temporal activity.

## Further reading

- [Pydantic AI capabilities](../capabilities/overview.md)
- [Toolsets](../toolsets.md)

## API reference

::: pydantic_ai_harness.Shell
