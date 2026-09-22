# Shell

Give an agent the ability to run shell commands, with allow/deny controls and
managed background processes.

[Source](https://github.com/pydantic/pydantic-ai-harness/tree/main/pydantic_ai_harness/shell/)

## The problem

Agents frequently need to run a build, a test suite, a linter, or a quick
`grep`. Wiring up subprocess handling -- streaming output, timeouts, truncation,
killing runaway processes, and cleaning up background jobs at the end of a run --
is fiddly boilerplate that every agent reinvents.

## The solution

`Shell` exposes command-execution tools rooted at a working directory, with
configurable allow/deny lists and automatic cleanup of background processes
when the agent run ends.

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

## Tools

| Tool | Purpose |
|---|---|
| `run_command` | Run a command synchronously and return labelled stdout/stderr plus exit code. Honors a per-call or default timeout. |
| `start_command` | Launch a long-running command (server, watcher) in the background; returns an ID. |
| `check_command` | Report the status and accumulated output of a background command. |
| `stop_command` | Terminate a background command and return its final output. |
| `shell` | Opt-in: run a command that outlives the run, in `foreground` (wait up to `timeout`, then hand back the still-running process) or `background` mode. Returns the PID and the paths of its output log and JSON status file. |

Output is labelled with `[stdout]` / `[stderr]` markers and an `[exit code: N]`
line on non-zero exit. When it exceeds `max_output_chars` the **tail** is kept
(the head is dropped), so errors, stack traces, and the `[stderr]` section --
which all land at the end -- survive truncation. Background command status and
exit metadata follow the captured output so they remain in the retained tail.

## Command controls

| Field | Effect |
|---|---|
| `allowed_commands` | If non-empty, only these executables may run (allowlist). |
| `denied_commands` | These executables are always rejected (denylist). |
| `denied_operators` | Shell operators (e.g. `>`, `>>`, `|`) that are rejected when present. |
| `allow_interactive` | If `False` (default), commands that expect a TTY (`vi`, `sudo`, `ssh`, ...) are blocked. |

`allowed_commands` and `denied_commands` are mutually exclusive -- set one, not
both. `denied_commands` defaults to a list of destructive commands (`rm`,
`rmdir`, `mkfs`, `dd`, `format`, `shutdown`, `reboot`, `halt`, `poweroff`,
`init`); pass an empty list to disable. The executable name is extracted with
`shlex`, so arguments don't bypass the check.

An empty `allowed_commands` collection does not select allowlist mode. The
configured `denied_commands` remain active; when omitted, this is the built-in
denylist. Pass `denied_commands=[]` to disable command-name filtering.

A denied or blocked command surfaces to the model as a `ModelRetry` (the model
can retry with an allowed command) rather than aborting the run. So does every
other failure the model can act on: a working directory an earlier command
deleted or replaced with a file, and a command the operating system refuses to
spawn because it holds a NUL byte or contains a character the operating system
cannot encode. Failures
the model can do nothing about still abort the run: a host that cannot allocate
a process, an argument or environment that exceeds the platform's combined
size limit, and an invalid character in an application-supplied `env`.

> **These checks are best-effort, not a security boundary.** `allowed_commands`
> is a guardrail against accidents, not a security boundary. Validation checks
> only the first token, and allowlisted commands such as `python`, `git`, `uv`,
> and `make` can spawn arbitrary processes. A model that wants to work around
> the allowlist can. For untrusted work, run the agent inside OS-level isolation
> such as [`ModalSandbox`](https://pydantic.dev/docs/ai/harness/modal-sandbox/) or a container.

## Limit files written by commands

Set `Shell(max_file_bytes=10_000_000)` to bound the size of each regular file
written by `run_command` and `start_command`, including redirected output and
background stdout/stderr logs. `None` (the default) adds no limit. This is
separate from `max_output_chars`, which bounds the returned tool result.

The positive integer is applied as POSIX `RLIMIT_FSIZE` in a child launcher
before executing the shell. Descendants inherit it; the harness parent's limits
are unchanged. A lower inherited hard limit takes precedence. Unsupported
platforms, `persist_cwd=True`, and `tools=['shell']` (the persistent tool) raise
`ValueError` when the toolset is constructed rather than ignoring the setting.
Working-directory persistence uses a child-written capture file, which would
also be subject to the limit.

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

By default a spawned command inherits the agent process's full environment. In a
sandbox that holds LLM API keys, tokens, or other secrets, a command the model
writes can read them. Two fields control what the subprocess sees:

| Field | Effect |
|---|---|
| `env` | Explicit environment that replaces inheritance for the subprocess's own environment. |
| `denied_env_patterns` | Glob patterns (`fnmatch`) for variable names stripped from the base environment. Mirrors `denied_commands`. |

`env` prevents inherited variables from appearing in the subprocess's own
environment (you supply `PATH` and anything else the command needs).
`denied_env_patterns` is a denylist over the inherited environment -- lighter to
configure when you only need to drop a few known-sensitive names. The two
compose: when both are set, patterns also filter the explicit `env`. Leaving
both unset preserves the inherit-everything default.

```python
from pydantic_ai_harness import LLM_API_KEY_ENV_PATTERNS, Shell

# Strip provider credentials from the inherited environment.
Shell(cwd='./repo', denied_env_patterns=LLM_API_KEY_ENV_PATTERNS)

# Or hand the subprocess a fixed environment, inheriting nothing.
import os
Shell(cwd='./repo', env={'PATH': os.environ['PATH'], 'HOME': os.environ['HOME']})
```

`LLM_API_KEY_ENV_PATTERNS` covers common provider prefixes (`ANTHROPIC_*`,
`OPENAI_*`, `OPENROUTER_*`, `GOOGLE_*`, `GEMINI_*`, `GATEWAY_*`) plus
`PYDANTIC_AI_GATEWAY_API_KEY`. It targets LLM credentials only -- it does not
cover other host secrets (a `LOGFIRE_TOKEN`, a GitHub token, cloud
credentials), and its prefixes are coarse, so `GOOGLE_*` also strips
non-credential vars like `GOOGLE_APPLICATION_CREDENTIALS`. Treat it as a
starting point and add your own patterns. It is not the default: stripping
environment variables silently would break agents that rely on inherited
credentials, so it is opt-in.

`env` is enforced at spawn, not applied as a post-hoc filter on a running
process: the subprocess starts with exactly the resolved environment (your
`env`, minus anything `denied_env_patterns` removes from it). Neither control is
a security boundary. A command running under the same OS identity may still
read the parent process's environment through system interfaces such as Linux
procfs, as well as other host files. Use OS-level isolation when commands are
untrusted. The flip side is that a pattern broad enough to strip `PATH` or
`HOME`, or an `env` that omits them, can break command resolution. External
commands may still run via the shell's built-in default `PATH` on some systems,
but don't rely on it -- set `PATH` explicitly when you replace the environment.

## Background processes

`start_command` writes stdout/stderr to temp files and returns a short ID. Use
`check_command(command_id)` to poll and `stop_command(command_id)` to terminate
and collect final output. Processes are launched in their own session (`start_new_session`)
so the whole process group can be signalled -- `SIGTERM`, escalating to
`SIGKILL` after a grace period.

On run end, the toolset's `__aexit__` terminates every still-running background
process and deletes its temp files. The agent runtime enters toolsets via an
`AsyncExitStack`, so this cleanup runs whether the run succeeds or raises -- an
agent that forgets to call `stop_command` won't leak processes.

## Persistent commands

The four tools above are run-scoped: their processes die with the run. Name
`shell` in `tools` to register the persistent tool instead:

```python
from pydantic_ai_harness import Shell

Shell(cwd='./repo', tools=['shell'])
```

`shell(command, mode='foreground', timeout=None)` hands the command to a small
supervisor process started in its own session. The supervisor appends the
command's combined stdout and stderr to an output log, publishes a JSON status
file (`{"pid": ..., "exit_code": ...}`, with `exit_code` null until the command
exits), and reaps the command. The tool returns the supervisor's PID and both
paths, so the model reads progress with its other tools and stops the process
with `kill -- -PID` on POSIX or `taskkill /PID <PID> /T /F` on Windows (the
process group or tree; the result names the right one). Foreground waits up to
`timeout` seconds (default `default_timeout`, at most `MAX_FOREGROUND_WAIT`,
270) for the exit status and returns the last 16,000 bytes of the log followed
by the handles, even if the command is still running; background returns the
handles at once. The handles come last so that `max_output_chars`, which keeps
the tail of an over-long result, cannot drop them. The 270-second cap keeps a tool call shorter than typical provider
request timeouts, so a long build or test run does not stall the conversation:
the model gets the handles back, does other work, and polls the status file.

The command outlives the agent run, the event loop, and (once it has started)
the calling interpreter, so a server the model starts keeps serving. Nothing
wakes the agent when the command finishes; the model polls. A foreground call
that is cancelled (a run cancellation, say) cannot hand back its handles, so it
kills the supervisor's whole session and removes the log directory instead. A
log directory that was handed back is never rotated or deleted: the caller owns
cleaning it up, and a verbose command should bound its own output. A supervisor that exits without
publishing a status (a broken interpreter, say) surfaces as a retry naming the
log directory.

`allowed_commands`, `denied_commands`, `denied_operators`, `allow_interactive`,
`env`, and `denied_env_patterns` apply to `shell` exactly as to `run_command`.
`persist_cwd` does not: every `shell` command starts in the configured `cwd`,
whatever `run_command` has tracked. Commands and logs are host-local, not durable
workflow activities, and are not replay-safe.

Each `shell` call emits progress events in the `shell` namespace, so a UI can
show output as it arrives without parsing the tool result:

| Event | Dispatch | Payload |
|---|---|---|
| `CommandStartedEvent` | stream | `command`, `pid` |
| `CommandOutputEvent` | stream | `text`: a chunk of the combined log, decoded incrementally |
| `CommandFinishedEvent` | stream | `pid`, `output_path`, `status_path`, `exit_code`, `truncated`, `total_lines` |

Output events are emitted while a foreground call waits: at most the first
16,000 bytes of the log per call, in chunks of up to 4,096 bytes, polled every
50 ms. A background call emits only the started and finished events. `finished`
means the tool stopped waiting, not that the command exited: `exit_code` is
`None` while no status has been published, and `truncated` says the log held
more than the events showed. `total_lines` counts logical lines in logs up to
1 MiB and is `None` for larger logs, which are not scanned. A cancelled call may
emit no finished event. Events carry command text and command output, so treat
them as untrusted when rendering. They add no telemetry spans; core already
traces the tool call.

## Working directory

By default each command runs in `cwd` and `cd` has no lasting effect. Set
`persist_cwd=True` to make `cd` sticky across calls: each command is wrapped so
that after it runs, its final working directory is recorded to a private temp
file, and that directory is carried into subsequent calls. The path is only
updated when the command exits `0`, and the record is written out-of-band (not
to stdout) so command output can never spoof the tracked directory.

## Configuration

```python
Shell(
    cwd='.',                       # str | Path -- working directory
    allowed_commands=[],           # allowlist (mutually exclusive with denied)
    denied_commands=[...],         # denylist (defaults to destructive commands)
    denied_operators=[],           # blocked shell operators
    default_timeout=30.0,          # seconds, per run_command
    max_output_chars=50_000,       # output cap returned to the model
    max_file_bytes=None,          # POSIX per-file size limit (None = no added limit)
    persist_cwd=False,             # make cd sticky across calls
    allow_interactive=False,       # allow TTY-style commands
    env=None,                      # explicit env, replacing inheritance (None = inherit)
    denied_env_patterns=[],        # glob patterns stripped from the inherited env
    tools=RUN_SCOPED_TOOL_NAMES,   # which tools to register ('shell' for persistent commands)
)
```

With `tools=['shell']`, `default_timeout` is also the foreground wait and must be
greater than zero and at most 270 seconds; that is checked at construction.

## Agent spec (YAML/JSON)

`Shell` works with Pydantic AI's
[agent spec](https://ai.pydantic.dev/agent-spec/):

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

- [Pydantic AI capabilities](https://ai.pydantic.dev/capabilities/)
- [Toolsets](https://ai.pydantic.dev/toolsets/)
