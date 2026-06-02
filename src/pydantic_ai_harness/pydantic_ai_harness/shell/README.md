# Shell

Give an agent the ability to run shell commands, with allow/deny controls and
managed background processes.

## The problem

Agents frequently need to run a build, a test suite, a linter, or a quick
`grep`. Wiring up subprocess handling — streaming output, timeouts, truncation,
killing runaway processes, and cleaning up background jobs at the end of a run —
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

Output is labelled with `[stdout]` / `[stderr]` markers and an `[exit code: N]`
line on non-zero exit. When it exceeds `max_output_chars` the **tail** is kept
(the head is dropped), so errors, stack traces, and the `[stderr]` section —
which all land at the end — survive truncation.

## Command controls

| Field | Effect |
|---|---|
| `allowed_commands` | If non-empty, only these executables may run (allowlist). |
| `denied_commands` | These executables are always rejected (denylist). |
| `denied_operators` | Shell operators (e.g. `>`, `>>`, `|`) that are rejected when present. |
| `allow_interactive` | If `False` (default), commands that expect a TTY (`vi`, `sudo`, `ssh`, …) are blocked. |

`allowed_commands` and `denied_commands` are mutually exclusive — set one, not
both. `denied_commands` defaults to a list of destructive commands (`rm`,
`rmdir`, `mkfs`, `dd`, `shutdown`, `reboot`, …); pass an empty list to disable.
The executable name is extracted with `shlex`, so arguments don't bypass the
check.

> **These checks are best-effort, not a security boundary.** A sufficiently
> motivated agent can defeat them (e.g. `bash -c '...'`, env-var indirection).
> For hard guarantees, run the agent inside OS-level isolation — a container or
> sandbox.

## Background processes

`start_command` writes stdout/stderr to temp files and returns a short ID. Use
`check_command(id)` to poll and `stop_command(id)` to terminate and collect
final output. Processes are launched in their own session (`start_new_session`)
so the whole process group can be signalled — `SIGTERM`, escalating to
`SIGKILL` after a grace period.

On run end, the toolset's `__aexit__` terminates every still-running background
process and deletes its temp files. The agent runtime enters toolsets via an
`AsyncExitStack`, so this cleanup runs whether the run succeeds or raises — an
agent that forgets to call `stop_command` won't leak processes.

## Working directory

By default each command runs in `cwd` and `cd` has no lasting effect. Set
`persist_cwd=True` to make `cd` sticky: the toolset appends a `pwd` sentinel to
successful commands, parses the result, and carries the new directory into
subsequent calls. Commands containing `;` skip the sentinel injection so the
`&&`-gated sentinel can't be bypassed.

## Configuration

```python
Shell(
    cwd='.',                       # str | Path — working directory
    allowed_commands=[],           # allowlist (mutually exclusive with denied)
    denied_commands=[...],         # denylist (defaults to destructive commands)
    denied_operators=[],           # blocked shell operators
    default_timeout=30.0,          # seconds, per run_command
    max_output_chars=50_000,       # output cap returned to the model
    persist_cwd=False,             # make cd sticky across calls
    allow_interactive=False,       # allow TTY-style commands
)
```

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
