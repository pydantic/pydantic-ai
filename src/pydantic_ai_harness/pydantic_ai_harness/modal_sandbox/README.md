# Modal Sandbox

`ModalSandbox` gives an agent an isolated cloud container for running commands
and working with files. Use it for coding, data processing, and other tasks that
should not execute model-generated commands on the application host.

The capability adds shell and file tools backed by a
[Modal sandbox](https://modal.com/docs/guide/sandbox). By default, every agent
run gets a fresh sandbox created from a container image. The capability requests
termination when the run ends. You can also attach an existing sandbox or reuse
one across several runs.

## Quick start

Install the `modal` extra and authenticate with the Modal CLI. In CI, set
`MODAL_TOKEN_ID` and `MODAL_TOKEN_SECRET` instead.

```bash
uv add "pydantic-ai-harness[modal]"
modal token new                # writes ~/.modal.toml
# or, e.g. in CI:
export MODAL_TOKEN_ID=...
export MODAL_TOKEN_SECRET=...
```

Add `ModalSandbox` to the agent:

```python
from pydantic_ai import Agent
from pydantic_ai_harness.modal_sandbox import ModalSandbox

agent = Agent(
    'anthropic:claude-sonnet-4-6',
    capabilities=[ModalSandbox(image='python:3.12-slim')],
)

result = agent.run_sync('Create a Python script and run its tests.')
print(result.output)
```

During the run, the agent can create files, inspect its working directory, run
commands, and react to command failures. The sandbox is separate from the host
filesystem and process space.

## Tools

| Tool | Purpose |
|---|---|
| `run_command` | Run a shell command (`sh -c`) in the sandbox. Pipes, redirection, `&&`, and globs work. Returns labelled stdout/stderr plus an exit code on failure. |
| `read_file` | Read a text file from the sandbox. |
| `write_file` | Write text to a file (creating parent directories). |
| `list_directory` | List a directory's entries (directories shown with a trailing `/`). |

Output is labelled with `[stdout]` / `[stderr]` markers and an `[exit code: N]`
line on non-zero exit. Each command stream (and each file read) is truncated
separately by `max_output_bytes` (UTF-8 bytes) and `max_output_lines` (lines),
whichever is hit first, so a large stderr cannot crowd out stdout and the labels
always survive. Labels, truncation or continuation notes, and command status add
a small amount beyond those payload limits. For commands the **tail** is kept, so
errors survive truncation; file reads keep the head and return the next `offset`
to page from. A non-zero exit from `run_command` is reported, not raised, so the
model can react to it; file-tool failures (missing path, etc.) come back as a
retry prompt.

The command reader retains exactly the last `max_output_bytes` from each stream
after each transport chunk arrives, and the cut is marked in the tool output.
One transport chunk can temporarily be larger than the configured limit. Command
output is read as bytes and decoded as UTF-8 with `errors='replace'`, so binary
or invalid UTF-8 output is reported with replacement characters instead of
crashing the run.

`run_command` runs through `sh -c`; `read_file`, `write_file`, and
`list_directory` use Modal's filesystem API directly (no shell), so writes stream
the content rather than passing it as a command argument, and parent directories
are created on write. Modal's filesystem API only accepts absolute paths, so a
relative path given to a file tool is resolved against the working directory used
by `run_command` (queried once with `pwd` and cached), keeping both views of the
tree consistent.

## Failure handling

Failures split into two kinds:

- **Recoverable** -- a bad path, a command that exits non-zero, a transient
  sandbox-side error. These come back to the model as a retry (`ModelRetry`) or,
  for `run_command`, as reported output it can react to. Retrying can plausibly
  work, so the run continues.
- **Terminal** -- the sandbox itself is gone (terminated, or expired at its
  `sandbox_timeout`), raising `ModalSandboxUnavailableError`, or the credentials
  were rejected, raising `ModalSandboxAuthError`. Re-running the command cannot
  fix these, so the tool lets them propagate (both are `ModalSandboxTerminalError`
  subclasses) and the run ends with an actionable message instead of looping the
  model against a dead sandbox. If owned runs legitimately hit the lifetime,
  raise `sandbox_timeout`.

## Sandbox lifetime

By default the capability is **owned**: each run creates a fresh sandbox and
requests its termination when the run ends. Teardown waits for confirmation for
a bounded period; if Modal's control plane does not respond, `sandbox_timeout`
remains the server-side cleanup backstop. Each owned run spins up its own sandbox,
so expect a cold-start cost per run. There are two ways to reuse one.

The sandbox is provisioned when a run enters the capability toolset, even if the
model does not call a sandbox tool. Pydantic AI's deferred tool loading controls
which tool definitions are sent to the model; it does not defer this toolset
lifecycle.

**Attach** to a sandbox you manage elsewhere (e.g. created via the Modal CLI) by
id. It is never terminated by the capability:

```python
from pydantic_ai_harness.modal_sandbox import ModalSandbox

ModalSandbox(sandbox_id='sb-abc123')   # attach to an existing sandbox
```

**Inject a session** you own to reuse one sandbox across runs while controlling
its lifetime yourself. The capability uses the session but never opens or
terminates it, so the owner decides when the sandbox goes away, and can read its
`sandbox_id`:

```python
from pydantic_ai import Agent
from pydantic_ai_harness.modal_sandbox import ModalSandbox, ModalSandboxSession

async with ModalSandboxSession(image='python:3.12-slim', sandbox_timeout=1800) as session:
    print(session.sandbox_id)   # the running sandbox id
    agent = Agent(
        'anthropic:claude-sonnet-4-6',
        capabilities=[ModalSandbox(session=session, max_command_timeout=600)],
    )
    await agent.run('clone the repo and install deps')   # same sandbox...
    await agent.run('run the test suite')                # ...reused across runs
# the owner requests sandbox termination when the session exits
```

Size the session's `sandbox_timeout` to the whole workload: the default 300s
would expire partway through a multi-run session like this one. The capability
cannot see a reused sandbox's real lifetime, so each command there is capped at
300s unless `max_command_timeout` raises the ceiling.

A reused sandbox (attach or injected session) is not concurrency-safe across
overlapping runs: they share one filesystem and one process space. Use separate
sandboxes for runs that overlap in time.

## Cancellation

Modal does not currently expose a way to kill a single running command, so a
command is stopped by its own deadline or by the whole sandbox being terminated.
The capability is built around that:

- A cancelled run stops waiting for the command immediately, but the command
  keeps running in the sandbox until its deadline. Every `run_command` carries
  one (`default_command_timeout`, or the per-call `timeout_seconds`), so a
  cancelled or abandoned command is reaped within that window rather than running
  on. Lower `default_command_timeout` to shorten the worst-case window. A
  model-supplied `timeout_seconds` is capped at `max_command_timeout` (which
  defaults to `sandbox_timeout`), so the model cannot ask for an unbounded one.
- When an owned run ends or is cancelled, the capability requests sandbox
  termination and waits for a bounded period. `sandbox_timeout` remains the
  server-side backstop if the teardown RPC cannot be confirmed.
- An attached or injected sandbox is never terminated by the capability (its
  owner controls that), so an in-flight command there is bounded only by its
  deadline.

## Lower-level access

`ModalSandbox` is the main entry point. The toolset is an implementation
detail. `ModalSandboxSession` is public for applications that need to create,
attach to, or share a sandbox explicitly:

```python
from pydantic_ai_harness.modal_sandbox import ModalSandboxSession

async with ModalSandboxSession(image='python:3.12-slim') as session:
    result = await session.exec(['echo', 'hello'])
    print(result.stdout, result.returncode)
```

## Configuration

```python
from pydantic_ai_harness.modal_sandbox import ModalSandbox

ModalSandbox(
    image='python:3.12-slim',     # registry image for owned sandboxes
    sandbox_id=None,              # attach to an existing sandbox instead of creating one
    session=None,                 # reuse a ModalSandboxSession you own across runs
    app_name='pydantic-ai-harness',  # Modal app the owned sandbox runs under
    create_app_if_missing=True,   # create the app if it does not exist
    sandbox_timeout=300,          # max lifetime (seconds) of an owned sandbox
    workdir=None,                 # working directory for commands (Modal default when None)
    env=None,                     # environment variables for an owned sandbox (dict)
    default_command_timeout=60.0, # default timeout for one run_command (seconds; fractions round up)
    max_command_timeout=None,     # hard ceiling for one command; None -> sandbox_timeout
    max_output_bytes=50 * 1024,   # per-stream payload cap in UTF-8 bytes before annotations
    max_output_lines=2000,        # per-stream payload line cap before annotations
    max_read_bytes=5 * 1024 * 1024,  # refuse read_file on files larger than this
    instructions=None,            # None: default usage instructions; '': none; str: your own
)
```

Modal enforces whole-second command deadlines, so a fractional
`default_command_timeout` or `timeout_seconds` rounds up (0.5 behaves as 1).
The default instructions state the tools, the command timeout, and its ceiling;
set `instructions=''` to add none, or pass your own text (needed when prefixing,
see below).

`read_file` loads a file fully before returning a window of it, so it refuses
files larger than `max_read_bytes` and tells the model to slice them with a shell
command (`head`, `tail`, `sed -n`, `grep`) instead. That guard reads the size from
a `stat` first and checks the returned byte count again. A file that grows
between those calls can temporarily exceed the limit in client memory before it is
rejected. The guard is not a defense against special or virtual files whose
reported size is misleading because Modal's filesystem API does not expose a
bounded read. Use `run_command` with a bounded shell command for those paths.

`list_directory` reads the whole directory listing before capping it (Modal has
no streaming list API), so listing a directory with a very large number of
entries costs memory proportional to the entry count. Point the model at a
narrowed `run_command` (`ls | head`, `find -maxdepth`) for directories that big.

## Not yet supported

- Streaming command output: `run_command` returns once the command finishes (or
  hits its deadline), not incrementally.
- Custom-built images, mounts, or `modal.Secret`: `image` takes a registry tag,
  and `env` takes plain environment variables. For anything richer, create the
  sandbox yourself with the Modal SDK and pass it via `sandbox_id` or `session`.
- Spilling full output to a file: truncated file reads end with the next
  `offset` to page from and oversized files get a shell-slice hint (`head`,
  `tail`, `sed -n`); truncated command output gets a truncation marker. Nothing
  is written to a file in the sandbox for the model to open. This is a
  deliberate choice for now.

Modal's SDK is asyncio-native, so the capability drives its async (`.aio`) API
directly and requires an asyncio event loop (it does not run under trio).

## Composing with other capabilities

Do not combine this capability with another unprefixed capability that registers
`run_command`, `read_file`, `write_file`, or `list_directory` (e.g. the Shell or
FileSystem capabilities). Pydantic AI rejects duplicate tool names. If an agent
needs both sets of tools, prefix one of the capabilities:

```python
from pydantic_ai.capabilities import PrefixTools

from pydantic_ai_harness.modal_sandbox import ModalSandbox

sandbox = PrefixTools(
    wrapped=ModalSandbox(
        instructions=(
            'You have a Modal cloud sandbox. Use the modal_-prefixed tools to run '
            'shell commands and manage files in it.'
        )
    ),
    prefix='modal',
)
```

Prefixing renames the tools (`modal_run_command`, ...) but does not rewrite the
capability's default instructions, which name the unprefixed tools -- pass
`instructions` with text that matches the prefixed names.

## Agent spec (YAML/JSON)

`ModalSandbox` works with Pydantic AI's
[agent spec](https://pydantic.dev/docs/ai/core-concepts/agent-spec/):

```yaml
# agent.yaml
model: anthropic:claude-sonnet-4-6
capabilities:
  - ModalSandbox:
      image: python:3.12-slim
      sandbox_timeout: 600
```

```python
from pydantic_ai import Agent
from pydantic_ai_harness.modal_sandbox import ModalSandbox

agent = Agent.from_file('agent.yaml', custom_capability_types=[ModalSandbox])
```

## Further reading

- [Modal sandboxes](https://modal.com/docs/guide/sandbox)
- [Pydantic AI capabilities](https://pydantic.dev/docs/ai/core-concepts/capabilities/)
- [Pydantic AI toolsets](https://pydantic.dev/docs/ai/tools-toolsets/toolsets/)
- [Modal Sandbox source code](https://github.com/pydantic/pydantic-ai-harness/tree/main/pydantic_ai_harness/modal_sandbox/)
- [Pydantic AI Harness version policy](https://github.com/pydantic/pydantic-ai-harness#version-policy)

The API may change between releases while Pydantic AI Harness is on 0.x
versions.
