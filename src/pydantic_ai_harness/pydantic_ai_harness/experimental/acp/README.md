# ACP (Agent Client Protocol)

> [!WARNING]
> **Experimental.** This capability lives under `pydantic_ai_harness.experimental` and may
> change or be removed in any release, without a deprecation period. Import it from the
> experimental path -- there is no top-level export:
>
> ```python
> from pydantic_ai_harness.experimental.acp import run_acp_stdio_sync
> ```
>
> Importing any experimental capability emits a `HarnessExperimentalWarning`. Silence **all**
> harness experimental warnings with a single filter (no per-capability lines needed):
>
> ```python
> import warnings
> from pydantic_ai_harness.experimental import HarnessExperimentalWarning
>
> warnings.filterwarnings('ignore', category=HarnessExperimentalWarning)
> ```

Expose a Pydantic AI agent to editors and terminal UIs over the [Agent Client Protocol](https://agentclientprotocol.com).

[Source](https://github.com/pydantic/pydantic-ai-harness/tree/main/pydantic_ai_harness/experimental/acp/)

## The problem

Editors like [Zed](https://zed.dev/docs/ai/external-agents) speak ACP: a stdio JSON-RPC protocol that lets a TUI or editor drive an external coding agent -- streaming its text, rendering its file edits as diffs, and prompting the user to approve sensitive tool calls. To plug a Pydantic AI agent into one of these editors you would otherwise have to implement the ACP server side yourself.

## The solution

`run_acp_stdio` serves any Pydantic AI `Agent` as an ACP agent over stdin/stdout. The editor launches your script as a subprocess and talks to it; the adapter translates between ACP and the agent's run loop:

| ACP needs | The adapter provides |
|---|---|
| Streamed assistant text and reasoning | Agent text/thinking deltas, chunked under the wire limit |
| Rich tool calls (`kind`, file `locations`, diffs) | A presenter that recognizes `FileSystem`/`Shell` tool calls |
| Human-in-the-loop tool approval | Maps ACP permission requests to Pydantic AI's deferred-approval tools |
| Per-workspace sessions | A `session_config` hook to root tools at the client's working directory |
| Cancellation, multi-turn history, session close | Handled per session |

## Installation

```bash
uv add "pydantic-ai-harness[acp]"
```

This pulls in the [`agent-client-protocol`](https://pypi.org/project/agent-client-protocol/) SDK. The rest of the harness does not depend on it -- only `pydantic_ai_harness.experimental.acp` does.

## Quick start

Write a script that builds your agent and serves it:

```python
# my_acp_agent.py
from pydantic_ai import Agent
from pydantic_ai_harness.experimental.acp import run_acp_stdio_sync


def build_agent() -> Agent[None, str]:
    return Agent('anthropic:claude-sonnet-4-6', instructions='You are a coding assistant.')


if __name__ == '__main__':
    run_acp_stdio_sync(build_agent())
```

`run_acp_stdio_sync` blocks for the lifetime of the connection -- it is the `main()` of an agent the editor launches. Inside an existing event loop, use the async `run_acp_stdio` instead.

## Connecting from an editor

ACP clients launch the agent as a subprocess. In Zed, register it as an [external agent](https://zed.dev/docs/ai/external-agents) in `settings.json`:

```json
{
  "agent_servers": {
    "My Pydantic AI Agent": {
      "type": "custom",
      "command": "python",
      "args": ["/absolute/path/to/my_acp_agent.py"],
      "env": { "ANTHROPIC_API_KEY": "..." }
    }
  }
}
```

Any ACP-compatible client works the same way -- point it at `python my_acp_agent.py`. Refer to your editor's external-agent documentation for the exact config location.

The provider environment must be available to the launched subprocess. GUI editors and SDK-based
test wrappers may not source your interactive shell startup files, and the ACP Python SDK's
`spawn_agent_process` helper starts from a trimmed default environment unless you pass `env`
explicitly. If a real-model agent exits before initialize or fails provider auth, first verify that
the command's process can see variables such as `ANTHROPIC_API_KEY`.

## Rooting tools at the workspace

A coding agent should read and write files in the workspace the editor opened, not wherever the subprocess started. ACP gives each session a working directory (`cwd`); a `session_config` factory turns that into per-session tools:

```python
from pydantic_ai import Agent
from pydantic_ai_harness.experimental.acp import AcpSession, AcpSessionConfig, run_acp_stdio_sync
from pydantic_ai_harness.filesystem import FileSystem
from pydantic_ai_harness.shell import Shell

agent = Agent('anthropic:claude-sonnet-4-6')


def session_config(session: AcpSession) -> AcpSessionConfig[None]:
    # Root file and shell tools at the workspace the client opened.
    return AcpSessionConfig(
        deps=None,
        toolsets=[
            FileSystem[None](root_dir=session.cwd).get_toolset(),
            Shell[None](cwd=session.cwd).get_toolset(),
        ],
    )


if __name__ == '__main__':
    run_acp_stdio_sync(agent, session_config=session_config)
```

The factory runs once per session with the client's [`AcpSession`][pydantic_ai_harness.experimental.acp.AcpSession] setup (its `cwd`, `mcp_servers`, and capabilities) and returns an [`AcpSessionConfig`][pydantic_ai_harness.experimental.acp.AcpSessionConfig] whose `deps` and `toolsets` apply to every run in that session. This is correct across multiple concurrent sessions in one process, where a single static `FileSystem` could not be.

## Editor-native filesystem and shell (optional)

The local `FileSystem` and `Shell` above operate on the agent process's own disk and subprocesses. An editor's source of truth is different: it has unsaved buffers, the file layout it considers the workspace, and -- for a remote or containerized editor -- the machine the code actually lives on. When the client advertises support, [`acp_filesystem`][pydantic_ai_harness.experimental.acp.acp_filesystem] and [`acp_terminal`][pydantic_ai_harness.experimental.acp.acp_terminal] give the agent `read_file`/`write_file`/`run_command` tools that route through the client, so it acts where the user is:

```python
from pydantic_ai_harness.experimental.acp import AcpSession, AcpSessionConfig, acp_filesystem, acp_terminal
from pydantic_ai_harness.filesystem import FileSystem
from pydantic_ai_harness.shell import Shell


def session_config(session: AcpSession) -> AcpSessionConfig[None]:
    # Use the editor's filesystem/terminal when offered; otherwise fall back to local.
    fs = acp_filesystem(session) or FileSystem[None](root_dir=session.cwd).get_toolset()
    shell = acp_terminal(session) or Shell[None](cwd=session.cwd).get_toolset()
    return AcpSessionConfig(deps=None, toolsets=[fs, shell])
```

Each helper returns `None` when the client did not advertise the capability, so the `or` falls back to local and the agent works either way. The tool names match the local `FileSystem`/`Shell`, so rich rendering (next section) is identical. `acp_terminal` runs the command in the editor's environment and returns its captured output (see [Limitations](#cancellation-and-limitations)).

If a client advertises filesystem *reads* but not *writes*, `acp_filesystem` keeps editor-native reads and sends writes to the local `FileSystem` rooted at `session.cwd` -- coherent only when the agent shares the workspace disk with the editor (same machine, or an agent inside the editor's container); for a remote editor those writes land on the agent's disk, not the editor's.

## Tool approval

Mark a tool to require approval and ACP relays the decision to the client, which shows the user an approve/reject prompt:

```python
@agent.tool_plain(requires_approval=True)
def delete_file(path: str) -> str:
    ...
```

The lifecycle the client sees is `pending` (awaiting approval) → `in_progress` (granted, running) → `completed`/`failed`, so an unapproved action is never shown as already running. "Always allow"/"always reject" decisions are remembered for the session, scoped by default to the exact call (tool name plus arguments) so approving one call never silently approves a different one. Pass `permission_policy` to widen or narrow that scope.

## Rich tool rendering

By default the adapter recognizes the harness `FileSystem` and `Shell` tool calls by name and annotates them with an ACP `kind` (`read`/`edit`/`search`/`execute`), the file `locations` they touch, and an inline diff for edits -- so the editor renders click-to-file links and diff views instead of opaque JSON. Pass `tool_presenter` to add rendering for your own tools (optionally with `chain_presenters` ahead of the default), or `lambda _call: None` to disable it.

## MCP servers

An ACP client may offer MCP servers during session setup. This adapter does not connect them itself; a `session_config` is the place to turn `session.mcp_servers` into Pydantic AI toolsets. If a client sends MCP servers and no `session_config` is installed to consume them, the session request is rejected (rather than silently ignoring them) so the mismatch is visible. The spec expects every agent to accept stdio MCP servers, so an agent meant for arbitrary editors should install a `session_config` that connects them (for example with `pydantic_ai.mcp.MCPServerStdio`).

A spec-following client only sends HTTP/SSE MCP servers when the agent advertises support for those transports during `initialize` (stdio servers are not capability-gated). When your `session_config` connects them, say so:

```python
PydanticAIACPAgent(
    agent,
    session_config=connect_mcp_servers,
    mcp_capabilities=schema.McpCapabilities(http=True, sse=True),
)
```

## Prompt content types

The agent advertises which prompt content it accepts. The default is **text only**, so a client is not invited to send blocks a text model cannot handle. Enable the kinds your model supports:

```python
from acp import schema

run_acp_stdio_sync(agent, prompt_capabilities=schema.PromptCapabilities(image=True, embedded_context=True))
```

## Session persistence

Pass a `session_store` to let a client reopen a past conversation with `session/load`. Each committed turn is persisted as two parts -- the model's message history and the client-visible transcript (the user's messages plus everything streamed back) -- and reopening restores the history into the agent and replays the transcript to the client, so its UI is rebuilt as the user last saw it. Without a store, `session/load` is advertised as unsupported.

```python
from pydantic_ai_harness.experimental.acp import InMemorySessionStore

run_acp_stdio_sync(agent, session_store=InMemorySessionStore())
```

`InMemorySessionStore` keeps sessions for the lifetime of the process. Implement the `SessionStore` protocol (`save`/`load` a `StoredSession`) over a file or database to make them survive a restart -- the stored values are Pydantic models, so they serialize with Pydantic.

Session persistence is for *reopening a conversation*; it is orthogonal to per-run durability. To also make individual turns crash-resilient (or resume a long sub-agent run), add a step-durability capability to the agent -- each ACP turn is one agent run, so the two layers compose with no glue.

## Model selection

Pass `models` to advertise a stable ACP session config option named `model` (using Pydantic AI [model names](https://ai.pydantic.dev/models/)). The first is each session's default. A selection is applied as a per-run override -- the shared agent is never mutated -- and is persisted with the session when a `session_store` is set.

```python
run_acp_stdio_sync(agent, models=['anthropic:claude-sonnet-4-6', 'anthropic:claude-opus-4-8', 'openai:gpt-4o'])
```

A model id is any string a Pydantic AI model accepts, so newer models not yet in `KnownModelName` work too. Pass `models='all'` to offer every model Pydantic AI knows (its default is then the first known model, so curate the list when you want a specific default). Without `models`, no model config option is advertised.

To advertise ids Pydantic AI's `infer_model` does not understand (for example OAuth or subscription models), pass `model_resolver` to map the selected id to a prebuilt `Model`; returning the id unchanged falls back to `infer_model`.

## Token usage

Each completed turn reports its token counts (input/output/total, plus cached tokens) on the ACP `PromptResponse`, summed across any approval pauses. This is an UNSTABLE ACP field, so clients that don't support it simply ignore it.

## Cancellation and limitations

- **Cancellation.** `session/cancel` and `session/close` cancel the in-flight turn; close waits for it to unwind before returning. Cooperative async tools stop promptly. A synchronous tool already running in a worker thread cannot be force-stopped, so its side effects may complete after the turn reports `cancelled` -- prefer async tools for cancellation-sensitive work.
- **Approval detection.** Tools that require approval are recognized when they live in a `FunctionToolset` (which the harness `FileSystem`/`Shell` and `@agent.tool` both use). A tool whose approval requirement is decided dynamically per call (by raising `ApprovalRequired` from its body) starts as `in_progress`, and any side effects it ran *before* raising have already happened by the time the client is asked -- use an `ApprovalRequiredToolset` (which gates before the tool body runs) for actions that must not partially execute before approval.
- **Overwrite diffs.** `write_file` renders an overwrite as if creating a new file (no prior contents), so the diff understates what it replaced.
- **Live terminal panes.** `acp_terminal` returns a command's captured output; it does not embed a *live* terminal pane in the tool call, which would need the terminal id at call-start, before the command runs.
- **Images.** Prompt image blocks are off by default and must be enabled via `prompt_capabilities` with a model that accepts them (see [Prompt content types](#prompt-content-types)). The harness `FileSystem.read_file` is text-only, so the agent cannot open image files from the workspace itself.
- **Slash commands.** The adapter does not yet advertise any commands (`available_commands`), so no slash commands appear in the client. Planned.
- **MCP servers.** Client-offered MCP servers are surfaced to your `session_config` to turn into toolsets (advertise the transports with `mcp_capabilities`); the adapter does not auto-connect them. Resource metadata is not yet wired end-to-end.

## API

```python {test="skip"}
run_acp_stdio(            # async; serve until the client disconnects
    agent,
    *,
    deps=None,
    name=None,            # advertised name; defaults to the agent's name
    version='0.1.0',
    session_config=None,  # per-session deps/toolsets from the client's setup
    permission_policy=None,   # scope of remembered "always" approval decisions
    prompt_capabilities=None, # defaults to text-only
    mcp_capabilities=None,    # MCP transports to advertise; needs a session_config to connect them
    tool_presenter=None,      # defaults to the FileSystem/Shell presenter
    session_store=None,       # enables session/load by persisting each session
    models=None,              # models offered as the `model` config option ('all' for every known model)
    model_resolver=None,      # maps an advertised model id to the Model used for the run
    usage_limits=None,        # per-run request/token ceilings
)

run_acp_stdio_sync(...)   # synchronous wrapper, same arguments

PydanticAIACPAgent(agent, *, ...)  # the ACP agent object, to embed in a custom server
```

## Further reading

- [Agent Client Protocol](https://agentclientprotocol.com) -- protocol specification
- [Zed external agents](https://zed.dev/docs/ai/external-agents) -- editor-side configuration
- [Human-in-the-loop tool approval](https://pydantic.dev/docs/ai/tools-toolsets/deferred-tools/#human-in-the-loop-tool-approval) (Pydantic AI)
- [Pydantic AI capabilities](https://ai.pydantic.dev/capabilities/)
