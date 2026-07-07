"""Per-session value types: the client's session setup, its run configuration, and live state."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Hashable, Sequence
from dataclasses import dataclass, field
from typing import Generic, Protocol, TypeAlias

from acp import Client, schema
from pydantic_ai.messages import ModelMessage
from pydantic_ai.tools import AgentDepsT
from pydantic_ai.toolsets import AbstractToolset

# A single MCP server configuration, in any of the transports ACP carries.
McpServer: TypeAlias = schema.HttpMcpServer | schema.SseMcpServer | schema.AcpMcpServer | schema.McpServerStdio

# MCP server configuration list, as carried by ACP session methods (matches `acp.Agent`).
McpServers: TypeAlias = list[McpServer] | None

# A single `session/update` payload, as accepted by the client's `session_update`. The adapter
# records these per session (the client-visible transcript) so `session/load` can replay them.
SessionUpdate: TypeAlias = (
    schema.UserMessageChunk
    | schema.AgentMessageChunk
    | schema.AgentThoughtChunk
    | schema.ToolCallStart
    | schema.ToolCallProgress
    | schema.AgentPlanUpdate
    | schema.AvailableCommandsUpdate
    | schema.CurrentModeUpdate
    | schema.ConfigOptionUpdate
    | schema.SessionInfoUpdate
    | schema.UsageUpdate
)


@dataclass(frozen=True, kw_only=True)
class AcpSession:
    """The setup an ACP client provides when it starts a session.

    Passed to the adapter's `session_config` factory so an embedder can derive per-session run
    configuration from the workspace the client opened -- for example rooting the agent's
    dependencies at `cwd` or turning the client-provided `mcp_servers` into toolsets.
    `client_capabilities` is whatever the client advertised during `initialize` (for example
    filesystem or terminal support).

    `client` and `session_id` are the live connection handle for this session: pass them to a
    client-backed toolset (see [`acp_filesystem`][pydantic_ai_harness.experimental.acp.acp_filesystem]) to
    route the agent's I/O through the editor rather than local disk.
    """

    cwd: str
    mcp_servers: list[McpServer]
    client_capabilities: schema.ClientCapabilities | None
    client: Client
    session_id: str


@dataclass(frozen=True, kw_only=True)
class AcpSessionConfig(Generic[AgentDepsT]):
    """Per-session run configuration returned by a `session_config` factory.

    `deps` and `toolsets` are applied to every agent run in that session, mirroring
    `Agent.run(..., deps=..., toolsets=...)`. `toolsets` is added to the agent's own toolsets
    rather than replacing them. `deps` is required; pass `deps=None` for an agent with no
    dependencies.
    """

    deps: AgentDepsT
    toolsets: Sequence[AbstractToolset[AgentDepsT]] | None = None


# A `Protocol` rather than a `Callable` alias so it stays subscriptable (`SessionConfigFunc[MyDeps]`)
# and evaluable by `typing.get_type_hints` in the public constructor's annotations.
class SessionConfigFunc(Protocol[AgentDepsT]):
    """Factory mapping an ACP session's setup to its run configuration; may be sync or async.

    Any callable taking the client's `AcpSession` and returning an `AcpSessionConfig` (or an
    awaitable of one) satisfies it.
    """

    # `session` is positional-only (`/`) so any callable taking one `AcpSession` satisfies the
    # protocol regardless of how it names the parameter -- matching a plain `Callable` alias.
    def __call__(
        self, session: AcpSession, /
    ) -> (
        AcpSessionConfig[AgentDepsT] | Awaitable[AcpSessionConfig[AgentDepsT]]
    ): ...  # pragma: no cover - structural protocol; the body never runs


@dataclass(kw_only=True)
class SessionState(Generic[AgentDepsT]):
    """All per-session adapter state; `new_session`/`load_session` create one, `close_session` removes it."""

    session_id: str
    config: AcpSessionConfig[AgentDepsT]
    cwd: str
    history: list[ModelMessage] = field(default_factory=list[ModelMessage])
    transcript: list[SessionUpdate] = field(default_factory=list[SessionUpdate])
    # The model the client selected for this session via the `model` config option, or `None` to
    # use the agent's own model. Applied as a per-run override so the shared agent is never mutated.
    model: str | None = None
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    active_turn: asyncio.Task[schema.PromptResponse] | None = None
    cancel_requested: bool = False
    always_allow: set[Hashable] = field(default_factory=set[Hashable])
    always_reject: set[Hashable] = field(default_factory=set[Hashable])
