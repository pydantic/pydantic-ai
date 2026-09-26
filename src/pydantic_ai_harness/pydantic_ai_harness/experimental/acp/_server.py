"""Entry points for serving a Pydantic AI agent over ACP."""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Sequence
from typing import Literal

import acp
from acp import schema
from pydantic_ai.agent import AbstractAgent
from pydantic_ai.models import KnownModelName, Model
from pydantic_ai.output import OutputDataT
from pydantic_ai.tools import AgentDepsT
from pydantic_ai.usage import UsageLimits

from pydantic_ai_harness.experimental.acp._adapter import DEFAULT_VERSION, PydanticAIACPAgent
from pydantic_ai_harness.experimental.acp._permission import PermissionPolicy
from pydantic_ai_harness.experimental.acp._presentation import ToolCallPresenter
from pydantic_ai_harness.experimental.acp._session import SessionConfigFunc
from pydantic_ai_harness.experimental.acp._store import SessionStore


async def run_acp_stdio(
    agent: AbstractAgent[AgentDepsT, OutputDataT],
    *,
    deps: AgentDepsT = None,
    name: str | None = None,
    version: str = DEFAULT_VERSION,
    session_config: SessionConfigFunc[AgentDepsT] | None = None,
    permission_policy: PermissionPolicy | None = None,
    prompt_capabilities: schema.PromptCapabilities | None = None,
    mcp_capabilities: schema.McpCapabilities | None = None,
    tool_presenter: ToolCallPresenter | None = None,
    session_store: SessionStore | None = None,
    models: Sequence[KnownModelName | str] | Literal['all'] | None = None,
    model_resolver: Callable[[str], Model | str] | None = None,
    usage_limits: UsageLimits | None = None,
) -> None:
    """Serve `agent` as an ACP agent over stdin/stdout until the client disconnects.

    This is the entry point an ACP client (such as an editor or terminal UI) launches as a
    subprocess. It blocks for the lifetime of the connection.

    Args:
        agent: The Pydantic AI agent to expose over ACP.
        deps: Dependencies passed to every agent run.
        name: Name advertised to the client. Defaults to the agent's name, then `'pydantic-ai-agent'`.
        version: Version advertised to the client.
        session_config: Per-session factory deriving deps/toolsets from the client's workspace setup.
            See [`PydanticAIACPAgent`][pydantic_ai_harness.experimental.acp.PydanticAIACPAgent].
        permission_policy: Scopes how "always allow"/"always reject" decisions are remembered.
            See [`PydanticAIACPAgent`][pydantic_ai_harness.experimental.acp.PydanticAIACPAgent].
        prompt_capabilities: Prompt content types the agent advertises support for.
            See [`PydanticAIACPAgent`][pydantic_ai_harness.experimental.acp.PydanticAIACPAgent].
        mcp_capabilities: MCP transports the agent advertises support for. Requires a
            `session_config` that connects them. See
            [`PydanticAIACPAgent`][pydantic_ai_harness.experimental.acp.PydanticAIACPAgent].
        tool_presenter: Maps tool calls to rich ACP presentation (kind, file locations, diffs).
            See [`PydanticAIACPAgent`][pydantic_ai_harness.experimental.acp.PydanticAIACPAgent].
        session_store: Enables `session/load` by persisting each session. See
            [`PydanticAIACPAgent`][pydantic_ai_harness.experimental.acp.PydanticAIACPAgent].
        models: Models the client may switch between with the `model` session config option. See
            [`PydanticAIACPAgent`][pydantic_ai_harness.experimental.acp.PydanticAIACPAgent].
        model_resolver: Maps an advertised model id to the `Model` (or model string) used for the
            run. See [`PydanticAIACPAgent`][pydantic_ai_harness.experimental.acp.PydanticAIACPAgent].
        usage_limits: Per-run request/token ceilings applied to every agent run. See
            [`PydanticAIACPAgent`][pydantic_ai_harness.experimental.acp.PydanticAIACPAgent].
    """
    adapter = PydanticAIACPAgent(
        agent,
        deps=deps,
        name=name,
        version=version,
        session_config=session_config,
        permission_policy=permission_policy,
        prompt_capabilities=prompt_capabilities,
        mcp_capabilities=mcp_capabilities,
        tool_presenter=tool_presenter,
        session_store=session_store,
        models=models,
        model_resolver=model_resolver,
        usage_limits=usage_limits,
    )
    # `session/close` is still UNSTABLE in the ACP SDK, and the SDK's router rejects unstable
    # methods with `method_not_found` unless this flag is set. Keep enabled until close stabilizes.
    await acp.run_agent(adapter, use_unstable_protocol=True)


def run_acp_stdio_sync(
    agent: AbstractAgent[AgentDepsT, OutputDataT],
    *,
    deps: AgentDepsT = None,
    name: str | None = None,
    version: str = DEFAULT_VERSION,
    session_config: SessionConfigFunc[AgentDepsT] | None = None,
    permission_policy: PermissionPolicy | None = None,
    prompt_capabilities: schema.PromptCapabilities | None = None,
    mcp_capabilities: schema.McpCapabilities | None = None,
    tool_presenter: ToolCallPresenter | None = None,
    session_store: SessionStore | None = None,
    models: Sequence[KnownModelName | str] | Literal['all'] | None = None,
    model_resolver: Callable[[str], Model | str] | None = None,
    usage_limits: UsageLimits | None = None,
) -> None:
    """Synchronous wrapper around [`run_acp_stdio`][pydantic_ai_harness.experimental.acp.run_acp_stdio].

    Convenient as the `main()` of an ACP agent script, which clients launch as a subprocess.
    """
    asyncio.run(
        run_acp_stdio(
            agent,
            deps=deps,
            name=name,
            version=version,
            session_config=session_config,
            permission_policy=permission_policy,
            prompt_capabilities=prompt_capabilities,
            mcp_capabilities=mcp_capabilities,
            tool_presenter=tool_presenter,
            session_store=session_store,
            models=models,
            model_resolver=model_resolver,
            usage_limits=usage_limits,
        )
    )
