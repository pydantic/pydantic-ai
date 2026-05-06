"""Provides an AG-UI protocol adapter for the Pydantic AI agent.

This package provides seamless integration between pydantic-ai agents and ag-ui
for building interactive AI applications with streaming event-based communication.
"""

# TODO (v2): Remove this module in favor of `pydantic_ai.ui.ag_ui`

from __future__ import annotations

from collections.abc import AsyncIterator, Sequence
from typing import Any, Literal

from . import DeferredToolResults
from .agent import AbstractAgent
from .agent.abstract import AgentMetadata
from .messages import ModelMessage
from .models import KnownModelName, Model
from .output import OutputSpec
from .settings import ModelSettings
from .tools import AgentDepsT
from .toolsets import AbstractToolset
from .usage import RunUsage, UsageLimits

try:
    from ag_ui.core import BaseEvent
    from ag_ui.core.types import RunAgentInput
    from starlette.requests import Request
    from starlette.responses import Response

    from .ui import SSE_CONTENT_TYPE, OnCompleteFunc, StateDeps, StateHandler
    from .ui.ag_ui import AGUIAdapter
    from .ui.ag_ui._utils import DEFAULT_AG_UI_VERSION
    from .ui.ag_ui.app import AGUIApp
except ImportError as e:  # pragma: no cover
    raise ImportError(
        'Please install the `ag-ui-protocol` and `starlette` packages to use `AGUIAdapter`, '
        'you can use the `ag-ui` optional group — `pip install "pydantic-ai-slim[ag-ui]"`'
    ) from e


__all__ = [
    'SSE_CONTENT_TYPE',
    'StateDeps',
    'StateHandler',
    'AGUIApp',
    'OnCompleteFunc',
    'handle_ag_ui_request',
    'run_ag_ui',
]


async def handle_ag_ui_request(
    agent: AbstractAgent[AgentDepsT, Any],
    request: Request,
    *,
    ag_ui_version: str = DEFAULT_AG_UI_VERSION,
    preserve_file_data: bool = False,
    output_type: OutputSpec[Any] | None = None,
    message_history: Sequence[ModelMessage] | None = None,
    deferred_tool_results: DeferredToolResults | None = None,
    conversation_id: str | None = None,
    model: Model | KnownModelName | str | None = None,
    deps: AgentDepsT = None,
    model_settings: ModelSettings | None = None,
    usage_limits: UsageLimits | None = None,
    usage: RunUsage | None = None,
    metadata: AgentMetadata[AgentDepsT] | None = None,
    infer_name: bool = True,
    toolsets: Sequence[AbstractToolset[AgentDepsT]] | None = None,
    on_complete: OnCompleteFunc[BaseEvent] | None = None,
    manage_system_prompt: Literal['server', 'client'] = 'server',
    allowed_file_url_schemes: frozenset[str] = frozenset({'http', 'https'}),
) -> Response:
    """Handle an AG-UI request by running the agent and returning a streaming response.

    Args:
        agent: The agent to run.
        request: The Starlette request (e.g. from FastAPI) containing the AG-UI run input.
        ag_ui_version: AG-UI protocol version controlling thinking/reasoning event format.
        preserve_file_data: Whether to preserve agent-generated files and uploaded files
            in AG-UI message conversion. See [`AGUIAdapter.preserve_file_data`][pydantic_ai.ui.ag_ui.AGUIAdapter.preserve_file_data].

        output_type: Custom output type to use for this run, `output_type` may only be used if the agent has no
            output validators since output validators would expect an argument that matches the agent's output type.
        message_history: History of the conversation so far.
        deferred_tool_results: Optional results for deferred tool calls in the message history.
        conversation_id: ID of the conversation this run belongs to. Pass `'new'` to start a fresh conversation, ignoring any `conversation_id` already on `message_history`. If omitted, falls back to the most recent `conversation_id` on `message_history` or a freshly generated UUID7.
        model: Optional model to use for this run, required if `model` was not set when creating the agent.
        deps: Optional dependencies to use for this run.
        model_settings: Optional settings to use for this model's request.
        usage_limits: Optional limits on model request count or token usage.
        usage: Optional usage to start with, useful for resuming a conversation or agents used in tools.
        metadata: Optional metadata to attach to this run. Accepts a dictionary or a callable taking
            [`RunContext`][pydantic_ai.tools.RunContext]; merged with the agent's configured metadata.
        infer_name: Whether to try to infer the agent name from the call frame if it's not set.
        toolsets: Optional additional toolsets for this run.
        on_complete: Optional callback function called when the agent run completes successfully.
            The callback receives the completed [`AgentRunResult`][pydantic_ai.agent.AgentRunResult] and can access `all_messages()` and other result data.
        manage_system_prompt: Who owns the system prompt. See [`UIAdapter.manage_system_prompt`][pydantic_ai.ui.UIAdapter.manage_system_prompt].
        allowed_file_url_schemes: URL schemes allowed for file URL parts from the client. See
            [`UIAdapter.allowed_file_url_schemes`][pydantic_ai.ui.UIAdapter.allowed_file_url_schemes].

    Returns:
        A streaming Starlette response with AG-UI protocol events.
    """
    return await AGUIAdapter[AgentDepsT].dispatch_request(
        request,
        agent=agent,
        ag_ui_version=ag_ui_version,
        preserve_file_data=preserve_file_data,
        deps=deps,
        output_type=output_type,
        message_history=message_history,
        deferred_tool_results=deferred_tool_results,
        conversation_id=conversation_id,
        model=model,
        model_settings=model_settings,
        usage_limits=usage_limits,
        usage=usage,
        metadata=metadata,
        infer_name=infer_name,
        toolsets=toolsets,
        on_complete=on_complete,
        manage_system_prompt=manage_system_prompt,
        allowed_file_url_schemes=allowed_file_url_schemes,
    )


def run_ag_ui(
    agent: AbstractAgent[AgentDepsT, Any],
    run_input: RunAgentInput,
    accept: str = SSE_CONTENT_TYPE,
    *,
    ag_ui_version: str = DEFAULT_AG_UI_VERSION,
    preserve_file_data: bool = False,
    output_type: OutputSpec[Any] | None = None,
    message_history: Sequence[ModelMessage] | None = None,
    deferred_tool_results: DeferredToolResults | None = None,
    conversation_id: str | None = None,
    model: Model | KnownModelName | str | None = None,
    deps: AgentDepsT = None,
    model_settings: ModelSettings | None = None,
    usage_limits: UsageLimits | None = None,
    usage: RunUsage | None = None,
    metadata: AgentMetadata[AgentDepsT] | None = None,
    infer_name: bool = True,
    toolsets: Sequence[AbstractToolset[AgentDepsT]] | None = None,
    on_complete: OnCompleteFunc[BaseEvent] | None = None,
    manage_system_prompt: Literal['server', 'client'] = 'server',
    allowed_file_url_schemes: frozenset[str] = frozenset({'http', 'https'}),
) -> AsyncIterator[str]:
    """Run the agent with the AG-UI run input and stream AG-UI protocol events.

    Args:
        agent: The agent to run.
        run_input: The AG-UI run input containing thread_id, run_id, messages, etc.
        accept: The accept header value for the run.
        ag_ui_version: AG-UI protocol version controlling thinking/reasoning event format.
        preserve_file_data: Whether to preserve agent-generated files and uploaded files
            in AG-UI message conversion. See [`AGUIAdapter.preserve_file_data`][pydantic_ai.ui.ag_ui.AGUIAdapter.preserve_file_data].

        output_type: Custom output type to use for this run, `output_type` may only be used if the agent has no
            output validators since output validators would expect an argument that matches the agent's output type.
        message_history: History of the conversation so far.
        deferred_tool_results: Optional results for deferred tool calls in the message history.
        conversation_id: ID of the conversation this run belongs to. Pass `'new'` to start a fresh conversation, ignoring any `conversation_id` already on `message_history`. If omitted, falls back to the most recent `conversation_id` on `message_history` or a freshly generated UUID7.
        model: Optional model to use for this run, required if `model` was not set when creating the agent.
        deps: Optional dependencies to use for this run.
        model_settings: Optional settings to use for this model's request.
        usage_limits: Optional limits on model request count or token usage.
        usage: Optional usage to start with, useful for resuming a conversation or agents used in tools.
        metadata: Optional metadata to attach to this run. Accepts a dictionary or a callable taking
            [`RunContext`][pydantic_ai.tools.RunContext]; merged with the agent's configured metadata.
        infer_name: Whether to try to infer the agent name from the call frame if it's not set.
        toolsets: Optional additional toolsets for this run.
        on_complete: Optional callback function called when the agent run completes successfully.
            The callback receives the completed [`AgentRunResult`][pydantic_ai.agent.AgentRunResult] and can access `all_messages()` and other result data.
        manage_system_prompt: Who owns the system prompt. See [`UIAdapter.manage_system_prompt`][pydantic_ai.ui.UIAdapter.manage_system_prompt].
        allowed_file_url_schemes: URL schemes allowed for file URL parts from the client. See
            [`UIAdapter.allowed_file_url_schemes`][pydantic_ai.ui.UIAdapter.allowed_file_url_schemes].

    Yields:
        Streaming event chunks encoded as strings according to the accept header value.
    """
    adapter = AGUIAdapter(
        agent=agent,
        run_input=run_input,
        accept=accept,
        ag_ui_version=ag_ui_version,
        preserve_file_data=preserve_file_data,
        manage_system_prompt=manage_system_prompt,
        allowed_file_url_schemes=allowed_file_url_schemes,
    )
    return adapter.encode_stream(
        adapter.run_stream(
            output_type=output_type,
            message_history=message_history,
            deferred_tool_results=deferred_tool_results,
            conversation_id=conversation_id,
            model=model,
            deps=deps,
            model_settings=model_settings,
            usage_limits=usage_limits,
            usage=usage,
            metadata=metadata,
            infer_name=infer_name,
            toolsets=toolsets,
            on_complete=on_complete,
        ),
    )
