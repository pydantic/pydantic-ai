"""AG-UI adapter for handling requests."""

# pyright: reportGeneralTypeIssues=false, reportInvalidTypeArguments=false

from __future__ import annotations

import json
from dataclasses import dataclass
from http import HTTPStatus
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, ValidationError

from ...tools import AgentDepsT
from .event_stream import (
    AGUIEventStream,
    RunAgentInput,
    StateHandler,
    _AGUIFrontendToolset,  # type: ignore[reportPrivateUsage]
    _InvalidStateError,  # type: ignore[reportPrivateUsage]
    _NoMessagesError,  # type: ignore[reportPrivateUsage]
    _RunError,  # type: ignore[reportPrivateUsage]
    protocol_messages_to_pai_messages,
)

if TYPE_CHECKING:
    from ...agent import Agent

__all__ = ['AGUIAdapter']


@dataclass
class AGUIAdapter:
    """Adapter for handling AG-UI protocol requests with Pydantic AI agents.

    This adapter provides an interface for integrating Pydantic AI agents
    with the AG-UI protocol, handling request parsing, message conversion,
    and event streaming.

    Example:
        ```python
        from pydantic_ai import Agent
        from pydantic_ai.ui.ag_ui import AGUIAdapter

        agent = Agent('openai:gpt-4')
        adapter = AGUIAdapter(agent)

        async def handle_request(request: RunAgentInput, deps=None):
            async for event_str in adapter.run_stream_sse(request, deps):
                yield event_str
        ```
    """

    agent: Agent[AgentDepsT]
    """The Pydantic AI agent to run."""

    async def run_stream(  # noqa: C901
        self,
        request: RunAgentInput,
        deps: AgentDepsT | None = None,
        *,
        output_type: Any = None,
        model: Any = None,
        model_settings: Any = None,
        usage_limits: Any = None,
        usage: Any = None,
        infer_name: bool = True,
        toolsets: Any = None,
        on_complete: Any = None,
    ):
        """Stream events from an agent run as AG-UI protocol events.

        This method provides a complete implementation with all AG-UI features including:
        - Frontend tools handling
        - State injection
        - Error handling (validation vs stream errors)
        - on_complete callback
        - RunStarted and RunFinished events

        Args:
            request: The AG-UI request data.
            deps: Optional dependencies to pass to the agent.
            output_type: Custom output type for this run.
            model: Optional model to use for this run.
            model_settings: Optional settings for the model's request.
            usage_limits: Optional limits on model request count or token usage.
            usage: Optional usage to start with.
            infer_name: Whether to infer the agent name from the call frame.
            toolsets: Optional additional toolsets for this run.
            on_complete: Optional callback called when the agent run completes.

        Yields:
            AG-UI protocol events (BaseEvent subclasses).

        Raises:
            _RunError: If request validation fails or other errors occur.
        """
        from ... import _utils
        from ...exceptions import UserError
        from ...tools import DeferredToolRequests
        from .event_stream import RunFinishedEvent, RunStartedEvent

        # Create event stream
        event_stream = self.create_event_stream()
        stream_started = False

        # Handle frontend tools
        if request.tools:
            toolset = _AGUIFrontendToolset[AgentDepsT](request.tools)
            toolsets = [*toolsets, toolset] if toolsets else [toolset]

        try:
            # Emit start event
            yield RunStartedEvent(
                thread_id=request.thread_id,
                run_id=request.run_id,
            )
            stream_started = True

            if not request.messages:
                raise _NoMessagesError

            # Handle state injection
            raw_state: dict[str, Any] = request.state or {}
            if isinstance(deps, StateHandler):
                if isinstance(deps.state, BaseModel):
                    try:
                        state = type(deps.state).model_validate(raw_state)
                    except ValidationError as e:  # pragma: no cover
                        raise _InvalidStateError from e
                else:
                    state = raw_state

                from dataclasses import replace

                deps = replace(deps, state=state)
            elif raw_state:
                raise UserError(
                    f'AG-UI state is provided but `deps` of type `{type(deps).__name__}` does not implement the `StateHandler` protocol: it needs to be a dataclass with a non-optional `state` field.'
                )

            # Convert AG-UI messages to pAI messages
            messages = protocol_messages_to_pai_messages(request.messages)

            # Run agent and stream events
            result = None
            async for event in self.agent.run_stream_events(
                user_prompt=None,
                output_type=[output_type or self.agent.output_type, DeferredToolRequests],
                message_history=messages,
                model=model,
                deps=deps,
                model_settings=model_settings,
                usage_limits=usage_limits,
                usage=usage,
                infer_name=infer_name,
                toolsets=toolsets,
            ):
                from ...run import AgentRunResultEvent

                # Capture result for on_complete callback
                if isinstance(event, AgentRunResultEvent):
                    result = event.result

                # Transform pAI events to AG-UI events
                async for ag_ui_event in event_stream.agent_event_to_events(event):  # type: ignore[arg-type]
                    yield ag_ui_event

            # Call on_complete callback
            if on_complete is not None and result is not None:
                if _utils.is_async_callable(on_complete):
                    await on_complete(result)
                else:
                    await _utils.run_in_executor(on_complete, result)

        except _RunError as e:
            if stream_started:
                async for error_event in event_stream.on_stream_error(e):
                    yield error_event
            else:
                async for error_event in event_stream.on_validation_error(e):
                    yield error_event
            raise
        except Exception as e:
            if stream_started:
                async for error_event in event_stream.on_stream_error(e):
                    yield error_event
            else:
                async for error_event in event_stream.on_validation_error(e):
                    yield error_event
            raise
        else:
            # Emit finish event
            yield RunFinishedEvent(
                thread_id=request.thread_id,
                run_id=request.run_id,
            )

    async def run_stream_sse(
        self,
        request: RunAgentInput,
        accept: str,
        *,
        output_type: Any = None,
        model: Any = None,
        deps: AgentDepsT | None = None,
        model_settings: Any = None,
        usage_limits: Any = None,
        usage: Any = None,
        infer_name: bool = True,
        toolsets: Any = None,
        on_complete: Any = None,
    ):
        """Stream SSE-encoded events from an agent run.

        This method wraps `run_stream` and encodes the events as SSE strings.

        Args:
            request: The AG-UI request data.
            accept: The accept header value for encoding.
            output_type: Custom output type for this run.
            model: Optional model to use for this run.
            deps: Optional dependencies to pass to the agent.
            model_settings: Optional settings for the model's request.
            usage_limits: Optional limits on model request count or token usage.
            usage: Optional usage to start with.
            infer_name: Whether to infer the agent name from the call frame.
            toolsets: Optional additional toolsets for this run.
            on_complete: Optional callback called when the agent run completes.

        Yields:
            SSE-formatted strings.
        """
        from ag_ui.encoder import EventEncoder

        encoder = EventEncoder(accept=accept)

        try:
            async for event in self.run_stream(
                request=request,
                deps=deps,
                output_type=output_type,
                model=model,
                model_settings=model_settings,
                usage_limits=usage_limits,
                usage=usage,
                infer_name=infer_name,
                toolsets=toolsets,
                on_complete=on_complete,
            ):
                yield encoder.encode(event)
        except _RunError:
            # Error events are already yielded by run_stream
            # This shouldn't actually be reached since run_stream yields error events before raising
            pass
        except Exception:
            # Let other exceptions propagate
            raise

    async def dispatch_request(
        self,
        request: Any,
        deps: AgentDepsT | None = None,
        *,
        output_type: Any = None,
        model: Any = None,
        model_settings: Any = None,
        usage_limits: Any = None,
        usage: Any = None,
        infer_name: bool = True,
        toolsets: Any = None,
        on_complete: Any = None,
    ) -> Any:
        """Handle an AG-UI request and return a streaming response.

        Args:
            request: The incoming Starlette/FastAPI request.
            deps: Optional dependencies to pass to the agent.
            output_type: Custom output type for this run.
            model: Optional model to use for this run.
            model_settings: Optional settings for the model's request.
            usage_limits: Optional limits on model request count or token usage.
            usage: Optional usage to start with.
            infer_name: Whether to infer the agent name from the call frame.
            toolsets: Optional additional toolsets for this run.
            on_complete: Optional callback called when the agent run completes.

        Returns:
            A streaming Starlette response with AG-UI protocol events.
        """
        try:
            from starlette.requests import Request
            from starlette.responses import Response, StreamingResponse
        except ImportError as e:  # pragma: no cover
            raise ImportError('Please install starlette to use dispatch_request') from e

        if not isinstance(request, Request):  # pragma: no cover
            raise TypeError(f'Expected Starlette Request, got {type(request).__name__}')

        accept = request.headers.get('accept', 'text/event-stream')

        try:
            input_data = RunAgentInput.model_validate(await request.json())
        except ValidationError as e:  # pragma: no cover
            return Response(
                content=json.dumps(e.json()),
                media_type='application/json',
                status_code=HTTPStatus.UNPROCESSABLE_ENTITY,
            )

        return StreamingResponse(
            self.run_stream_sse(
                request=input_data,
                accept=accept,
                deps=deps,
                output_type=output_type,
                model=model,
                model_settings=model_settings,
                usage_limits=usage_limits,
                usage=usage,
                infer_name=infer_name,
                toolsets=toolsets,
                on_complete=on_complete,
            ),
            media_type=accept,
        )

    def create_event_stream(self) -> AGUIEventStream[AgentDepsT]:
        """Create a new AG-UI event stream.

        Returns:
            An AGUIEventStream instance.
        """
        return AGUIEventStream[AgentDepsT]()
