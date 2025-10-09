"""Vercel AI adapter for handling requests."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from functools import cached_property
from typing import (
    Any,
)

from ... import DeferredToolResults
from ...agent import AbstractAgent, AgentDepsT
from ...messages import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    SystemPromptPart,
    TextPart,
    UserPromptPart,
)
from ...models import KnownModelName, Model
from ...output import OutputSpec
from ...settings import ModelSettings
from ...toolsets import AbstractToolset
from ...usage import RunUsage, UsageLimits
from ..adapter import BaseAdapter, OnCompleteFunc
from ..event_stream import BaseEventStream
from ._event_stream import VercelAIEventStream
from ._request_types import RequestData, TextUIPart, UIMessage, request_data_ta
from ._response_types import BaseChunk
from ._utils import VERCEL_AI_DSP_HEADERS

try:
    from starlette.requests import Request
    from starlette.responses import Response
except ImportError as e:  # pragma: no cover
    raise ImportError(
        'Please install the `starlette` package to use `Agent.to_ag_ui()` method, '
        'you can use the `ag-ui` optional group — `pip install "pydantic-ai-slim[ag-ui]"`'
    ) from e


__all__ = ['VercelAIAdapter']


@dataclass
class VercelAIAdapter(BaseAdapter[RequestData, UIMessage, BaseChunk, AgentDepsT]):
    """Adapter for handling Vercel AI protocol requests with Pydantic AI agents.

    This adapter provides a simplified interface for integrating Pydantic AI agents
    with the Vercel AI protocol, handling request parsing, message conversion,
    and event streaming.

    Example:
        ```python
        from pydantic_ai import Agent
        from pydantic_ai.ui.vercel_ai import VercelAIAdapter

        agent = Agent('openai:gpt-4')
        adapter = VercelAIAdapter(agent)

        async def handle_request(data: RequestData, deps=None):
            async for chunk in adapter.run_stream(data, deps):
                yield chunk.sse()
        ```
    """

    def create_event_stream(self) -> BaseEventStream[RequestData, BaseChunk, AgentDepsT]:
        return VercelAIEventStream(self.request)

    def encode_event(self, event: BaseChunk, accept: str | None = None) -> str:
        return f'data: {event.model_dump_json(by_alias=True, exclude_none=True)}\n\n'

    @classmethod
    async def validate_request(cls, request: Request) -> RequestData:
        """Validate a Vercel AI request."""
        return request_data_ta.validate_json(await request.json())

    @classmethod
    async def dispatch_request(
        cls,
        agent: AbstractAgent[AgentDepsT, Any],
        request: Request,
        *,
        message_history: Sequence[ModelMessage] | None = None,
        deferred_tool_results: DeferredToolResults | None = None,
        model: Model | KnownModelName | str | None = None,
        deps: AgentDepsT = None,
        output_type: OutputSpec[Any] | None = None,
        model_settings: ModelSettings | None = None,
        usage_limits: UsageLimits | None = None,
        usage: RunUsage | None = None,
        infer_name: bool = True,
        toolsets: Sequence[AbstractToolset[AgentDepsT]] | None = None,
        on_complete: OnCompleteFunc | None = None,
    ) -> Response:
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
        response = await super().dispatch_request(
            agent,
            request,
            message_history=message_history,
            deferred_tool_results=deferred_tool_results,
            model=model,
            deps=deps,
            output_type=output_type,
            model_settings=model_settings,
            usage_limits=usage_limits,
            usage=usage,
            infer_name=infer_name,
            toolsets=toolsets,
            on_complete=on_complete,
        )
        response.headers.update(VERCEL_AI_DSP_HEADERS)
        return response

    @cached_property
    def messages(self) -> list[ModelMessage]:
        """Convert Vercel AI protocol messages to Pydantic AI messages.

        Args:
            messages: List of Vercel AI UIMessage objects.

        Returns:
            List of Pydantic AI ModelMessage objects.

        Raises:
            ValueError: If message format is not supported.
        """
        pai_messages: list[ModelMessage] = []

        for msg in self.request.messages:
            if msg.role == 'user':
                # User message - extract text from parts
                texts: list[str] = []
                for part in msg.parts:
                    if isinstance(part, TextUIPart):
                        texts.append(part.text)
                    else:
                        raise ValueError(f'Only text parts are supported for user messages, got {type(part).__name__}')

                if texts:
                    pai_messages.append(ModelRequest(parts=[UserPromptPart(content='\n'.join(texts))]))

            elif msg.role == 'assistant':
                # Assistant message - for now, just extract text
                # Full reconstruction of ModelResponse with tool calls would require more complex logic
                texts: list[str] = []
                for part in msg.parts:
                    if isinstance(part, TextUIPart):
                        texts.append(part.text)
                    # TODO: Handle ToolOutputAvailablePart for full message history reconstruction

                if texts:
                    pai_messages.append(ModelResponse(parts=[TextPart(content='\n'.join(texts))]))

            elif msg.role == 'system':
                # System message - not in standard Vercel AI protocol but might be custom
                texts: list[str] = []
                for part in msg.parts:
                    if isinstance(part, TextUIPart):
                        texts.append(part.text)

                if texts:
                    pai_messages.append(ModelRequest(parts=[SystemPromptPart(content='\n'.join(texts))]))

        return pai_messages
