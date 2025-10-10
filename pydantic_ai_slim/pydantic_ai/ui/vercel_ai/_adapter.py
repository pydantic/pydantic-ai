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
    """TODO (DouwM): Docstring."""

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
            agent: The agent to run.
            request: The incoming Starlette/FastAPI request.

            output_type: Custom output type to use for this run, `output_type` may only be used if the agent has no
                output validators since output validators would expect an argument that matches the agent's output type.
            message_history: History of the conversation so far.
            deferred_tool_results: Optional results for deferred tool calls in the message history.
            model: Optional model to use for this run, required if `model` was not set when creating the agent.
            deps: Optional dependencies to use for this run.
            model_settings: Optional settings to use for this model's request.
            usage_limits: Optional limits on model request count or token usage.
            usage: Optional usage to start with, useful for resuming a conversation or agents used in tools.
            infer_name: Whether to try to infer the agent name from the call frame if it's not set.
            toolsets: Optional additional toolsets for this run.
            on_complete: Optional callback function called when the agent run completes successfully.
                The callback receives the completed [`AgentRunResult`][pydantic_ai.agent.AgentRunResult] and can access `all_messages()` and other result data.

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

    # TODO (DouweM): model, builtin_tools?

    # TODO (DouweM): static load_messages, dump_messages
