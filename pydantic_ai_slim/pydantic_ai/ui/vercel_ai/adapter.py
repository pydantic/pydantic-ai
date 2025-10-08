"""Vercel AI adapter for handling requests."""

# pyright: reportGeneralTypeIssues=false

from __future__ import annotations

from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from ...tools import AgentDepsT
from .event_stream import VercelAIEventStream
from .request_types import RequestData, TextUIPart, UIMessage
from .response_types import AbstractSSEChunk, DoneChunk

if TYPE_CHECKING:
    from ...agent import Agent

__all__ = ['VercelAIAdapter']


@dataclass
class VercelAIAdapter:
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

    agent: Agent[AgentDepsT]
    """The Pydantic AI agent to run."""

    async def run_stream(
        self, request: RequestData, deps: AgentDepsT | None = None
    ) -> AsyncIterator[AbstractSSEChunk | DoneChunk]:
        """Stream events from an agent run as Vercel AI protocol events.

        Args:
            request: The Vercel AI request data.
            deps: Optional dependencies to pass to the agent.

        Yields:
            Vercel AI protocol events (AbstractSSEChunk or DoneChunk).

        Raises:
            ValueError: If request validation fails.
        """
        # Parse messages from request
        messages = self.parse_request_messages(request)

        # Extract prompt from last user message (for now, simple implementation)
        if not messages:
            raise ValueError('no messages provided')

        message = messages[-1]
        prompt_parts: list[str] = []
        for part in message.parts:
            if isinstance(part, TextUIPart):
                prompt_parts.append(part.text)
            else:
                raise ValueError(f'Only text parts are supported yet, got {part}')

        # Create event stream
        event_stream = self.create_event_stream()

        # Run agent and stream events
        async for event in self.agent.run_stream_events('\n'.join(prompt_parts), deps=deps):
            async for chunk in event_stream.agent_event_to_events(event):
                yield chunk

        # Emit after-stream events
        async for chunk in event_stream.after_stream():
            yield chunk

    def create_event_stream(self) -> VercelAIEventStream[AgentDepsT]:
        """Create a new Vercel AI event stream.

        Returns:
            A VercelAIEventStream instance.
        """
        return VercelAIEventStream[AgentDepsT]()

    def parse_request_messages(self, request: RequestData) -> list[UIMessage]:
        """Extract messages from the Vercel AI request.

        Args:
            request: The Vercel AI request data.

        Returns:
            List of UIMessage objects.
        """
        return request.messages

    async def dispatch_request(self, request: Any, deps: AgentDepsT | None = None) -> Any:
        """Handle a request and return a streamed SSE response.

        Args:
            request: The incoming Starlette/FastAPI request.
            deps: The dependencies for the agent.

        Returns:
            A streamed SSE response.
        """
        try:
            from starlette.requests import Request
            from starlette.responses import JSONResponse
        except ImportError as e:  # pragma: no cover
            raise ImportError('Please install starlette to use dispatch_request') from e

        try:
            from sse_starlette.sse import EventSourceResponse
        except ImportError as e:  # pragma: no cover
            raise ImportError('Please install sse_starlette to use dispatch_request') from e

        from pydantic import ValidationError

        if not isinstance(request, Request):  # pragma: no cover
            raise TypeError(f'Expected Starlette Request, got {type(request).__name__}')

        from .request_types import request_data_ta

        try:
            data = request_data_ta.validate_json(await request.json())

            async def run_sse() -> AsyncIterator[str]:
                async for chunk in self.run_stream(data, deps=deps):
                    yield chunk.sse()

            from ._utils import VERCEL_AI_DSP_HEADERS

            return EventSourceResponse(run_sse(), headers=VERCEL_AI_DSP_HEADERS)
        except ValidationError as e:
            return JSONResponse({'errors': e.errors()}, status_code=422)
        except Exception as e:
            return JSONResponse({'errors': str(e)}, status_code=500)
