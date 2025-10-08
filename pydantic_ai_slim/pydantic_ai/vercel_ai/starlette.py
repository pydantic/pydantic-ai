from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Generic

from pydantic import ValidationError

from ..agent import Agent
from ..run import AgentRunResultEvent
from ..tools import AgentDepsT
from .request_types import RequestData, TextUIPart, request_data_ta
from .response_stream import VERCEL_AI_DSP_HEADERS, DoneChunk, EventStreamer
from .response_types import AbstractSSEChunk

try:
    from sse_starlette.sse import EventSourceResponse
    from starlette.requests import Request
    from starlette.responses import JSONResponse, Response
except ImportError as e:
    raise ImportError('To use Vercel AI Elements, please install starlette and sse_starlette') from e


@dataclass
class StarletteChat(Generic[AgentDepsT]):
    """Starlette support for Pydantic AI's Vercel AI Elements integration.

    This can be used with either FastAPI or Starlette apps.
    """

    agent: Agent[AgentDepsT]

    async def dispatch_request(self, request: Request, deps: AgentDepsT) -> Response:
        """Handle a request and return a streamed SSE response.

        Args:
            request: The incoming Starlette/FastAPI request.
            deps: The dependencies for the agent.

        Returns:
            A streamed SSE response.
        """
        try:
            data = request_data_ta.validate_json(await request.json())

            async def run_sse() -> AsyncIterator[str]:
                async for chunk in self.run(data, deps=deps):
                    yield chunk.sse()

            return EventSourceResponse(run_sse(), headers=VERCEL_AI_DSP_HEADERS)
        except ValidationError as e:
            return JSONResponse({'errors': e.errors()}, status_code=422)
        except Exception as e:
            return JSONResponse({'errors': str(e)}, status_code=500)

    async def run(self, data: RequestData, deps: AgentDepsT = None) -> AsyncIterator[AbstractSSEChunk | DoneChunk]:
        """Stream events from an agent run as Vercel AI Elements events.

        Args:
            data: The data to run the agent with.
            deps: The dependencies to pass to the agent.

        Yields:
            An async iterator text lines to stream over SSE.
        """
        # TODO (DouweM): Use .model and .builtin_tools

        # TODO: Use entire message history

        if not data.messages:
            raise ValueError('no messages provided')

        message = data.messages[-1]
        prompt: list[str] = []
        for part in message.parts:
            if isinstance(part, TextUIPart):
                prompt.append(part.text)
            else:
                raise ValueError(f'Only text parts are supported yet, got {part}')

        event_streamer = EventStreamer()
        async for event in self.agent.run_stream_events('\n'.join(prompt), deps=deps):
            if not isinstance(event, AgentRunResultEvent):
                async for chunk in event_streamer.event_to_chunks(event):
                    yield chunk
        async for chunk in event_streamer.finish():
            yield chunk
