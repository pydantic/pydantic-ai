from dataclasses import dataclass
from typing import Generic

from pydantic import ValidationError

from ..agent import Agent
from ..tools import AgentDepsT
from .request_types import RequestData, TextUIPart, request_data_schema
from .response_stream import VERCEL_AI_ELEMENTS_HEADERS, sse_stream

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
        body = await request.body()
        try:
            data = request_data_schema.validate_json(body)
        except ValidationError as e:
            return JSONResponse({'errors': e.errors()}, status_code=422)
        else:
            return await self.handle_request_data(data, deps)

    async def handle_request_data(self, data: RequestData, deps: AgentDepsT) -> Response:
        """Handle request data that has already been validated and return a streamed SSE response.

        Args:
            data: The validated request data.
            deps: The dependencies for the agent.

        Returns:
            A streamed SSE response.
        """
        if not data.messages:
            return JSONResponse({'errors': 'no messages provided'})

        message = data.messages[-1]
        prompt: list[str] = []
        for part in message.parts:
            if isinstance(part, TextUIPart):
                prompt.append(part.text)
            else:
                return JSONResponse({'errors': 'only text parts are supported yet'})

        return EventSourceResponse(
            sse_stream(self.agent, '\n'.join(prompt), deps=deps), headers=VERCEL_AI_ELEMENTS_HEADERS
        )
