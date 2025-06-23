"""Basic example of using pydantic_ai.ag_ui with FastAPI."""

from __future__ import annotations

from typing import TYPE_CHECKING, Annotated

from fastapi import FastAPI, Header
from fastapi.responses import StreamingResponse

from pydantic_ai import Agent
from pydantic_ai.ag_ui import SSE_CONTENT_TYPE, Adapter

if TYPE_CHECKING:
    from ag_ui.core import RunAgentInput

app = FastAPI(title='AG-UI Endpoint')

agent: Agent[None, str] = Agent(
    'openai:gpt-4o-mini',
    instructions='You are a helpful assistant.',
)
adapter: Adapter[None, str] = agent.to_ag_ui()


@app.post('/agent')
async def handler(
    input_data: RunAgentInput, accept: Annotated[str, Header()] = SSE_CONTENT_TYPE
) -> StreamingResponse:
    """Endpoint to handle AG-UI protocol requests and stream responses.

    Args:
        input_data: The AG-UI run input.
        accept: The Accept header to specify the response format.

    Returns:
        A streaming response with event-stream media type.
    """
    return StreamingResponse(
        adapter.run(input_data, accept),
        media_type=SSE_CONTENT_TYPE,
    )


if __name__ == '__main__':
    import uvicorn

    from .cli import Args, parse_args

    args: Args = parse_args()

    uvicorn.run(
        'pydantic_ai.ag_ui_examples.dojo_server:app',
        port=args.port,
        reload=args.reload,
        log_level=args.log_level,
        log_config=args.log_config(),
    )
