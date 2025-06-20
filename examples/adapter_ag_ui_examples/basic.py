"""Basic example of using adapter_ag_ui with FastAPI."""

from __future__ import annotations

from typing import TYPE_CHECKING, Annotated

from adapter_ag_ui.adapter import AdapterAGUI
from adapter_ag_ui.consts import SSE_ACCEPT
from fastapi import FastAPI, Header
from fastapi.responses import StreamingResponse

from pydantic_ai import Agent

if TYPE_CHECKING:
    from ag_ui.core import RunAgentInput

app = FastAPI(title='AG-UI Endpoint')

agent: Agent[None, str] = Agent(
    'openai:gpt-4o-mini',
    instructions='You are a helpful assistant.',
)
adapter: AdapterAGUI[None, str] = agent.to_ag_ui()


@app.post('/agent')
async def handler(
    input_data: RunAgentInput, accept: Annotated[str, Header()] = SSE_ACCEPT
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
        media_type='text/event-stream',
    )


if __name__ == '__main__':
    import logging

    import uvicorn

    logging.basicConfig(
        format='%(levelname)s: %(message)s',
        level=logging.INFO,
        force=True,
    )

    uvicorn.run(
        'adapter_ag_ui_examples.dojo_server:app',
        host='127.0.0.1',
        port=9000,
        reload=True,
        log_level='info',
    )
