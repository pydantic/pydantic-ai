"""Agentic Chat feature."""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING, Annotated
from zoneinfo import ZoneInfo

from ag_ui.core import RunAgentInput
from fastapi import APIRouter, Header
from fastapi.responses import StreamingResponse

from pydantic_ai.ag_ui import SSE_CONTENT_TYPE

from .agent import AGUIAgent

if TYPE_CHECKING:  # pragma: no cover
    from ag_ui.core import RunAgentInput


router: APIRouter = APIRouter(prefix='/agentic_chat')
agui: AGUIAgent = AGUIAgent()


@agui.agent.tool_plain
async def current_time(timezone: str = 'UTC') -> str:
    """Get the current time in ISO format.

    Args:
        timezone: The timezone to use.

    Returns:
        The current time in ISO format string.
    """
    tz: ZoneInfo = ZoneInfo(timezone)
    return datetime.now(tz=tz).isoformat()


@router.post('')
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
        agui.adapter.run(input_data, accept),
        media_type=SSE_CONTENT_TYPE,
    )
