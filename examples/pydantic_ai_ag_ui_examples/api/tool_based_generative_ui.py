"""Tool Based Generative UI feature.

No special handling is required for this feature.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Annotated

from ag_ui.core import RunAgentInput
from fastapi import APIRouter, Header
from fastapi.responses import StreamingResponse

from pydantic_ai.ag_ui import SSE_CONTENT_TYPE

from .agent import AGUIAgent

if TYPE_CHECKING:  # pragma: no cover
    from ag_ui.core import RunAgentInput


router: APIRouter = APIRouter(prefix='/tool_based_generative_ui')
agui: AGUIAgent = AGUIAgent()


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
