"""Human in the Loop Feature.

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


instructions: str = """When planning tasks use tools only, without any other messages.
IMPORTANT:
- Use the `generate_task_steps` tool to display the suggested steps to the user
- Never repeat the plan, or send a message detailing steps
- If accepted, confirm the creation of the plan and the number of selected (enabled) steps only
- If not accepted, ask the user for more information, DO NOT use the `generate_task_steps` tool again
"""
router: APIRouter = APIRouter(prefix='/human_in_the_loop')
agui: AGUIAgent = AGUIAgent(instructions=instructions)


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
