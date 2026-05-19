"""Tool approval with AG-UI interrupts.

Demonstrates the AG-UI interrupt lifecycle: a tool declared with `requires_approval=True`
pauses the run when the model proposes a call. The adapter emits `RUN_FINISHED` with
`outcome.type == "interrupt"`; the client renders an approval UI from `outcome.interrupts[]`
and posts a follow-up `RunAgentInput` carrying `resume[]` to approve, deny, or edit the call.

Requires `ag-ui-protocol >= 0.1.19`. See https://docs.ag-ui.com/concepts/interrupts.
"""

from __future__ import annotations

from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import Response
from starlette.routing import Route

from pydantic_ai import Agent
from pydantic_ai.tools import DeferredToolRequests
from pydantic_ai.ui.ag_ui import AGUIAdapter

# `output_type` must include `DeferredToolRequests` so the run can pause on a pending approval
# instead of erroring when the model proposes a `requires_approval=True` tool.
agent = Agent('openai:gpt-5-mini', output_type=[str, DeferredToolRequests])


@agent.tool_plain(requires_approval=True)
def delete_file(path: str) -> str:
    """Delete a file. The run pauses here and waits for the user to approve before executing."""
    # Real implementation would actually delete the file.
    return f'deleted {path}'


async def run_agent(request: Request) -> Response:
    # `ag_ui_version='0.1.19'` is needed until ag-ui-protocol cuts the 0.1.19 release: the
    # unreleased SDK installed from `main` still reports `version='0.1.18'` even though it
    # ships the interrupt types, so the adapter's default version gate blocks `outcome` emission.
    # Drop this override once `pip show ag-ui-protocol` reports `0.1.19` or later.
    return await AGUIAdapter.dispatch_request(
        request, agent=agent, ag_ui_version='0.1.19'
    )


app = Starlette(routes=[Route('/', run_agent, methods=['POST'])])
